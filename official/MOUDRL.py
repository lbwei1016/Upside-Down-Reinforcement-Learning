"""## Implementation

### Replay Utilities
"""

import copy
import os
import pickle
from abc import ABC, abstractmethod

import numpy as np
from sortedcontainers import SortedListWithKey

from torch.utils.tensorboard import SummaryWriter

from util.config import Config


writer_train = SummaryWriter("./tb_record/train")
writer_eval = SummaryWriter("./tb_record/eval")


class Episode:
    """
    For any episode, this container has len(actions) == len(rewards) == len(states) - 1
    This is because we initialize using the starting state.
    The add() method adds the action just taken, the obtained reward, and the **next** state.

    This makes accessing the episode data simple:
    states[0] is the first state
    actions[0] is the action taken in that state
    rewards[0] reward obtained by taking the action, and so on

    The last state added is never actually processed by the agent.
    """

    def __init__(self, init_state, desired_return, desired_horizon):
        self.states = [init_state]
        self.actions = []
        self.rewards = []
        self.desired_return = desired_return
        self.desired_horizon = desired_horizon

    def add(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    @property
    def total_reward(self):
        return sum(self.rewards)

    @property
    def steps(self):
        return len(self.actions)

    @property
    def return_gap(self):
        return self.desired_return - self.total_reward

    @property
    def horizon_gap(self):
        return self.desired_horizon - self.steps


def get_reward(episode: Episode):
    return episode.total_reward


def make_replay(config: Config):
    if config.replay == "highest":
        replay = HighestReplay(max_size=config.replay_size)
    # elif config.replay == "recent":
    #     replay = RecentReplay(max_size=config.replay_size)
    else:
        raise NotImplementedError
    return replay


class Replay(ABC):
    def __init__(self):
        self.episodes: List[Episode] = []
        self.known_returns = []
        self.known_horizons = []

    @abstractmethod
    def add(self, episode):
        raise NotImplementedError

    @property
    def best_episode(self):
        return max(self.episodes, key=get_reward)

    def get_closest_horizon(self, desired_return):
        idx = np.abs(np.asarray(self.known_returns) - desired_return).argmin()
        return self.known_horizons[idx]

    @property
    def returns(self):
        return [episode.total_reward for episode in self.episodes]


class HighestReplay(Replay):
    def __init__(self, max_size: int):
        super().__init__()
        self.episodes = SortedListWithKey(key=get_reward)
        self.max_size = max_size

    def add(self, episode: Episode):
        self.episodes.add(episode)
        self.known_returns.append(episode.total_reward)
        self.known_horizons.append(episode.steps)
        if len(self.episodes) > self.max_size:
            self.episodes.pop(0)


"""TODO
A new method class for replay (support Pareto front or others)
"""


def trailing_segments(episode: Episode, nprnd: np.random.RandomState):
    steps = episode.steps
    i = nprnd.randint(0, steps)
    j = steps
    return episode.states[i], sum(episode.rewards[i:j]), (j - i), episode.actions[i]


def sample_batch(replay: Replay, batch_size: int, nprnd: np.random.RandomState):
    idxs = nprnd.randint(0, len(replay.episodes), batch_size)
    episodes = [replay.episodes[idx] for idx in idxs]
    segments = [trailing_segments(episode, nprnd) for episode in episodes]

    states, desired_rewards, horizons, actions = [], [], [], []
    for state, desired_reward, horizon, action in segments:
        states.append(state)
        desired_rewards.append(desired_reward)
        horizons.append(horizon)
        actions.append(action)

    states = np.array(states, dtype=np.float32)
    desired_rewards = np.array(desired_rewards, dtype=np.float32)[:, None]
    horizons = np.array(horizons, dtype=np.float32)[:, None]
    actions = np.array(actions, dtype=np.float32)

    return states, desired_rewards, horizons, actions


"""### Behavior Function"""

import numpy as np
import torch
import torch.nn as nn

from torch.nn.init import orthogonal_
import torch.nn.functional as F


class Categorical:
    def __init__(self, dim: int):
        self.dim = dim
        self.loss = nn.CrossEntropyLoss(reduction="mean")

    @staticmethod
    def distribution(probs):
        return torch.distributions.Categorical(probs)

    def sample(self, scores):
        probs = F.softmax(scores, dim=1)
        dist = self.distribution(probs)
        sample = dist.sample().item()  # item() forces single env
        return sample

    def mode(self, scores):
        probs = F.softmax(scores, dim=1)
        mode = probs.to("cpu").data.numpy()[0].argmax()  # [0] forces single env
        return mode

    def random_sample(self):
        return torch.randint(0, self.dim, (1,)).item()  # (1,) & item() force single env

    def clip(self, action):
        return action


class ScaledIntent:
    """TODO
    If there are multiple objectives, the must be multiple scales, which is cumbersome.
    Try to get rid of the scale.
    """

    def __init__(self, return_scale: float, horizon_scale: float, max_return: float):
        self.return_scale = return_scale
        self.horizon_scale = horizon_scale
        self.max_return = max_return

    """TODO
    Support multiple returns
    """

    def __call__(self, intent):
        _intent = np.zeros_like(intent)
        returns = np.minimum(intent[:, 0], self.max_return)
        horizons = np.maximum(intent[:, 1], 1)
        _intent[:, 0] = returns * self.return_scale
        _intent[:, 1] = horizons * self.horizon_scale
        intent = _intent.astype(np.float32)
        return intent


def make_behavior_fn(
    config: Config, nprnd: np.random.RandomState, device: torch.device
):
    intent_transform = ScaledIntent(
        config.return_scale, config.horizon_scale, config.env_max_return
    )
    behavior_fn = BehaviorFn(
        ProductNetwork(config),
        intent_transform=intent_transform,
        config=config,
        nprnd=nprnd,
        device=device,
    )
    return behavior_fn


class BehaviorFn(nn.Module):
    def __init__(
        self,
        net: nn.Module,
        intent_transform: ScaledIntent,
        config: Config,
        nprnd: np.random.RandomState,
        device="cpu",
    ):
        super().__init__()
        self.net = net
        self.intent_transform = intent_transform
        self.nprnd = nprnd
        self.device = device
        self.state_dtype = np.float32

        if config.action_type == "discrete":
            self.action_dist = Categorical(config.n_action)
            self.action_dtype = np.int64
        else:
            raise NotImplementedError(config.action_type)

    def forward(self, state, desired_reward, horizon, device=None):
        """TODO
        Support multiple rewards (esp. np.concat).
        """
        if device is None:
            device = self.device
        state = np.asarray(state)
        intent = np.concatenate([desired_reward, horizon], axis=1)
        transformed_intent = self.intent_transform(intent)

        state = self.make_variable(state, dtype=self.state_dtype, device=device)
        intent = self.make_variable(
            transformed_intent, dtype=self.state_dtype, device=device
        )
        net_output = self.net(state, intent)
        if hasattr(self, "logstd"):
            net_output = torch.cat(
                (net_output, self.logstd.expand_as(net_output)), dim=-1
            )
        return net_output

    def loss(self, states, desired_rewards, horizons, actions) -> torch.Tensor:
        outputs = self(states, desired_rewards, horizons)
        targets = self.make_variable(actions, dtype=self.action_dtype)
        loss = self.action_dist.loss(outputs, targets)
        return loss

    def make_variable(self, x, dtype, device=None):
        """
        Construct a `torch.Tensor` from the input `x`.
        """
        if device is None:
            device = self.device
        return torch.from_numpy(np.asarray(x, dtype=dtype)).to(device)


class ProductNetwork(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        activation, n_state, n_output, net_arch = (
            config.activation,
            config.n_state,
            config.n_action,
            config.net_arch,
        )

        self.dropout = config.dropout

        self.n_state = n_state
        self.net_option = config.net_option
        n_proj = net_arch[0]

        if activation == "relu":
            activation = nn.ReLU
            gain = np.sqrt(2)
        elif activation == "tanh":
            activation = nn.Tanh
            gain = 1.0
        else:
            raise NotImplementedError

        if self.dropout:
            dropout = nn.Dropout(p=self.dropout)

        self.layer1 = FastWeightLayer(
            n_proj, n_state, 2, activation, option=self.net_option
        )
        hids = []
        n_last = n_proj
        for n_current in net_arch[1:]:
            if self.dropout:
                hids += [nn.Linear(n_last, n_current), activation(), dropout]
            else:
                hids += [nn.Linear(n_last, n_current), activation()]
            n_last = n_current
        self.hids = nn.Sequential(*hids)

        self.op = nn.Linear(n_last, n_output)

        self.init_params(gain)

    def forward(self, state, intent):
        out = self.layer1(state, intent)
        out = self.hids(out) if len(self.hids) > 0 else out
        out = self.op(out)
        return out

    def init_params(self, gain):
        def init(m):
            if type(m) == nn.Linear:
                orthogonal_(m.weight.data, gain=gain)
                m.bias.data.fill_(0.0)

        def init_hyper(m):
            if type(m) == nn.Linear:
                orthogonal_(m.weight.data, gain=1.0)
                m.bias.data.fill_(0.0)

        self.layer1.apply(init_hyper)
        self.hids.apply(init)


class FastWeightLayer(nn.Module):
    def __init__(self, size, x_size, c_size, activation, option):
        super().__init__()
        self.size = size
        self.x_size = x_size
        self.c_size = c_size
        self.option = option
        self.activation = activation

        if option == "bilinear":
            self.Wlinear = nn.Linear(c_size, self.size * self.x_size)
            self.blinear = nn.Linear(c_size, self.size)
        elif option == "gated":
            self.xlinear = nn.Linear(x_size, size)
            self.clinear = nn.Linear(c_size, size)
        else:
            raise NotImplementedError(option)

    def forward(self, x, c):
        if self.option == "bilinear":
            batch_size = x.shape[0]
            W, b = self.Wlinear(c), self.blinear(c)
            W = torch.reshape(W, (batch_size, self.x_size, self.size))
            x = torch.reshape(
                x, (batch_size, 1, self.x_size)
            )  # add a dimension for matmul, then remove it
            output = self.activation()(
                torch.matmul(x, W).reshape((batch_size, self.size)) + b
            )
        elif self.option == "gated":
            x_proj = self.activation()(self.xlinear(x))
            c_proj = torch.sigmoid(self.clinear(c))
            output = x_proj * c_proj
        else:
            raise NotImplementedError(self.option)
        return output


"""### Agent Implementation"""

from collections import deque
from typing import List, Tuple, Union

import gym
import numpy as np
import torch
from gym.core import Wrapper
from tqdm.notebook import trange


class SeedEnv(Wrapper):
    """Every reset() set a new seed from a given seed range"""

    def __init__(self, env, seed_range):
        super().__init__(env)
        self.seed_range = seed_range

    def reset(self, **kwargs):
        self.env.seed(np.random.randint(*self.seed_range))
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)


def get_stats(scalar_list: list) -> dict:
    if len(scalar_list) == 0:
        stats = {key: np.nan for key in ("max", "mean", "median", "min", "std")}
        stats["size"] = 0
    else:
        stats = {
            "max": np.max(scalar_list),
            "mean": np.mean(scalar_list),
            "median": np.median(scalar_list),
            "min": np.min(scalar_list),
            "std": np.std(scalar_list, ddof=1),
            "size": len(scalar_list),
        }
    return stats


class UpsideDownAgent:

    def __init__(self, config: Config):

        self.config = config
        self.msg = print if config.verbose else lambda *a, **k: None
        self.device = torch.device(
            "cuda:0" if config.use_gpu and torch.cuda.is_available() else "cpu"
        )
        self.msg("Using device", self.device)

        seed = config.seed
        self.nprnd = np.random.RandomState(seed)
        np.random.seed(Config.seed)
        torch.manual_seed(seed)

        """TODO
        Support mo_gym.
        """
        self.train_env = SeedEnv(
            gym.make(config.env_name), seed_range=config.train_seeds
        )
        self.test_env = SeedEnv(gym.make(config.env_name), seed_range=config.eval_seeds)

        self.replay: Replay = make_replay(config)
        self.behavior_fn: BehaviorFn = make_behavior_fn(config, self.nprnd, self.device)
        self.optimizer = torch.optim.Adam(
            self.behavior_fn.parameters(), lr=config.learning_rate
        )

        self.iters = 0
        self.total_episodes = 0
        self.total_steps = 0
        self.best_onpolicy_mean = np.array(-np.inf)
        self.best_greedy_mean = np.array(-np.inf)
        self.best_rolling_mean = np.array(-np.inf)
        self.rolling_returns = deque(maxlen=config.n_eval_episodes)

        self.current_step_limit = (
            config.warmup_step_limit
        )  # Used only for warm up inputs
        self.current_desired_return = (config.warmup_desired_return, 0)

    def warm_up(self) -> List[Tuple]:

        results: List[Tuple] = []

        episodes, _ = self.run_episodes(
            self.current_step_limit,
            self.current_desired_return,
            label="Warmup",
            actions="random",
            n_episodes=self.config.n_warm_up_episodes,
        )
        self.total_episodes += self.config.n_warm_up_episodes
        for episode in episodes:
            self.replay.add(episode)
            self.rolling_returns.append(episode.total_reward)

        stats = get_stats(self.replay.returns)
        self.msg(
            f"\nWarmup | Replay max: {stats['max']:7.2f} mean: {stats['mean']:7.2f} "
            f"min: {stats['min']:7.2f} size: {stats['size']:3}"
        )
        results += [
            ("replay." + k, stats[k], self.total_steps)
            for k in ["max", "mean", "min", "size"]
        ]

        return results

    def train_step(self) -> List[Tuple]:

        results: List[Tuple] = []
        n_updates = self.config.n_updates_per_iter
        self.msg(f"\nIteration {(self.iters + 1):3} | Training for {n_updates} updates")

        ##### Learn behavior function #####
        torch.set_grad_enabled(True)
        self.behavior_fn.to(self.device)
        self.behavior_fn.train()
        loss = None
        tq = trange(n_updates, disable=self.config.verbose is not True)
        losses = []
        for _ in tq:
            self.optimizer.zero_grad()
            s, r, h, a = sample_batch(self.replay, self.config.batch_size, self.nprnd)
            loss = self.behavior_fn.loss(s, r, h, a)
            losses.append(loss.item())
            tq.set_postfix(loss=loss.item())
            loss.backward()
            self.optimizer.step()

        results += [("loss", loss.item(), self.total_steps)]
        ##### End of learning BF #####

        ##### Generate more data #####
        # The replay buffer is sorted, such that the `last_few` episodes are of the highest returns.
        last_few_episodes = self.replay.episodes[-self.config.last_few :]
        last_few_durations = [episodes.steps for episodes in last_few_episodes]
        last_few_returns = [episodes.total_reward for episodes in last_few_episodes]
        self.current_step_limit = int(np.mean(last_few_durations))
        self.current_desired_return = (
            np.mean(last_few_returns),
            np.std(last_few_returns),
        )
        ##### End of generating data #####

        """TODO
        Change how the statistics of returns are calculated.
        """

        episodes, eval_results = self.run_episodes(
            self.current_step_limit,
            self.current_desired_return,
            label="Train",
            actions=self.config.actions,
            n_episodes=self.config.n_episodes_per_iter,
        )
        self.total_episodes += self.config.n_episodes_per_iter

        returns = []
        for episode in episodes:
            self.replay.add(episode)
            episode_return = episode.total_reward
            returns.append(episode_return)
            self.rolling_returns.append(episode_return)
        del episodes

        # Logging
        self.iters += 1
        results += eval_results

        rolling_mean = np.mean(self.rolling_returns)
        results += [("rollouts.rolling_mean", rolling_mean, self.total_steps)]
        self.best_rolling_mean = (
            rolling_mean
            if rolling_mean > self.best_rolling_mean
            else self.best_rolling_mean
        )

        stats = get_stats(self.replay.returns)
        self.msg(
            f"Iteration {self.iters:3} | "
            f"Rollouts max: {np.max(returns):7.2f} mean: {np.mean(returns):7.2f} min: {np.min(returns):7.2f} | "
            f"Replay max: {stats['max']:7.2f} mean: {stats['mean']:7.2f} "
            f"min: {stats['min']: 7.2f} size: {stats['size']:3} | "
            f"steps so far: {self.total_steps:7} episodes so far: {self.total_episodes:6} | "
            f"Rolling Mean ({len(self.rolling_returns)}): {rolling_mean:7.2f}"
        )

        results += [("iteration", self.iters, self.total_steps)]
        results += [("current_step_limit", self.current_step_limit, self.total_steps)]
        results += [
            ("current_desired_return.mean", np.mean(last_few_returns), self.total_steps)
        ]
        results += [
            ("current_desired_return.std", np.std(last_few_returns), self.total_steps)
        ]
        results += [
            ("replay." + k, stats[k], self.total_steps)
            for k in ["max", "mean", "min", "size"]
        ]

        stats = get_stats(returns)
        results += [
            ("rollouts." + k, stats[k], self.total_steps)
            for k in ["max", "mean", "min"]
        ]
        results += [("total_steps", self.total_steps, self.total_steps)]

        ##### Tensorboard logging #####
        writer_train.add_scalar("loss", loss.item(), self.total_steps)
        writer_train.add_scalar("total steps", self.total_steps, self.iters)
        writer_train.add_scalar("total episodes", self.total_episodes, self.iters)
        writer_train.add_scalar(
            "desired return (mean)", np.mean(last_few_returns), self.total_steps
        )
        writer_train.add_scalar(
            "desired return (std)", np.std(last_few_returns), self.total_steps
        )
        writer_train.add_scalar(
            "desired horizon (mean)", np.mean(last_few_durations), self.total_steps
        )
        writer_train.add_scalar(
            "desired horizon (std)", np.std(last_few_durations), self.total_steps
        )
        writer_train.add_scalar(
            f"rolling mean return ({self.config.n_eval_episodes} episodes)",
            np.mean(self.rolling_returns),
            self.iters,
        )
        ##### End of TB #####

        return results

    def _eval(self) -> List[Tuple]:

        results: List[Tuple] = [("episodes", self.total_episodes, self.total_steps)]
        if self.config.eval_goal == "max":
            desired_test_return = self.config.env_max_return
        elif self.config.eval_goal == "current":
            desired_test_return = self.current_desired_return[0]
        else:
            raise NotImplementedError

        actions = "on_policy"  # can also evaluation with "greedy" actions here

        self.msg(
            f"\nTesting on {self.config.n_eval_episodes} episodes with {actions} actions"
        )
        episodes, _ = self.run_episodes(
            self.current_step_limit,
            desired_test_return,
            label="Test",
            actions=actions,
            n_episodes=self.config.n_eval_episodes,
        )
        stats = get_stats([episode.total_reward for episode in episodes])
        results += [
            (f"eval.{actions}.{k}", stats[k], self.total_steps)
            for k in ["max", "median", "mean", "std", "min"]
        ]
        print(
            f"Eval | {actions} | max: {stats['max']:7.2f} | median: {stats['median']:7.2f} | "
            f"mean: {stats['mean']:7.2f} | std: {stats['std']: 7.2f} | min: {stats['min']:7.2f} | "
            f"steps so far: {self.total_steps:7} | episodes so far: {self.total_episodes:6}"
        )

        ##### Tensorbaord #####
        writer_eval.add_scalar("returns max", stats["max"], self.total_steps)
        writer_eval.add_scalar("returns min", stats["min"], self.total_steps)
        writer_eval.add_scalar(
            f"returns mean ({self.config.n_eval_episodes} episodes)",
            stats["mean"],
            self.total_steps,
        )
        writer_eval.add_scalar(
            f"returns std ({self.config.n_eval_episodes} episodes)",
            stats["std"],
            self.total_steps,
        )
        writer_eval.add_scalar("total steps", self.total_steps, self.iters)
        writer_eval.add_scalar("total episodes", self.total_episodes, self.iters)
        ##### End of TB #####

        if actions == "on_policy":
            self.best_onpolicy_mean = max(stats["mean"], self.best_onpolicy_mean)
        else:
            self.best_greedy_mean = max(stats["mean"], self.best_greedy_mean)
        del episodes, stats

        return results

    def run_episodes(
        self,
        step_limit: int,
        desired_return: Union[float, Tuple],
        label: str,
        actions: str,
        n_episodes: int = 1,
        render: bool = False,
    ) -> Tuple[List[Episode], List[Tuple]]:

        assert label in ["Warmup", "Train", "Test"]
        assert actions in ["random", "on_policy", "greedy"] or actions.startswith(
            "epsg"
        )

        behavior_fn, config, device, nprnd = (
            self.behavior_fn,
            self.config,
            self.device,
            self.nprnd,
        )

        torch.set_grad_enabled(False)
        behavior_fn.eval()
        behavior_fn.to(device)
        episodes: List[Episode] = []
        eval_results: List[Tuple] = []
        env = self.test_env if label == "Test" else self.train_env

        """TODO
        Check desired_return_final (compatible for multiple objectives?)
        """
        for i in range(n_episodes):
            if isinstance(desired_return, tuple):
                desired_return_final = (
                    desired_return[0] + desired_return[1] * nprnd.random_sample()
                )
            else:
                desired_return_final = desired_return
            if (
                config.env_name == "TakeCover-v0"
                or config.env_name == "CartPoleContinuous-v0"
            ):
                desired_return_final = int(desired_return_final)
                step_limit = desired_return_final

            # Prepare env
            state = env.reset()
            if render:
                env.render()

            # Generate episode
            episode = Episode(state, desired_return_final, step_limit)
            done = False

            while episode.steps < config.env_step_limit and not done:
                state = np.asarray(state)

                if actions == "random":
                    action = behavior_fn.action_dist.random_sample()
                elif actions == "on_policy":
                    desired_return_remaining = np.array(
                        [[desired_return_final - episode.total_reward]]
                    )
                    steps_remaining = np.array([[step_limit - episode.steps]])
                    action_scores = behavior_fn(
                        state[None],
                        desired_return_remaining,
                        steps_remaining,
                        device=device,
                    )
                    action = behavior_fn.action_dist.sample(action_scores)
                else:
                    raise NotImplementedError

                clipped_action = behavior_fn.action_dist.clip(action)
                state, reward, done, _ = env.step(clipped_action)
                if render:
                    env.render()

                if label == "Test":
                    episode.add(0, 0, reward)  # reduce memory usage
                else:
                    episode.add(state, clipped_action, reward)
                    self.total_steps += 1
                    if (
                        label == "Train"
                        and self.total_steps % self.config.eval_freq == 0
                    ):
                        eval_results += self._eval()

            self.msg(
                f"{label} | {actions} | Episode {i:3} | "
                f"Goals: ({desired_return_final:7.2f}, {step_limit:4}) | "
                f"Return: {episode.total_reward:7.2f} Steps: {episode.steps:4} | "
                f"Return gap: {episode.return_gap:7.2f} Horizon gap: {episode.horizon_gap:5} "
            )

            episodes.append(episode)

        return episodes, eval_results


"""# Run Experiment"""

if __name__ == "__main__":
    ud = UpsideDownAgent(Config)
    print("Using device", ud.device)

    ud.warm_up()
    print(f"Warm-up complete. Starting training.")
    eval_means, eval_medians = [], []

    while ud.total_steps < Config.max_training_steps:
        results = ud.train_step()
        for r in results:
            if r[0] == "eval.on_policy.mean":
                eval_means.append(r[1])
            if r[0] == "eval.on_policy.median":
                eval_medians.append(r[1])
        ud.msg(f"Iteration {ud.iters} complete\n")
