from dataclasses import dataclass


@dataclass
class Config:
    # environment
    env_name = "LunarLander-v2"
    n_state = 8
    n_action = 4
    train_dir = "."
    env_max_return = 300
    env_step_limit = 1000
    clip_limits = None
    warmup_desired_return = 0
    warmup_step_limit = 100

    # agent
    net_option = "bilinear"
    net_arch = (64, 128, 128)
    action_type = "discrete"
    activation = "relu"
    return_scale = 0.015
    horizon_scale = 0.03

    # training & testing
    replay = "highest"
    replay_size = 600
    n_warm_up_episodes = 50
    n_episodes_per_iter = 20
    last_few = 100
    learning_rate = 0.0008709635899560805
    batch_size = 768
    n_updates_per_iter = 150
    max_training_steps = 10_000_000
    eval_freq = 50_000
    eval_goal = "current"

    train_alg = "udrl"
    train_seeds = (1_000_000, 10_000_000)
    eval_seeds = (1, 500_000)
    n_eval_episodes = 100
    actions = "on_policy"
    save_model = True
    verbose = False
    use_gpu = False
    ### Is seems that using GPU is slower...
    # use_gpu = True
    seed = 9

    ##### Regularization #####
    dropout: float = None
