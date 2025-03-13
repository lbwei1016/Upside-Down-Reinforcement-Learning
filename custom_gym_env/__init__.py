from gym.envs.registration import register

register(
    id='custom_gym_env/CartPoleContinuous-v0',
    entry_point='custom_gym_env.envs:ContinuousCartPoleEnv',
    max_episode_steps=200,
)