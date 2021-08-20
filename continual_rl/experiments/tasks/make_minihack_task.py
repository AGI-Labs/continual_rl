import gym
import numpy as np

from .image_task import ImageTask


class MiniHackObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # obs_space = env.observation_space['pixel_crop']
        self.observation_space = gym.spaces.Box(low=0, high=255, dtype=np.uint8, shape=(84, 84, 3))

    def observation(self, obs):
        obs = obs["pixel_crop"]
        obs = np.pad(obs, [(2, 2), (2, 2), (0, 0)])
        return obs


def make_minihack(env_name, observation_keys=["pixel_crop"], **kwargs):
    import minihack

    env = gym.make(f"MiniHack-{env_name}", observation_keys=observation_keys, **kwargs)  # each env specifies its own self._max_episode_steps
    env = MiniHackObsWrapper(env)
    return env


def get_single_minihack_task(action_space_id, env_name, num_timesteps, eval_mode=False, **kwargs):
    return ImageTask(
        action_space_id=action_space_id,
        env_spec=lambda: make_minihack(env_name, **kwargs),
        num_timesteps=num_timesteps,
        time_batch_size=1,  # no framestack
        eval_mode=eval_mode,
        image_size=[84, 84],
        grayscale=False,
    )
