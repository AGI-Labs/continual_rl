import gym
import numpy as np
import os

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


# from https://github.com/MiniHackPlanet/MiniHack/blob/e9c8c20fb2449d1f87163314f9b3617cf4f0e088/minihack/scripts/venv_demo.py#L28
class MiniHackMakeVecSafeWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.basedir = os.getcwd()

    def step(self, action: int):
        os.chdir(self.env.env._vardir)
        x = self.env.step(action)
        os.chdir(self.basedir)
        return x

    def reset(self):
        os.chdir(self.env.env._vardir)
        x = self.env.reset()
        os.chdir(self.basedir)
        return x

    def close(self):
        os.chdir(self.env.env._vardir)
        self.env.close()
        os.chdir(self.basedir)

    def seed(self, core=None, disp=None, reseed=False):
        os.chdir(self.env.env._vardir)
        self.env.seed(core, disp, reseed)
        os.chdir(self.basedir)


def make_minihack(env_name, observation_keys=["pixel_crop"], **kwargs):
    import minihack

    env = gym.make(f"MiniHack-{env_name}", observation_keys=observation_keys, **kwargs)  # each env specifies its own self._max_episode_steps
    env = MiniHackMakeVecSafeWrapper(env)
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
