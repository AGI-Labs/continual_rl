import gym
import numpy as np
import os

from .image_task import ImageTask


class MiniHackObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
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


# Ref: https://github.com/MiniHackPlanet/MiniHack/blob/e124ae4c98936d0c0b3135bf5f202039d9074508/minihack/agent/polybeast/config.yaml#L48
# https://github.com/facebookresearch/nle/blob/b85184f65426e8a7a63b3fdbb1dead135e01e6cc/nle/env/tasks.py#L41
def make_minihack(
    env_name,
    observation_keys=["pixel_crop"],
    reward_win=1,
    reward_lose=0,
    penalty_time=0.0,
    penalty_step=-0.001,  # MiniHack uses different than -0.01 default of NLE
    penalty_mode="constant",
    character="mon-hum-neu-mal",
    savedir=None,  # save_tty=False -> savedir=None, see https://github.com/MiniHackPlanet/MiniHack/blob/e124ae4c98936d0c0b3135bf5f202039d9074508/minihack/agent/common/envs/tasks.py#L168
    **kwargs,
):
    import minihack

    env = gym.make(
        f"MiniHack-{env_name}",
        observation_keys=observation_keys,
        reward_win=reward_win,
        reward_lose=reward_lose,
        penalty_time=penalty_time,
        penalty_step=penalty_step,
        penalty_mode=penalty_mode,
        character=character,
        savedir=savedir,
        **kwargs,
    )  # each env specifies its own self._max_episode_steps
    env = MiniHackMakeVecSafeWrapper(env)
    env = MiniHackObsWrapper(env)
    return env


def get_single_minihack_task(task_id, action_space_id, env_name, num_timesteps, eval_mode=False, **kwargs):
    return ImageTask(
        task_id=task_id,
        action_space_id=action_space_id,
        env_spec=lambda: make_minihack(env_name, **kwargs),
        num_timesteps=num_timesteps,
        time_batch_size=1,  # no framestack
        eval_mode=eval_mode,
        image_size=[84, 84],
        grayscale=False,
    )
