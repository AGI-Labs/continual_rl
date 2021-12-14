import gym
import numpy as np
import os
import torch
import nle.env.tasks
import nle.env.base

from continual_rl.experiments.tasks.preprocessor_base import PreprocessorBase
from continual_rl.experiments.tasks.task_base import TaskBase
from continual_rl.utils.utils import Utils
from continual_rl.experiments.experiment import Experiment
import continual_rl.experiments.envs.nethack_envs
import continual_rl.experiments.envs.nethack_inv_envs
from hackrl.wrappers import RenderCharImagesWithNumpyWrapper


class MiniHackObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, dtype=np.uint8, shape=(84, 84, 3))

    def observation(self, obs):
        obs = obs["pixel_crop"]
        obs = np.pad(obs, [(2, 2), (2, 2), (0, 0)])
        return obs


class MiniHackMultiObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space

    def observation(self, obs):
        for obs_key in obs.keys():
            obs[obs_key] = torch.tensor(obs[obs_key])
        return obs


class MiniHackPreprocessor(PreprocessorBase):

    def preprocess(self, batched_env_image):
        return batched_env_image

    def render_episode(self, episode_observations):
        observations = []
        for frame_observation in episode_observations:
            if "pixel_crop" in frame_observation:
                pixel_obs = frame_observation["pixel_crop"].squeeze(0).squeeze(0)
                observations.append(pixel_obs.permute(2, 0, 1))
            elif "screen_image" in frame_observation:
                pixel_obs = frame_observation["screen_image"].squeeze(0).squeeze(0)
                observations.append(pixel_obs)

        if len(observations) > 0:
            observations = torch.stack(observations).unsqueeze(0)
        else:
            observations = None #torch.zeros((1, 1, 3, 5, 5))  # Just display a white square if we have nothing else

        return observations


class MinihackTask(TaskBase):
    def __init__(self, task_id, action_space_id, env_spec, num_timesteps, eval_mode,
                 continual_eval=True):
        dummy_env, _ = Utils.make_env(env_spec)
        preprocessor = MiniHackPreprocessor(dummy_env.observation_space)

        super().__init__(task_id, action_space_id, preprocessor, env_spec, dummy_env.observation_space,
                         dummy_env.action_space, num_timesteps, eval_mode, continual_eval=continual_eval)


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


def make_hackrl_minihack():
    from hackrl.environment import create_env
    import dotmap
    flags = {"env": {"name": "challenge", "max_episode_steps": 250},
            "character": 'mon-hum-neu-mal',
            "penalty_step": -0.001,
            "penalty_time": 0.0,
            "fn_penalty_step": "constant",
            "add_image_observation": True,
            "state_counter": "none"}
    return create_env(dotmap.DotMap(flags))


# Ref: https://github.com/MiniHackPlanet/MiniHack/blob/e124ae4c98936d0c0b3135bf5f202039d9074508/minihack/agent/polybeast/config.yaml#L48
# https://github.com/facebookresearch/nle/blob/b85184f65426e8a7a63b3fdbb1dead135e01e6cc/nle/env/tasks.py#L41
def make_minihack(
    env_name,
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
    # TODO: NOTE do not use "internal" in your models. It's included for logging (TODO: better place)
    #observation_keys=["glyphs", "chars", "colors", "specials", "blstats", "message", "tty_chars", "tty_colors", "internal", "inv_oclasses", "inv_glyphs", "inv_strs", "inv_letters", "tty_cursor"] #, "pixel_crop"],  Pixel crop not available much to my infinite displeasure
    observation_keys=["glyphs", "blstats", "message", "internal", "inv_glyphs", "tty_chars", "tty_colors"] #, "pixel_crop"],  Pixel crop not available much to my infinite displeasure
    #observation_keys=["blstats", "message", "tty_chars", "tty_colors", "internal"] #, "pixel_crop"],  Pixel crop not available much to my infinite displeasure
    #actions = nle.env.tasks.TASK_ACTIONS  # TODO: this is trimmed down, i.e. doesn't include like "wear"
    actions = nle.env.base.FULL_ACTIONS

    if "MiniHack" in env_name:
        #observation_keys += ["pixel_crop"]
        env = gym.make(
            f"{env_name}",
            observation_keys=observation_keys,
            #reward_win=reward_win,
            #reward_lose=reward_lose,
            #penalty_time=penalty_time,
            #penalty_step=penalty_step,
            #penalty_mode=penalty_mode,
            #character=character,
            savedir=savedir,
            actions=actions
            #**kwargs,
        )  # each env specifies its own self._max_episode_steps
        #env = MiniHackMakeVecSafeWrapper(env)  # TODO: check if still necessary
    else:
        env = gym.make(f"{env_name}",
            observation_keys=observation_keys,
            actions=actions)  # TODO: kind of hacky quick way to get the NLE challenge going. Means none of the passed in params are used, also the name is misleading

    env = RenderCharImagesWithNumpyWrapper(env)
    env = MiniHackMultiObsWrapper(env)  # TODO: configurable?
    return env


def get_single_minihack_task(task_id, action_space_id, env_name, num_timesteps, eval_mode=False, use_hackrl=False, **kwargs):
    env_spec = (lambda: make_hackrl_minihack()) if use_hackrl else (lambda: make_minihack(env_name, **kwargs))
    return MinihackTask(
        task_id=task_id,
        action_space_id=action_space_id,
        env_spec=env_spec,
        num_timesteps=num_timesteps,
        eval_mode=eval_mode,
    )


def create_minihack_loader(
    task_prefix,
    env_name_pairs,
    num_timesteps=10e6,
    task_params=None,
    continual_testing_freq=1000,
    cycle_count=1,
    use_hackrl=False
):
    task_params = task_params if task_params is not None else {}

    def loader():
        tasks = []
        for action_space_id, pairs in enumerate(env_name_pairs):
            # If we passed in a list of timesteps, pick the appropriate one. Otherwise use the same number fora ll
            task_timesteps = num_timesteps[action_space_id] if isinstance(num_timesteps, list) else num_timesteps

            if pairs[0] is not None:
                train_task = get_single_minihack_task(f"{task_prefix}_{action_space_id}", action_space_id, pairs[0],
                                                    task_timesteps, use_hackrl=use_hackrl, **task_params)
                tasks += [train_task]

            if pairs[1] is not None:
                # TODO: task_timesteps was originally 0, so it was *just* a CRL thing, making non-zero so we can use it without train
                eval_task = get_single_minihack_task(f"{task_prefix}_{action_space_id}_eval", action_space_id, pairs[1],
                                                    num_timesteps=1e5, eval_mode=True, use_hackrl=use_hackrl, **task_params)

                tasks += [eval_task]

        return Experiment(
            tasks,
            continual_testing_freq=continual_testing_freq,
            cycle_count=cycle_count,
        )
    return loader
