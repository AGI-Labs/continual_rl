import torch
from continual_rl.experiments.tasks.task_base import TaskBase
from continual_rl.experiments.tasks.preprocessor_base import PreprocessorBase
from continual_rl.utils.utils import Utils
import gym_minigrid  # Needed for Utils.make_env
from gym.spaces.box import Box


class MiniGridPreprocessor(PreprocessorBase):
    def __init__(self, dummy_env):
        image_observation_space = dummy_env.observation_space['image']
        observation_size = image_observation_space.shape
        rearranged_observation_size = [observation_size[2], observation_size[0], observation_size[1]]

        # Minigrid tasks are represented by integers in the range [0, 10]
        # Specifically, each of the 3 channels is [OBJECT_IDX, COLOR_IDX, STATE]
        # OBJECT_IDX is [0, 10], COLOR_IDX is [0, 5], and STATE is [0, 2]
        # (https://github.com/maximecb/gym-minigrid/blob/master/gym_minigrid/minigrid.py)
        observation_space = Box(low=0,
                                high=10,
                                shape=rearranged_observation_size)
        super().__init__(observation_space)

    def preprocess(self, x):
        # Minigrid images are [H, W, C], so rearrange to pytorch's expectations.
        return torch.Tensor(x['image']).permute(2, 0, 1)

    def render_episode(self, episode_observations):
        """
        Turn a list of observations gathered from the episode into a video that can be saved off to view behavior.
        """
        # Note: the 3 channels aren't really representing RGB, so this is a convenient but not necessarily
        # optimally understandable representation
        return torch.stack(episode_observations).unsqueeze(0)


class MiniGridTask(TaskBase):
    """
    MiniGrid has a custom observation format, so we have a separate Task type to handle parsing it
    """
    def __init__(self, action_space_id, env_spec, time_batch_size, num_timesteps, eval_mode):
        dummy_env = Utils.make_env(env_spec)
        action_space = dummy_env.action_space
        preprocessor = MiniGridPreprocessor(dummy_env)

        super().__init__(action_space_id, preprocessor, env_spec, preprocessor.observation_space, action_space,
                         time_batch_size, num_timesteps, eval_mode)
