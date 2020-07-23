import torch
import numpy as np
from continual_rl.experiments.tasks.task_base import TaskBase
from continual_rl.utils.utils import Utils
import gym_minigrid  # Needed for Utils.make_env


class MiniGridTask(TaskBase):
    """
    MiniGrid has a custom observation format, so we have a separate Task type to handle parsing it
    """
    def __init__(self, action_space_id, env_spec, time_batch_size, num_timesteps, eval_mode, output_dir):
        dummy_env = Utils.make_env(env_spec)
        observation_size = np.array(dummy_env.observation_space['image'].shape)
        rearranged_observation_size = [observation_size[2], observation_size[0], observation_size[1]]
        action_space = dummy_env.action_space.n

        super().__init__(action_space_id, env_spec, rearranged_observation_size, action_space, time_batch_size, num_timesteps,
                         eval_mode, output_dir)

    def preprocess(self, x):
        # Minigrid images are [H, W, C], so rearrange to pytorch's expectations.
        return torch.Tensor(x['image']).permute(2, 0, 1)
