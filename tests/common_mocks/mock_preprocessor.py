from continual_rl.experiments.tasks.preprocessor_base import PreprocessorBase
from gym.spaces.box import Box
import torch


class MockPreprocessor(PreprocessorBase):
    def __init__(self):
        observation_space = Box(low=0, high=1, shape=[1, 2, 3])
        super().__init__(observation_space)

    def preprocess(self, observation):
        return torch.Tensor(observation)

    def render_episode(self, episode_observations):
        """
        Turn a list of observations gathered from the episode into a video that can be saved off to view behavior.
        """
        raise NotImplementedError("render_episode not implemented for mock_preprocessor")
