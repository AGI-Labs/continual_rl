from abc import ABC, abstractmethod


class PreprocessorBase(ABC):
    def __init__(self, observation_space):
        self._observation_space = observation_space

    @property
    def observation_space(self):
        return self._observation_space

    @abstractmethod
    def preprocess(self, observation):
        raise NotImplementedError()

    @abstractmethod
    def render_episode(self, episode_observations):
        raise NotImplementedError()
