from abc import ABC, abstractmethod


class EpisodeRunnerBase(ABC):
    """
    Episode runners handle the collection of episode data. They are a separate class because this can be done in 
    several ways. E.g. synchronously, batched, or fully parallel (each episode on a separate process).
    """
    def __init__(self):
        pass

    @abstractmethod
    def collect_episode_data(self):
        pass
