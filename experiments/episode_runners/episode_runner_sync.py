from .episode_runner_base import EpisodeRunnerBase


class EpisodeRunnerSync(EpisodeRunnerBase):
    """
    An episode collection class that will collect the data synchronously.
    """
    def __init__(self):
        super().__init__()

    def collect_episode_data(self):
        pass
