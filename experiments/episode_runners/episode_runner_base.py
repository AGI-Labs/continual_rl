

class EpisodeRunnerBase(object):
    """
    Episode runners handle the collection of episode data. They are a separate class because this can be done in 
    several ways. E.g. synchronously, batched, or fully parallel (each episode on a separate process).
    """
    def __init__(self):
        pass

    def collect_episode_data(self):
        raise NotImplementedError("EpisodeRunner's collect_episode_data not implemented")
