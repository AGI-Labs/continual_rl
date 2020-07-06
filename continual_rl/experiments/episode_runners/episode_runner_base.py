import types
import gym
from abc import ABC, abstractmethod


class EpisodeRunnerBase(ABC):
    """
    Episode runners handle the collection of episode data. They are a separate class because this can be done in 
    several ways. E.g. synchronously, batched, or fully parallel (each episode on a separate process).

    The arguments provided to __init__ are from the policy.
    The arguments provided to collect_data are from the task.
    """
    def __init__(self):
        pass

    @abstractmethod
    def collect_data(self, time_batch_size, env_spec, preprocessor, task_action_count):
        """
        Returns a list of tuples, where each tuple contains: (timesteps, info_to_store[], rewards) generated from an
        environment.
        """
        pass

    @classmethod
    def make_env(cls, env_spec):
        if isinstance(env_spec, types.LambdaType):
            env = env_spec()
        else:
            env = gym.make(env_spec)

        return env
