from abc import ABC, abstractmethod


class EnvironmentRunnerBase(ABC):
    """
    Environment runners handle the collection of data from the environment. They are a separate class because this can
    be done in several ways. E.g. synchronously, batched, or fully parallel (each episode on a separate process).
    These are specified by the policy, because they determine what form the data provided to the policy takes (e.g. batched).

    The arguments provided to __init__ are from the policy.
    The arguments provided to collect_data are from the task.
    """
    def __init__(self):
        pass

    @abstractmethod
    def collect_data(self, time_batch_size, env_spec, preprocessor, task_action_count):
        """
        Returns a list of lists, representing all the data collected across potentially multiple environments:
        [[(info_to_store, reward, done)]]. Each inner list represents all the data collected within one environment.
        When done=True, the next entry will be in a reset of the environment (new episode).
        :param time_batch_size: The number of sequential observations to collect. Will be the first dimension of the
        observation passed to the policy
        :param env_spec: A specification to use to make environments with Utils.make_env
        :param preprocessor: The preprocessor for the observation, e.g. to convert it to a tensor. Provided by
        the subclass of TaskBase that calls this function
        :param task_action_count: The number of actions accepted by the environment
        :return: See comment
        """
        pass
