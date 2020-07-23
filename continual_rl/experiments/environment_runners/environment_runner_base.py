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
    def collect_data(self, time_batch_size, env_spec, preprocessor, action_space_id):
        """
        Returns a list of InfoToStores, each representing the data collected at a particular timestep.
        The policy creates an instance of its subclass of InfoToStore, and populates it with the appropriate data.
        Then this method should populate InfoToStore.reward and InfoToStore.done.
        Also returns the total number of timesteps run during this collection and if any episodes finished, what
        their final reward was.
        :param time_batch_size: The number of sequential observations to collect. Will be the first dimension of the
        observation passed to the policy
        :param env_spec: A specification to use to make environments with Utils.make_env
        :param preprocessor: The preprocessor for the observation, e.g. to convert it to a tensor. Provided by
        the subclass of TaskBase that calls this function
        :param action_space_id: The unique identifier for the action space of the task being run. Multiple tasks
        that share the same action space will have the same id.
        :return: timesteps, InfoToStores[], rewards_to_report
        """
        pass
