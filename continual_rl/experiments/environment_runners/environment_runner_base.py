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
    def collect_data(self, task_spec):
        """
        Preprocesses the observations received from the environment with the preprocessor then sends these observations
        to the policy. Should generally adhere to the specifications provided by the task_spec.
        (E.g. return_after_episode_num)
        Finally returns a list of lists of TimestepDatas, such that the outer list is by "process" and the inner list
        is by "time".
        ("Process" here can just mean anything that results in multiple sets of collections being returned.)
        The policy creates an instance of its subclass of TimestepData, and populates it with the appropriate data.
        Then this method should populate TimestepData.reward and TimestepData.done.
        Also returns the total number of timesteps run during this collection and if any episodes finished, what
        their final return was.
        It also returns any logs that should be written out.
        :param task_spec: An object of type TaskSpec that contains the task information the runner can access.
        :return: timesteps, TimestepData[][], returns_to_report, logs_to_report
        """
        pass

    def cleanup(self, task_spec):
        """
        An opportunity, at the end of a task, for the environment to clean itself up.
        :param task_spec: An object of type TaskSpec that contains the task information the runner can access.
        """
        pass
