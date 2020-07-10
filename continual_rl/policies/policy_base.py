from abc import ABC, abstractmethod


class PolicyBase(ABC):
    """
    The base class that all agents should implement, enabling them to act in the world.
    """
    def __init__(self):
        """
        Subclass policies will always be initialized with: (config, observation_size, action_size).
        No other parameters should be added - the policy won't be loaded with them from the configuration loader.
        Any custom parameters should be put on config.
        """
        pass

    def shutdown(self):
        """
        Indicates the experiment has shutdown, and the policy should cleanup any resources it has open.
        Optional.
        """
        pass

    @abstractmethod
    def get_environment_runner(self):
        """
        Return an instance of the subclass of EnvironmentRunnerBase to be used to run an environment with this policy.
        This is policy-dependent because it determines the cadence and type of observations provided to the policy.
        If the policy supports multiple, which one is used can be configured using the policy_config.
        Each time this function is called, a new EnvironmentRunner should be returned.
        :return: an instance of an EnvironmentRunnerBase subclass
        """
        pass

    @abstractmethod
    def compute_action(self, observation, task_action_count):
        """
        If a non-synchronous environment runner is specified (or may be in the future), this method should not change
        any instance state, because this method may be run on different  processes or threads to enable parallelization.
        Any information that is needed for updating the policy should be specified in info_to_store.

        :param observation: The expected observation is dependent on what environment runner has been configured for
        the policy, as well as the task type. For instance, an ImageTask with EnvironmentRunnerBatch configured
        will provide an observation that is of shape [batch, time, channels, width, height]. See the documentation for
        collect_data for a given EnvironmentRunner for more detail.
        :param task_action_count: The number of actions allowed by the task currently being executed. This policy's 
        action space might be larger than that of the task currently being executed, so compute_action() here is
        provided with this parameter to enable it to select an action that is within the allowable action space of the task.
        :return: (selected action, info_to_store): info_to_store is an object arbitrarily specified by the subclass.
        It should contain whatever extra information is required for training. A list of lists of info_to_store are 
        provided to train(), and are described more there.
        """
        pass

    @abstractmethod
    def train(self, storage_buffer):
        """
        By default, training will not be parallelized, therefore this method may freely update instance state.
        :param storage_buffer: A list of lists: [[(info_to_store, reward, done)]]. Each inner list represents the data
        collected by a single environment since the last time train() was called. This list is generated by the
        EnvironmentRunner, so further details can be viewed there.
        :return: None
        """
        pass

    @abstractmethod
    def save(self, output_path_dir, task_id, task_total_steps):
        """
        Saving is delegated to the policy, as there may be more complexity than just torch.save().
        :param output_path_dir: The directory to which the model should be saved
        :param task_id: The task currently being executed when a save was triggered
        :param task_total_steps: The number of steps into this task we are at the time of saving.
        :return: The full path to the saved file
        """
        pass

    @abstractmethod
    def load(self, model_path):
        """
        Load the model from model_path.
        :param model_path: The path 
        :return: The loaded model (can be self if the model was loaded into the current policy)
        """
        pass
