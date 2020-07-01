

class PolicyBase(object):
    """
    The base class that all agents should implement, enabling them to act in the world.
    """
    def __init__(self):
        pass

    def compute_action(self, observation, task_action_count):
        """
        This method should not change any instance state, because this method may be run on different 
        processes or threads to enable parallelization. Any information that is needed for updating the policy should be
        specified in info_to_store.
        
        :param observation: The expected observation is dependent on what episode runner has been configured for 
        the policy, as well as the task type. For instance, an ImageTask with episode_runner_batch configured 
        will provide an observation that is of shape [batch, channels, width, height]. See the documentation for more 
        detail.
        :param task_action_count: The number of actions allowed by the task currently being executed. This policy's 
        action space might be larger than that of the task currently being executed, so act() here specifically gets an 
        action that is within the allowable action space of the task.
        :return: (selected action, info_to_store): info_to_store is an object arbitrarily specified by the subclass.
        It should contain whatever extra information is required for training. A list of lists of info_to_store are 
        provided to train(), and are described more there.
        """
        raise NotImplementedError("Policy's act method not implemented")

    def train(self, storage_buffer):
        """
        By default, training will not be parallelized, therefore this method may freely update instance state.
        :param storage_buffer: A list of lists of (info_to_store, reward). Each inner list represents the data collected
        by a single environment since the last time train() was called. The outer list, therefore, is of length 
        num_environments.
        :return: None
        """
        raise NotImplementedError("Policy's train method not implemented")

    def save(self, output_path_dir, task_id, task_total_steps):
        """
        Saving is delegated to the policy, as there may be more complexity than just torch.save().
        :param output_path_dir: The directory to which the model should be saved
        :param task_id: The task currently being executed when a save was triggered
        :param task_total_steps: The number of steps into this task we are at the time of saving.
        :return: The full path to the saved file
        """
        raise NotImplementedError("Policy's save method not implemented")

    def load(self, model_path):
        """
        Load the model from model_path.
        :param model_path: The path 
        :return: The loaded model (can be self if the model was loaded into the current policy)
        """
        raise NotImplementedError("Policy's load method not implemented")
