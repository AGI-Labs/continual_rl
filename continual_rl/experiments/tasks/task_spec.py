

class TaskSpec(object):
    """
    Contains the task information that can be shared with the environment runners as the experiment runs.
    The comments below are written from the perspective of how to use these parameters when writing an
    EnvironmentRunner.
    """
    def __init__(self, task_id, action_space_id, preprocessor, env_spec, num_timesteps, eval_mode,
                 return_after_episode_num=None, continual_eval=True, demonstration_task=False):
        self._task_id = task_id
        self._action_space_id = action_space_id
        self._preprocessor = preprocessor
        self._env_spec = env_spec
        self._num_timesteps = num_timesteps
        self._eval_mode = eval_mode
        self._return_after_episode_num = return_after_episode_num
        self._with_continual_eval = continual_eval
        self._demonstration_task = demonstration_task

    @property
    def task_id(self):
        """
        An ID specific to this task. By contrast to action_space_id, only this task will have this task id.
        """
        return self._task_id

    @property
    def action_space_id(self):
        """
        The id of the action space this task is using. Should be passed into the policy. Action space id indicates
        whether tasks share an action space (multiple tasks can be in the same environment).
        """
        return self._action_space_id

    @property
    def preprocessor(self):
        """
        An instance of a PreprocessorBase subclass. EnvironmentRunners should use this to process the observation
        before passing it to the policy. Also contains a render_episode function that can be used to visualize the
        environment during training.
        """
        return self._preprocessor

    @property
    def env_spec(self):
        """
        Use Utils.make_env to turn the spec into a fully realized environment instance.
        """
        return self._env_spec

    @property
    def num_timesteps(self):
        """
        The total number of timesteps the task is run. Any EnvironmentRunners will likely want to return results more
        often than this.
        """
        return self._num_timesteps

    @property
    def eval_mode(self):
        """
        Whether or not the task should be done in evaluation mode (i.e. the model should not be updated).
        """
        return self._eval_mode

    @property
    def return_after_episode_num(self):
        """
        Return after this number of episodes completes. In batched cases this is best-effort: when a set of runs
        finishes, it might put the total number of episodes over this number.
        """
        return self._return_after_episode_num

    @property
    def with_continual_eval(self):
        """
        Whether or not to use the task in continual evaluation (e.g. for some eval tasks it's ambiguous).
        """
        return self._with_continual_eval

    @property
    def demonstration_task(self):
        """
        Whether or not the task is a demonstration task (i.e. uses demonstrations). This may change, for example, how
        a model uses the results to train.
        """
        return self._demonstration_task
