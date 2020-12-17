

class TaskSpec(object):
    """
    Contains the task information that can be shared with the environment runners as the experiment runs.
    """
    def __init__(self, action_space_id, preprocessor, env_spec, time_batch_size, num_timesteps, eval_mode,
                 return_after_reward_num=None):
        self._action_space_id = action_space_id
        self._preprocessor = preprocessor
        self._env_spec = env_spec
        self._time_batch_size = time_batch_size
        self._num_timesteps = num_timesteps
        self._eval_mode = eval_mode
        self._return_after_reward_num = return_after_reward_num

    @property
    def action_space_id(self):
        return self._action_space_id

    @property
    def preprocessor(self):
        return self._preprocessor

    @property
    def env_spec(self):
        return self._env_spec

    @property
    def time_batch_size(self):
        return self._time_batch_size

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def eval_mode(self):
        return self._eval_mode

    @property
    def return_after_reward_num(self):
        return self._return_after_reward_num
