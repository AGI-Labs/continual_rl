import torch


class DirectoryData(object):
    """
    Contains exclusively the data held in the hypothesis directory, and simple shared accessors
    There are separate classes that use this data, depending on which process we're currently in.
    """

    def __init__(self, use_cuda, output_dir, obs_space, action_size, config, replay_buffer_size, filter_learning_rate, is_sync):
        self._long_term_directory = []  # "Long term storage"
        self._max_reward_received = 0.01  # Used for scaling rewards

        self._use_cuda = use_cuda
        self._is_sync = is_sync
        self._obs_space = obs_space
        self._action_size = action_size
        self._output_dir = output_dir
        self._master_device_id = torch.device("cpu" if torch.cuda.device_count() == 0 else "cpu")  # cuda:0")  # TODO: make this accessible from policy

        # Filter params
        self._config = config
        self._replay_buffer_size = replay_buffer_size
        self._filter_learning_rate = filter_learning_rate

        # TODO: all of this should be consolidated in sane_policy_config, just being slow about it
        self._max_hypotheses_per_layer = self._config.max_hypotheses_per_layer  # The number of entries here determines the number of layers as well
        self._refractory_step_counts_per_layer = self._config.refractory_step_counts_per_layer  #[None, (50, 5)] #2  # Number of steps used in a row before the hypothesis enters "refractory", and the number of steps to wait before being active again. If None, refractory is turned off
        self._merge_ratio = self._config.merge_ratio
        self._min_short_term_total_usage_count = self._config.min_short_term_total_usage_count  # Min before eligible to become long-term
        self._num_times_to_train_meta = self._config.num_train_meta
        self._num_times_to_train_long_term = self._config.num_train_long_term
        self._num_times_to_train_short_term = self._config.num_train_short_term
        self._allowed_error_scale = self._config.allowed_error_scale  # Used in hypothesis selection (mean + k * error)
        self._allowed_error_scale_strict = self._config.allowed_error_scale_for_creation  # TODO: remove "None" capability, and rename Defines how much error is tolerated on either side of the mean, before action is taken (create ST, or move ST to LT)
        self._max_processes = self._config.max_processes  # 4 gpus, n proc per
        self._use_negative_examples = self._config.negative_example_scale > 0
        self._use_curiosity = False
        self._usage_count_min_to_convert_to_long_term = self._config.usage_count_min_to_convert_to_long_term  # Edge case initialization
        self._closeness_threshold = 1e-1  # How close 2 hypotheses need to be to trigger the "randomly pick" selection method
        self._always_train_all_long_term = False
        self._duplicate_uses_replay = True
        self._skip_short_term_greater_than_long_term = True
        self._num_before_train = self._config.num_before_train

        assert self._refractory_step_counts_per_layer is None or len(self._refractory_step_counts_per_layer) == len(self._max_hypotheses_per_layer), "Is refractory step count out of step?"

    @property
    def all_hypotheses(self):
        for long_term_entry in self._long_term_directory:
            yield from long_term_entry.all_hypotheses

    def set_from(self, directory_data):
        self._long_term_directory = directory_data._long_term_directory

    def get_hypothesis_from_id(self, id):
        selected_hypothesis = None

        for entry in self.all_hypotheses:
            if entry.unique_id == id:
                selected_hypothesis = entry
                break
            if entry.is_long_term and entry.prototype.unique_id == id:
                selected_hypothesis = entry.prototype
                break

        # May return None if the hypothesis was not found.
        return selected_hypothesis
