from continual_rl.policies.config_base import ConfigBase


class SanePolicyConfig(ConfigBase):

    def __init__(self):
        super().__init__()
        self.replay_buffer_size = 4096
        self.filter_train_batch_size = 4096  # How many get trained total - TODO: rename
        self.batch_size = 1024  # How the total is split up (number per group)
        self.filter_learning_rate = 1e-4
        self.consequent_learning_rate = 1e-4
        self.use_cuda = False
        self.comment = ""
        self.reward_decay_rate = .99  # The decay rate for the rewards in an episode
        self.is_sync = False  # If not synchronous, will use multiple processes
        self.env_mode = "parallel"  # "batch" also available
        self.negative_example_scale = 0
        self.max_hypotheses_per_layer = [20, 4]
        self.merge_ratio = 3.0/4.0
        self.num_train_meta = 1
        self.num_train_long_term = 2
        self.num_train_short_term = 2
        self.max_processes = 12
        self.timesteps_per_collection = 1000
        self.num_parallel_envs = 1  # Too high a value here can impact both learning speed and recall (it seems, TODO)
        self.random_action_rate = 0.02
        self.refractory_step_counts_per_layer = None
        self.allowed_error_scale = 1.0  # During selection (UCB)
        self.allowed_error_scale_for_creation = [1.0, 1.0]  # If you put just one number, it uses it for both. 2 here so dict loading works easily
        self.min_short_term_total_usage_count = 1500.0
        self.usage_count_min_to_convert_to_long_term = 10000  # If less than 0, assume no "outgrowing" should occur
        self.num_before_train = 0  # !! WARNING: seems bad (at least 1k does), use with CAUTION. How many samples are necessary before training triggers for a hypothesis (cumulative)
        self.usage_scale = 20000  # Roughly how many samples to "top out" the scale (past this all hypos are equal)
        self.render_freq = 500000
        self.large_file_path = "tmp"
        self.num_on_merge_check_val = 5
        self.recently_used_multiplier = 0  # 0 turns this off
        self.used_hypotheses_count = 6
        self.always_keep_non_decayed = True
        self.average_non_decayed_on_merge = False  # If False, it sums (capped at a max)

    def _load_from_dict_internal(self, config_dict):
        self._auto_load_class_parameters(config_dict)

        return self
