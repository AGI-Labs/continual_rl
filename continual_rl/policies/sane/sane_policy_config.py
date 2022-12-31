from continual_rl.policies.clear.clear_policy_config import ClearPolicyConfig


class SanePolicyConfig(ClearPolicyConfig):

    def __init__(self):
        super().__init__()
        self.allowed_uncertainty_scale_for_creation = [1.0, 10.0]
        self.uncertainty_scale = 1.0
        self.min_steps_before_force_create = 100000
        self.max_nodes = 16
        self.fraction_of_nodes_mergeable = 0.75  # Of max_nodes
        self.create_adds_replay = False
        self.clear_loss_coeff = 1.0
        self.merge_by_frame = True
        self.merge_by_batch = True  # Alternative: merge by average of entire buffer
        self.uncertainty_scale_in_get_active = 1.0
        self.merge_batch_scale = 5.0  # How many batches to use when computing the merge metric
        self.visualize_nodes = False
        self.keep_larger_reservoir_val_in_merge = True
        self.creation_pattern = "asymmetric_reset_anchor"
        self.use_slow_critic = False
        self.slow_critic_update_cadence = 10000
        self.only_create_from_active = True
        self.slow_critic_ema_new_weight = -1.0  # -1 means use equally weighted average
        self.usage_count_based_merge = False
        self.train_all = False
        self.duplicate_optimizer = True
        self.static_ensemble = False  # Baseline
        self.map_task_id_to_module = False
        self.baseline_extended_arch = True
        self.baseline_includes_uncertainty = True  # Necessary for SANE, overrides IMPALA

    def _load_from_dict_internal(self, config_dict):
        config = super()._load_from_dict_internal(config_dict)
        assert int(self.keep_larger_reservoir_val_in_merge) + int(self.usage_count_based_merge) <= 1, "Only one merge strategy should be specified"
        assert not self.map_task_id_to_module or self.static_ensemble, "map_task_id_to_module requires static_ensemble"
        return config
