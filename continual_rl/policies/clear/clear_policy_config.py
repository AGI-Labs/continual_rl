from continual_rl.policies.impala.impala_policy_config import ImpalaPolicyConfig


class ClearPolicyConfig(ImpalaPolicyConfig):

    def __init__(self):
        super().__init__()
        self.num_actors = 8  # Must be at least as large as batch_size * replay_ratio
        self.batch_size = 8

        self.replay_buffer_frames = 1e8
        self.replay_ratio = 1.0  # The number of replay entries added to the batch = replay_ratio * batch_size
        # CLEAR reports using a 50-50 mixture of novel and replay experiences

        self.policy_cloning_cost = 0.01
        self.value_cloning_cost = 0.005
        self.large_file_path = None  # No default, since it can be very large and we want no surprises

        self.zarr_filebacked = False
        self.chunk_size = 8

    def _load_from_dict_internal(self, config_dict):
        config = super()._load_from_dict_internal(config_dict)
        assert config.large_file_path is not None, "A file path must be specified where large files may be stored."
        return config
