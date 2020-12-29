from continual_rl.policies.config_base import ConfigBase


class ImpalaPolicyConfig(ConfigBase):
    def __init__(self):
        super().__init__()
        self.num_actors = 4
        self.batch_size = 8
        self.unroll_length = 80
        self.num_buffers = None
        self.num_learner_threads = 2
        self.use_lstm = False  # Not presently fully supported
        self.entropy_cost = 0.0006
        self.baseline_cost = 0.5
        self.policy_cloning_cost = 0.01
        self.value_cloning_cost = 0.005
        self.discounting = 0.99
        self.reward_clipping = "abs_one"
        self.learning_rate = 0.00048
        self.alpha = 0.99  # RMSProp smoothing constant
        self.momentum = 0  # RMSProp momentum
        self.epsilon = 0.01  # RMSProp epsilon
        self.grad_norm_clipping = 40.0
        self.disable_cuda = False
        self.disable_checkpoint = False
        self.use_clear = False
        self.comment = ""
        self.replay_buffer_frames = 1250000  # Half the number of frames in the full MNIST experiment
        self.large_file_path = "tmp"
        self.net_flavor = "default"  # "default", "100x"
        self.replay_ratio = 0.5  # Half of samples trained on are from the replay buffer

    def _load_from_dict_internal(self, config_dict):
        self._auto_load_class_parameters(config_dict)
        return self
