from continual_rl.policies.config_base import ConfigBase


class PPOPolicyConfig(ConfigBase):

    def __init__(self):
        super().__init__()
        # Defaults from the a2c_ppo_acktr_gail repo
        self.eps = 1e-5
        self.learning_rate = 7e-4
        self.gamma = .99
        self.use_gae = False
        self.gae_lambda = 0.95
        self.entropy_coef = 0.01
        self.value_loss_coef = 0.5
        self.max_grad_norm = 0.5
        self.seed = 1  # TODO (Issue 42): This is what the original code does, but IMO the default should be random
        self.num_processes = 16
        self.num_steps = 5
        self.ppo_epoch = 4
        self.num_mini_batch = 32
        self.clip_param = 0.2
        self.use_proper_time_limits = False  # Whether to use "bad_masks": checks time limit
        self.recurrent_policy = False
        self.use_linear_lr_decay = False
        self.cuda = True
        self.render_collection_freq = 10000  # timesteps

    def _load_from_dict_internal(self, config_dict):
        loaded_policy_config = self._auto_load_class_parameters(config_dict)
        return loaded_policy_config