from continual_rl.policies.config_base import ConfigBase


class PPOPolicyConfig(ConfigBase):

    def __init__(self):
        super().__init__()
        self.eps = 1e-5
        self.learning_rate = 7e-4
        self.gamma = .99
        self.use_gae = False
        self.gae_lambda = 0.95
        self.entropy_coef = 0.01
        self.value_loss_coef = 0.5
        self.max_grad_norm = 0.5
        self.seed = 1  # TODO: This is what the original code does, but IMO the default should be random
        self.num_processes = 16  # TODO: num_parallel_envs
        self.num_steps = 5  # TODO: so low?
        self.ppo_epoch = 4
        self.num_mini_batch = 32
        self.clip_param = 0.2
        self.save_interval = 100  # TODO: convert to timesteps, also currently unused
        self.use_proper_time_limits = False
        self.recurrent_policy = False
        self.use_linear_lr_decay = False
        self.cuda = True
        self.render_collection_freq = 10000  # timesteps

    def _load_from_dict_internal(self, config_dict):
        loaded_policy_config = self._auto_load_class_parameters(config_dict)
        return loaded_policy_config
