from continual_rl.policies.config_base import ConfigBase


class PPOPolicyConfig(ConfigBase):

    def __init__(self):
        super().__init__()
        # Defaults from the torch-ac package
        self.timesteps_per_collection = 128  # Per process, for batch
        self.num_parallel_envs = 16  # If None we operate synchronously, otherwise we batch
        self.discount = 0.99
        self.learning_rate = 0.001
        self.gae_lambda = 0.95
        self.entropy_coef = 0.01
        self.value_loss_coef = 0.5
        self.max_grad_norm = 0.5
        self.adam_eps = 1e-8
        self.clip_eps = 0.2
        self.epochs = 4
        self.batch_size = 256
        self.use_cuda = False

    def _load_from_dict_internal(self, config_dict):
        self.timesteps_per_collection = config_dict.pop("timesteps_per_collection", self.timesteps_per_collection)

        # Only necessary because the default is "None"
        self.num_parallel_envs = config_dict.pop("num_parallel_envs", self.num_parallel_envs)
        self.num_parallel_envs = int(self.num_parallel_envs) if self.num_parallel_envs is not None else None

        self.discount = config_dict.pop("discount", self.discount)
        self.learning_rate = config_dict.pop("learning_rate", self.learning_rate)
        self.gae_lambda = config_dict.pop("gae_lambda", self.gae_lambda)
        self.entropy_coef = config_dict.pop("entropy_coef", self.entropy_coef)
        self.value_loss_coef = config_dict.pop("value_loss_coef", self.value_loss_coef)
        self.max_grad_norm = config_dict.pop("max_grad_norm", self.max_grad_norm)
        self.adam_eps = config_dict.pop("adam_eps", self.adam_eps)
        self.clip_eps = config_dict.pop("clip_eps", self.clip_eps)
        self.epochs = config_dict.pop("epochs", self.epochs)
        self.batch_size = config_dict.pop("batch_size", self.batch_size)
        self.use_cuda = config_dict.pop("use_cuda", self.use_cuda)

        return self
