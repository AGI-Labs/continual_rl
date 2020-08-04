from continual_rl.policies.config_base import ConfigBase


class DiscreteRandomPolicyConfig(ConfigBase):

    def __init__(self):
        super().__init__()
        self.timesteps_per_collection = 128  # Per process, for batch
        self.num_parallel_envs = None  # If None we operate synchronously, otherwise we batch

    def _load_from_dict_internal(self, config_dict):
        self.timesteps_per_collection = config_dict.pop("timesteps_per_collection", self.timesteps_per_collection)

        # Only necessary because the default is "None"
        self.num_parallel_envs = config_dict.pop("num_parallel_envs", self.num_parallel_envs)
        self.num_parallel_envs = int(self.num_parallel_envs) if self.num_parallel_envs is not None else None

        return self
