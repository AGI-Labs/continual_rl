from continual_rl.utils.config_base import ConfigBase


class RandomPolicyConfig(ConfigBase):

    def __init__(self):
        super().__init__()
        self.timesteps_per_collection = 100

    def _load_from_dict_internal(self, config_dict):
        self.timesteps_per_collection = config_dict.pop("timesteps_per_collection", self.timesteps_per_collection)
        return self
