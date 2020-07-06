from continual_rl.utils.config_base import ConfigBase


class PrototypePolicyConfig(ConfigBase):

    def __init__(self):
        super().__init__()
        self.example_param = 100

    def _load_from_dict_internal(self, config_dict):
        self.example_param = config_dict.pop("example_param", self.example_param)
        return self
