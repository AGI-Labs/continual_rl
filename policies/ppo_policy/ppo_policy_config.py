from utils.config_base import ConfigBase


class PPOPolicyConfig(ConfigBase):

    def __init__(self):
        super().__init__()

    def _load_from_dict_internal(self, config_dict):
        return self
