from continual_rl.policies.config_base import ConfigBase


class PlayPolicyConfig(ConfigBase):

    def __init__(self):
        super().__init__()
        self.key_bindings = "atari"  # Supported ["atari"]

    def _load_from_dict_internal(self, config_dict):
        self._auto_load_class_parameters(config_dict)
        return self
