from continual_rl.policies.config_base import ConfigBase


class PrototypePolicyConfig(ConfigBase):

    def __init__(self):
        super().__init__()

    def _load_from_dict_internal(self, config_dict):
        # Note: If your config parsing requires something more complex, then you can custom load certain parameters
        # first, ie
        # ```self.example_param = custom_parse_fn(config_dict.pop("example_param", self.example_param))```
        self._auto_load_class_parameters(config_dict)
        return self
