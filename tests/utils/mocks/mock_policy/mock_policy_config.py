from continual_rl.policies.config_base import ConfigBase


class MockPolicyConfig(ConfigBase):

    def __init__(self):
        super().__init__()
        self.test_param = "unfilled"
        self.test_param_2 = "also unfilled"

    def _load_from_dict_internal(self, config_dict):
        self.test_param = config_dict.pop("test_param", self.test_param)
        self.test_param_2 = config_dict.pop("test_param_2", self.test_param_2)

        return self
