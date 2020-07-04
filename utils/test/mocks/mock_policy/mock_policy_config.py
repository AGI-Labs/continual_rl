from utils.config_base import ConfigBase


class MockPolicyConfig(ConfigBase):

    def __init__(self):
        super().__init__()
        self.test_param = "unfilled"

    def _load_from_dict_internal(self, config_json):
        self.test_param = config_json.pop("test_param", self.test_param)

        return self
