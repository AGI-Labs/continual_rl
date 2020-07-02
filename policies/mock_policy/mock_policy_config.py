from utils.config_base import ConfigBase


class MockPolicyConfig(ConfigBase):

    def __init__(self, config_path, output_dir):
        super().__init__(config_path, output_dir)

    def _load_single_experiment_from_config(self, config_json):
        pass
