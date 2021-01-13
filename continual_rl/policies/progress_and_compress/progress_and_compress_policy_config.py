from continual_rl.policies.ewc.ewc_policy_config import OnlineEWCPolicyConfig


class ProgressAndCompressPolicyConfig(OnlineEWCPolicyConfig):

    def __init__(self):
        super().__init__()
        self.example_param = 100

    def _load_from_dict_internal(self, config_dict):
        # Note: consider using _auto_load_class_parameters if your parsing is simple
        self.example_param = config_dict.pop("example_param", self.example_param)
        return self
