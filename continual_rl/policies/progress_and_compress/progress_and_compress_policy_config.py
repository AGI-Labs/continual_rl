from continual_rl.policies.ewc.ewc_policy_config import OnlineEWCPolicyConfig


class ProgressAndCompressPolicyConfig(OnlineEWCPolicyConfig):

    def __init__(self):
        super().__init__()
