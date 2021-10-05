from continual_rl.policies.impala.impala_policy import ImpalaPolicy
from continual_rl.policies.clear.clear_policy_config import ClearPolicyConfig
from continual_rl.policies.clear.clear_monobeast import ClearMonobeast


class ClearPolicy(ImpalaPolicy):
    def __init__(self, config: ClearPolicyConfig, observation_space, action_spaces, policy_net_class=None):
        super().__init__(config, observation_space, action_spaces, impala_class=ClearMonobeast, policy_net_class=policy_net_class)
