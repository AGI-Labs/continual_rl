from continual_rl.policies.impala.impala_policy import ImpalaPolicy
from continual_rl.policies.ewc.ewc_policy_config import EWCPolicyConfig
from continual_rl.policies.ewc.ewc_monobeast import EWCMonobeast 


class EWCPolicy(ImpalaPolicy):
    def __init__(self, config: EWCPolicyConfig, observation_space, action_spaces):
        super().__init__(config, observation_space, action_spaces, impala_class=EWCMonobeast)

    def set_action_space(self, action_space_id):
        super().set_action_space(action_space_id)
        self.impala_trainer.set_current_task(action_space_id)