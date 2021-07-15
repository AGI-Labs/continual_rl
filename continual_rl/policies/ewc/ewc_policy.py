from continual_rl.policies.impala.impala_policy import ImpalaPolicy
from continual_rl.policies.ewc.ewc_policy_config import EWCPolicyConfig
from continual_rl.policies.ewc.ewc_monobeast import EWCMonobeast 


class EWCPolicy(ImpalaPolicy):
    def __init__(self, config: EWCPolicyConfig, observation_space, action_spaces, impala_class: EWCMonobeast = None,
                 policy_net_class=None):
        if impala_class is None:
            impala_class = EWCMonobeast

        super().__init__(config, observation_space, action_spaces, impala_class=impala_class,
                         policy_net_class=policy_net_class)

    def set_task_ids(self, task_ids):
        self.impala_trainer.initialize_tasks(task_ids)
