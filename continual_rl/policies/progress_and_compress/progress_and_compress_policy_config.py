from continual_rl.policies.ewc.ewc_policy_config import OnlineEWCPolicyConfig


class ProgressAndCompressPolicyConfig(OnlineEWCPolicyConfig):

    def __init__(self):
        super().__init__()
        self.num_train_steps_of_progress = 1000  # Task steps / (batch size * rollout length)
        self.ewc_per_task_min_frames = 0  # Cadence of compress (incl EWC) cycles is handled differently in P&C
        self.kl_div_scale = 1.0
        self.use_collection_pause = False
        self.eval_on_kb = True
        self.eval_is_stochastic = False
