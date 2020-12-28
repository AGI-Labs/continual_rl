from continual_rl.policies.impala.torchbeast.monobeast import Monobeast


class ClearMonobeast(Monobeast):
    """
    An implementation of Experience Replay for Continual Learning (Rolnick et al, 2019):
    https://arxiv.org/pdf/1811.11682.pdf
    """
    def __init__(self, model_flags, observation_space, action_space, policy_class):
        super().__init__(model_flags, observation_space, action_space, policy_class)
