from continual_rl.policies.impala.impala_policy_config import ImpalaPolicyConfig


class ClearPolicyConfig(ImpalaPolicyConfig):

    def __init__(self):
        super().__init__()
        self.replay_buffer_size = 1e8
        self.replay_ratio = 1.0  # The number of replay entries added to the batch = replay_ratio * batch_size
        self.policy_cloning_cost = 0.01
        self.value_cloning_cost = 0.005
        self.large_file_path = "tmp"
