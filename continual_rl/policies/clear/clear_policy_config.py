from continual_rl.policies.impala.impala_policy_config import ImpalaPolicyConfig


class ClearPolicyConfig(ImpalaPolicyConfig):

    def __init__(self):
        super().__init__()
        self.replay_buffer_size = 1e8
        self.large_file_path = "tmp"
