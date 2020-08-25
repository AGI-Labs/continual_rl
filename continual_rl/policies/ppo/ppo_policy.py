from continual_rl.policies.policy_base import PolicyBase
from continual_rl.policies.prototype.ppo_policy_config import PPOPolicyConfig  # Switch to your config type


class PPOPolicy(PolicyBase):
    """
    A simple implementation of policy as a sample of how policies can be created.
    Refer to policy_base itself for more detailed descriptions of the method signatures.
    """
    def __init__(self, config: PPOPolicyConfig, observation_size, action_spaces):  # Switch to your config type
        super().__init__()
        pass

    def get_environment_runner(self):
        pass

    def compute_action(self, observation, action_space_id, last_timestep_data):
        pass

    def train(self, storage_buffer):
        pass

    def save(self, output_path_dir, task_id, task_total_steps):
        pass

    def load(self, model_path):
        pass
