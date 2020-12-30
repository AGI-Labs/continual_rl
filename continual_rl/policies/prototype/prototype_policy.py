from continual_rl.policies.policy_base import PolicyBase
from continual_rl.policies.prototype.prototype_policy_config import PrototypePolicyConfig  # Switch to your config type


class PrototypePolicy(PolicyBase):
    """
    A simple implementation of policy as a sample of how policies can be created.
    Refer to policy_base itself for more detailed descriptions of the method signatures.
    """
    def __init__(self, config: PrototypePolicyConfig, observation_space, action_spaces):  # Switch to your config type
        super().__init__()
        pass

    def get_environment_runner(self, task_spec):
        pass

    def compute_action(self, observation, action_space_id, last_timestep_data, eval_mode):
        pass

    def train(self, storage_buffer):
        pass

    def save(self, output_path_dir, task_id, task_total_steps):
        pass

    def load(self, model_path):
        pass
