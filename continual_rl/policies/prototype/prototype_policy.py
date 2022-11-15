from continual_rl.policies.policy_base import PolicyBase

from .prototype_policy_config import PrototypePolicyConfig  # Switch to your config type


class PrototypePolicy(PolicyBase):
    """
    A simple implementation of policy as a sample of how policies can be created.
    Refer to policy_base itself for more detailed descriptions of the method signatures.
    """
    def __init__(self, config: PrototypePolicyConfig, observation_space, action_spaces):  # Switch to your config type
        super().__init__(config)
        self._config = config
        self._observation_space = observation_space
        self._action_spaces = action_spaces

    def get_environment_runner(self, task_spec):
        raise NotImplementedError

    def compute_action(self, observation, task_id, action_space_id, last_timestep_data, eval_mode):
        raise NotImplementedError

    def train(self, storage_buffer):
        raise NotImplementedError

    def save(self, output_path_dir, cycle_id, task_id, task_total_steps):
        raise NotImplementedError

    def load(self, output_path_dir):
        raise NotImplementedError
