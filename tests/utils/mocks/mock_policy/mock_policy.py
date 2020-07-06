from continual_rl.policies.policy_base import PolicyBase
from tests.utils.mocks.mock_policy.mock_policy_config import MockPolicyConfig


class MockPolicy(PolicyBase):
    """
    A mock policy for use in unit testing
    """
    def __init__(self, config: MockPolicyConfig):
        super().__init__()
        self._config = config
        pass

    def get_environment_runner(self):
        pass

    def compute_action(self, observation, task_action_count):
        pass

    def train(self, storage_buffer):
        pass

    def save(self, output_path_dir, task_id, task_total_steps):
        pass

    def load(self, model_path):
        pass
