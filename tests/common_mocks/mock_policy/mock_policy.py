from continual_rl.policies.policy_base import PolicyBase
from tests.common_mocks.mock_policy.mock_policy_config import MockPolicyConfig
from tests.common_mocks.mock_environment_runner import MockEnvironmentRunner


class MockPolicy(PolicyBase):
    """
    A mock policy for use in unit testing. This is just basically a de-abstraction of the base class.
    For any test-specific usages, monkeypatch the appropriate function.
    """
    def __init__(self, config: MockPolicyConfig, observation_space, action_spaces):
        super().__init__()
        self._config = config
        pass

    def get_environment_runner(self):
        return MockEnvironmentRunner()

    def compute_action(self, observation, action_space_id, last_timestep_data, eval_mode):
        pass

    def train(self, storage_buffer):
        pass

    def save(self, output_path_dir, task_id, task_total_steps):
        pass

    def load(self, model_path):
        pass
