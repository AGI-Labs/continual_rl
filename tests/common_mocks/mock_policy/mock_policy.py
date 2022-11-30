from continual_rl.policies.policy_base import PolicyBase
from tests.common_mocks.mock_policy.mock_policy_config import MockPolicyConfig
from tests.common_mocks.mock_environment_runner import MockEnvironmentRunner


class MockPolicy(PolicyBase):
    """
    A mock policy for use in unit testing. This is just basically a de-abstraction of the base class.
    For any test-specific usages, monkeypatch the appropriate function.
    """
    def __init__(self, config: MockPolicyConfig, observation_space, action_spaces):
        super().__init__(config)
        self._config = config
        self.train_run_count = 0
        self.load_count = 0
        self.save_count = 0
        self.current_env_runner = None

    def get_environment_runner(self, task_spec):
        # In general this should not be saved off, but doing so here to use it as a spy into env runner behavior.
        self.current_env_runner = MockEnvironmentRunner()
        return self.current_env_runner

    def compute_action(self, observation, task_id, action_space_id, last_timestep_data, eval_mode):
        pass

    def train(self, storage_buffer):
        self.train_run_count += 1

    def save(self, output_path_dir, cycle_id, task_id, task_total_steps):
        self.save_count += 1

    def load(self, model_path):
        self.load_count += 1
