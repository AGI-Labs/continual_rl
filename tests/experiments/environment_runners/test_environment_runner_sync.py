import torch
import numpy as np
from continual_rl.experiments.environment_runners.environment_runner_sync import EnvironmentRunnerSync
from tests.common_mocks.mock_policy.mock_policy import MockPolicy
from tests.common_mocks.mock_policy.mock_policy_config import MockPolicyConfig
from tests.common_mocks.mock_policy.mock_info_to_store import MockInfoToStore


class MockEnv(object):
    def __init__(self):
        self.actions_executed = []
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1
        return np.array([0, 1, 2])

    def step(self, action):
        self.actions_executed.append(action)
        observation = np.array([12, 13, 14])
        reward = 1.5
        return observation, reward, False, {"info": "unused"}


class TestEnvironmentRunnerSync(object):

    def test_collect_data_simple_success(self, monkeypatch):
        """
        Setup the simple happy-path for collect_data, and make sure things are being populated correctly.
        Simple: no done=True, no rewards returned, etc.
        """
        # Arrange
        # Mock the policy we're running. action_size and observation_size not used.
        mock_policy = MockPolicy(MockPolicyConfig(), action_size=None, observation_size=None)
        monkeypatch.setattr(MockPolicy, "compute_action", lambda _, obs, task_action_count: (3, MockInfoToStore((obs, task_action_count))))

        # The object under test
        runner = EnvironmentRunnerSync(policy=mock_policy, timesteps_per_collection=123)

        # Arguments to collect_data
        time_batch_size = 6
        mock_env_spec = lambda : MockEnv()
        mock_preprocessor = lambda x: torch.Tensor(x)
        task_action_count = 3

        # Act
        timesteps, collected_data, rewards_reported = runner.collect_data(time_batch_size=time_batch_size,
                                                                          env_spec=mock_env_spec,
                                                                          preprocessor=mock_preprocessor,
                                                                          task_action_count=task_action_count)

        # Assert
        # Basic return checks
        assert timesteps == 123, "Number of timesteps returned inaccurate. Got {timesteps}."
        assert len(collected_data) == 123, f"Amount of collected data unexpected. Got {len(collected_data)}."
        assert len(rewards_reported) == 0, "Rewards were reported when none were expected."

        # Check that MockInfoToStore is getting properly updated
        assert isinstance(collected_data[0], MockInfoToStore), "Unexpected InfoToStore returned."
        assert collected_data[0].reward == 1.5, "MockInfoToStore not correctly populated with reward."
        assert not collected_data[0].done, "MockInfoToStore not correctly populated with done."

        # Check that the observation is being created correctly
        observation_to_policy, received_task_action_count = collected_data[0].data_to_store
        assert received_task_action_count == task_action_count, "task_action_count getting intercepted somehow."
        assert observation_to_policy.shape[0] == time_batch_size, "Time not being batched correctly"
        assert observation_to_policy.shape[1] == 3, "Incorrect obs shape"  # From how MockEnv is written, returning obs of length 3

