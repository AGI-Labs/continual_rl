import numpy as np
from pathlib import Path
from tests.common_mocks.mock_task import MockTask
from tests.common_mocks.mock_policy.mock_policy import MockPolicy
from tests.common_mocks.mock_policy.mock_policy_config import MockPolicyConfig


class MockEnv(object):
    def __init__(self):
        pass

    def seed(self, seed):
        self.seed_set = seed

    def reset(self):
        return np.array([0, 1, 2])

    def step(self, action):
        observation = np.array([12, 13, 14])
        reward = 1.5
        done = action == 4  # Simple way to force the done state we want
        return observation, reward, done, {"info": "unused"}


class TestTaskBase(object):

    def test_run_train(self, set_tmp_directory, cleanup_experiment, request):
        """
        Sanity check running a "train" primary task.
        """
        # Arrange
        # Using our mock task because we're testing the base class specifically (so thin wrapper on it)
        task = MockTask(action_space_id=0, env_spec=lambda: MockEnv(), action_space=[5, 3], time_batch_size=3,
                        num_timesteps=23, eval_mode=False)
        config = MockPolicyConfig()  # Uses MockEnvironmentRunner
        policy = MockPolicy(config, observation_space=None, action_spaces=None)  # collect_data not called (MockRunner)
        Path(request.node.experiment_output_dir).mkdir()  # TaskBase doesn't create the folder, so create it here

        # Act
        task_runner = task.run(run_id=0, policy=policy, summary_writer=None,
                               output_dir=request.node.experiment_output_dir)

        timesteps = next(task_runner)

        # Assert
        assert policy.train_run_count == 1, "Train should only be called once"
        assert timesteps == 10, "Timesteps incorrect"  # Specified by the EnvironmentRunner

    def test_run_eval(self):
        # Arrange
        # Using our mock task because we're testing the base class specifically (so thin wrapper on it)
        task = MockTask(action_space_id=0, env_spec=lambda: MockEnv(), action_space=[5, 3], time_batch_size=3,
                        num_timesteps=23, eval_mode=True)

        # Act

        # Assert
        pass

    def test_continual_eval(self):
        # Arrange

        # Act

        # Assert
        pass
