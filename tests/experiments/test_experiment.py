import pytest
import os
from pathlib import Path
from continual_rl.experiments.experiment import Experiment, InvalidTaskAttributeException
from tests.common_mocks.mock_task import MockTask
from tests.common_mocks.mock_policy.mock_policy import MockPolicy
from tests.common_mocks.mock_policy.mock_policy_config import MockPolicyConfig


class TestExperiment(object):

    def test_get_common_attribute_all_scenarios(self):
        """
        Tests the Experiment helper function that ensures a common attribute across all entries, or fails when
        there's a mismatch. This code is sufficiently simple I'm not splitting these two tests up.
        """
        # Arrange
        vals_to_test_success = [3, 3, 3]
        vals_to_test_fail = [3, 2, 3]

        # Act
        common_val = Experiment._get_common_attribute(vals_to_test_success)

        with pytest.raises(InvalidTaskAttributeException):
            Experiment._get_common_attribute(vals_to_test_fail)

        # Assert
        assert common_val == 3, "Common value found"

    def test_get_action_spaces_success(self):
        """
        Tests that get_action_spaces produces a map from task id to the common size.
        """
        # Arrange
        # Using arbitrary task_ids because they shouldn't need to be sequential - and indeed may someday
        # be replaced with consistent UUIDs
        fake_tasks = [
            MockTask(task_id="act_spaces_1", action_space_id=12, env_spec=None, action_space=5, time_batch_size=3,
                     num_timesteps=None, eval_mode=None),
            MockTask(task_id="act_spaces_2", action_space_id="a38bc4", env_spec=None, action_space=10,
                     time_batch_size=3, num_timesteps=None, eval_mode=None),
            MockTask(task_id="act_spaces_3", action_space_id=12, env_spec=None, action_space=5, time_batch_size=3,
                     num_timesteps=None, eval_mode=None),
            MockTask(task_id="act_spaces_4", action_space_id="bbbbb", env_spec=None, action_space=240,
                     time_batch_size=3, num_timesteps=None, eval_mode=None)
        ]

        # Act
        action_spaces = Experiment._get_action_spaces(fake_tasks)

        # Assert
        assert action_spaces == {12: 5, "a38bc4": 10, "bbbbb": 240}

    def test_get_action_spaces_failure(self):
        """
        Tests that get_action_spaces fails to produces a map from task id to the common size when there's a mismatch
        """
        # Arrange
        # Using arbitrary task_ids because they shouldn't need to be sequential - and indeed may someday
        # be replaced with consistent UUIDs
        fake_tasks = [
            MockTask(task_id="act_spaces_fail_1", action_space_id=12, env_spec=None, action_space=5, time_batch_size=3,
                     num_timesteps=None, eval_mode=None),
            MockTask(task_id="act_spaces_fail_2", action_space_id="a38bc4", env_spec=None, action_space=10,
                     time_batch_size=3, num_timesteps=None, eval_mode=None),
            MockTask(task_id="act_spaces_fail_3", action_space_id=12, env_spec=None, action_space=15, time_batch_size=3,
                     num_timesteps=None, eval_mode=None),
            MockTask(task_id="act_spaces_fail_4", action_space_id="bbbbb", env_spec=None, action_space=240,
                     time_batch_size=3, num_timesteps=None, eval_mode=None)
        ]

        # Act & Assert
        with pytest.raises(InvalidTaskAttributeException):
            Experiment._get_action_spaces(fake_tasks)

    def test_policy_save(self, set_tmp_directory, cleanup_experiment, request):
        """
        Tests that the policy's save functionality is being called per the configuration
        """
        # Arrange
        output_dir = Path(request.node.experiment_output_dir, "test_policy_save")
        os.makedirs(output_dir)

        experiment = Experiment(tasks=[
            MockTask(task_id="save_test_0", action_space_id=12, env_spec=None, action_space=5, time_batch_size=3,
                     num_timesteps=100, eval_mode=None)])
        experiment.set_output_dir(output_dir)

        config = MockPolicyConfig()
        config.timesteps_per_save = 20
        policy = MockPolicy(config, None, None)

        # Act
        experiment.try_run(policy, None)  # Runs 10 timesteps every call (per the mock)

        # Assert
        assert policy.save_count == 7  # One at the start and an extra one at the end
        assert policy.load_count == 1

    def test_multitask_policy_save(self, set_tmp_directory, cleanup_experiment, request):
        """
        Tests that the policy's save functionality is being called per the configuration
        """
        # Arrange
        output_dir = Path(request.node.experiment_output_dir, "test_multitask_policy_save")
        os.makedirs(output_dir)

        experiment = Experiment(tasks=[
            MockTask(task_id="save_test_1", action_space_id=12, env_spec=None, action_space=5, time_batch_size=3,
                     num_timesteps=80, eval_mode=None),
            MockTask(task_id="save_test_2", action_space_id=12, env_spec=None, action_space=5, time_batch_size=3,
                     num_timesteps=80, eval_mode=None),
            MockTask(task_id="save_test_3", action_space_id=12, env_spec=None, action_space=5, time_batch_size=3,
                     num_timesteps=80, eval_mode=None)
        ])
        experiment.set_output_dir(output_dir)

        config = MockPolicyConfig()
        config.timesteps_per_save = 20
        policy = MockPolicy(config, None, None)

        # Act
        experiment.try_run(policy, None)  # Runs 10 timesteps every call (per the mock)

        # Assert
        assert policy.save_count == 18  # One at the start and an extra one at the end
        assert policy.load_count == 1
