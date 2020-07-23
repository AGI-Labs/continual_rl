import pytest
from continual_rl.experiments.experiment import Experiment, InvalidTaskAttributeException
from tests.common_mocks.mock_task import MockTask


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

    def test_get_action_sizes_success(self):
        """
        Tests that get_action_sizes produces a map from task id to the common size.
        """
        # Arrange
        # Using arbitrary task_ids because they shouldn't need to be sequential - and indeed may someday
        # be replaced with consistent UUIDs
        fake_tasks = [
            MockTask(action_space_id=12, env_spec=None, observation_size=[None], action_size=5, time_batch_size=None,
                     num_timesteps=None, eval_mode=None, output_dir=None),
            MockTask(action_space_id="a38bc4", env_spec=None, observation_size=[None], action_size=10, time_batch_size=None,
                     num_timesteps=None, eval_mode=None, output_dir=None),
            MockTask(action_space_id=12, env_spec=None, observation_size=[None], action_size=5, time_batch_size=None,
                     num_timesteps=None, eval_mode=None, output_dir=None),
            MockTask(action_space_id="bbbbb", env_spec=None, observation_size=[None], action_size=240, time_batch_size=None,
                     num_timesteps=None, eval_mode=None, output_dir=None)
        ]

        # Act
        action_sizes = Experiment._get_action_sizes(fake_tasks)

        # Assert
        assert action_sizes == {12: 5, "a38bc4": 10, "bbbbb": 240}

    def test_get_action_sizes_failure(self):
        """
        Tests that get_action_sizes fails to produces a map from task id to the common size when there's a mismatch
        """
        # Arrange
        # Using arbitrary task_ids because they shouldn't need to be sequential - and indeed may someday
        # be replaced with consistent UUIDs
        fake_tasks = [
            MockTask(action_space_id=12, env_spec=None, observation_size=[None], action_size=5, time_batch_size=None,
                     num_timesteps=None, eval_mode=None, output_dir=None),
            MockTask(action_space_id="a38bc4", env_spec=None, observation_size=[None], action_size=10, time_batch_size=None,
                     num_timesteps=None, eval_mode=None, output_dir=None),
            MockTask(action_space_id=12, env_spec=None, observation_size=[None], action_size=15, time_batch_size=None,
                     num_timesteps=None, eval_mode=None, output_dir=None),
            MockTask(action_space_id="bbbbb", env_spec=None, observation_size=[None], action_size=240, time_batch_size=None,
                     num_timesteps=None, eval_mode=None, output_dir=None)
        ]

        # Act & Assert
        with pytest.raises(InvalidTaskAttributeException):
            Experiment._get_action_sizes(fake_tasks)
