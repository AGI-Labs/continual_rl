import pytest
from pathlib import Path
from tests.common_mocks.mock_task import MockTask
from tests.common_mocks.mock_policy.mock_policy import MockPolicy
from tests.common_mocks.mock_policy.mock_policy_config import MockPolicyConfig


class TestTaskBase(object):

    def test_run_train(self, set_tmp_directory, cleanup_experiment, request):
        """
        Sanity check running a "train" primary task. This is dependent on the MockEnvironmentRunner, which generates
        the timesteps and rewards. (Specifically it does not depend on a MockEnv.)
        """
        # Arrange
        # Using our mock task because we're testing the base class specifically (so thin wrapper on it)
        # The env_spec is not used by the MockEnvRunner, so don't even populate it.
        timestep_start = 1234
        task = MockTask(action_space_id=0, env_spec=lambda: None, action_space=[5, 3], time_batch_size=3,
                        num_timesteps=timestep_start+23, eval_mode=False)
        config = MockPolicyConfig()
        policy = MockPolicy(config, observation_space=None, action_spaces=None)  # collect_data not called (MockRunner)
        Path(request.node.experiment_output_dir).mkdir()  # TaskBase doesn't create the folder, so create it here

        # Act & Assert
        task_runner = task.run(run_id=0, policy=policy, summary_writer=None,
                               output_dir=request.node.experiment_output_dir, task_timestep_start=timestep_start)

        for step in range(3):
            task_timesteps, data_returned = next(task_runner)

            assert data_returned[0] == [step, step+1, step+2], "Reward returned was not as expected"
            assert data_returned[1][0]['step_count'] == 456+step, "Log was not as expected"
            assert not data_returned[1][1]['eval_mode'], "Log of eval_mode was not as expected"
            assert task_timesteps == 10*(step+1)+timestep_start, "Timesteps incorrect"
            assert policy.train_run_count == step+1, "Train count incorrect"
            assert not policy.current_env_runner.cleanup_called, "Cleanup shouldn't be called until the run is done"

        # In this step we break out of the loop and finish. Values should be the same as last
        final_timesteps, final_data_returned = next(task_runner)
        assert final_timesteps == 30+timestep_start, "Timesteps incorrect"
        assert policy.train_run_count == 3, "Train count incorrect"
        assert final_data_returned is None, "Data should be returned continuously, not saved to the end"
        assert policy.current_env_runner.cleanup_called, "Env should be cleanup up"

        # The task should be finished at this point
        with pytest.raises(StopIteration):
            next(task_runner)

    def test_run_eval(self, set_tmp_directory, cleanup_experiment, request):
        """
        Sanity check running an "eval" primary task. This is dependent on the MockEnvironmentRunner, which generates
        the timesteps and rewards. (Specifically it does not depend on a MockEnv.)
        Should be basically the same as "train" except that train doesn't get called.
        """
        # Arrange
        # Using our mock task because we're testing the base class specifically (so thin wrapper on it)
        # The env_spec is not used by the MockEnvRunner, so don't even populate it.
        timestep_start = 1234
        task = MockTask(action_space_id=0, env_spec=lambda: None, action_space=[5, 3], time_batch_size=3,
                        num_timesteps=timestep_start+23, eval_mode=True)
        config = MockPolicyConfig()  # Uses MockEnvironmentRunner
        policy = MockPolicy(config, observation_space=None, action_spaces=None)  # collect_data not called (MockRunner)
        Path(request.node.experiment_output_dir).mkdir()  # TaskBase doesn't create the folder, so create it here

        # Act & Assert
        task_runner = task.run(run_id=0, policy=policy, summary_writer=None,
                               output_dir=request.node.experiment_output_dir, task_timestep_start=timestep_start)

        for step in range(3):
            task_timesteps, data_returned = next(task_runner)

            assert data_returned[0] == [step, step+1, step+2], "Reward returned was not as expected"
            assert data_returned[1][0]['step_count'] == 456+step, "Log was not as expected"
            assert data_returned[1][1]['eval_mode'], "Log of eval_mode was not as expected"
            assert task_timesteps == 10*(step+1)+timestep_start, "Timesteps incorrect"
            assert policy.train_run_count == 0, "Train count incorrect"
            assert not policy.current_env_runner.cleanup_called, "Cleanup shouldn't be called until the run is done"

        # In this step we break out of the loop and finish. Values should be the same as last
        final_timesteps, final_data_returned = next(task_runner)
        assert final_timesteps == 30+timestep_start, "Timesteps incorrect"
        assert policy.train_run_count == 0, "Train count incorrect"
        assert final_data_returned is None, "Data should be returned continuously, not saved to the end"
        assert policy.current_env_runner.cleanup_called, "Env should be cleanup up"

        # The task should be finished at this point
        with pytest.raises(StopIteration):
            next(task_runner)

    def test_continual_eval(self, set_tmp_directory, cleanup_experiment, request):
        """
        Sanity check running continual eval task. This is dependent on the MockEnvironmentRunner, which generates
        the timesteps and rewards. (Specifically it does not depend on a MockEnv.)
        """
        # Arrange
        # Using our mock task because we're testing the base class specifically (so thin wrapper on it)
        # The env_spec is not used by the MockEnvRunner, so don't even populate it.
        task = MockTask(action_space_id=0, env_spec=lambda: None, action_space=[5, 3], time_batch_size=3,
                        num_timesteps=23, eval_mode=False)
        config = MockPolicyConfig()  # Uses MockEnvironmentRunner
        policy = MockPolicy(config, observation_space=None, action_spaces=None)  # collect_data not called (MockRunner)
        Path(request.node.experiment_output_dir).mkdir()  # TaskBase doesn't create the folder, so create it here

        # Act & Assert
        task_runner = task.continual_eval(run_id=0, policy=policy, summary_writer=None,
                                          output_dir=request.node.experiment_output_dir)

        for step in range(5):
            task_timesteps, data_returned = next(task_runner)

            # Currently the number of data points to collect for continual_eval is just hard-coded at 10.
            # Since we return 3 rewards per collection, and the return_after_episode_num triggers on the following
            # call, it should return non-None after 5 calls (step=4).
            assert (step == 4 and data_returned is not None) or (step != 4 and data_returned is None)
            assert data_returned is None or len(data_returned[0]) == 10, "Incorrect amount of data returned"
            assert data_returned is None or task_timesteps == 40, "Timesteps incorrect"
            assert policy.train_run_count == 0, "Train count incorrect"
            assert (step < 4 and not policy.current_env_runner.cleanup_called) or \
                   (step == 4 and policy.current_env_runner.cleanup_called), "Cleanup should only be called on the last"

        # The task should be finished at this point
        with pytest.raises(StopIteration):
            next(task_runner)

    def test_task_ids(self):
        """
        Ensure that task ids are getting created properly, and fail when a duplicate is found
        """
        # Arrange & Act (IDs created in constructor)
        task_1 = MockTask(task_id="mock_1", action_space_id=0, env_spec=lambda: None, action_space=[5, 3], time_batch_size=3,
                        num_timesteps=23, eval_mode=False)

        task_2 = MockTask(task_id="mock_2", action_space_id=0, env_spec=lambda: None, action_space=[5, 3], time_batch_size=3,
                        num_timesteps=23, eval_mode=False)

        # Fail when the id is the same
        with pytest.raises(AssertionError):
            task_2_duplicate = MockTask(task_id="mock_2", action_space_id=0, env_spec=lambda: None, action_space=[5, 3], time_batch_size=3,
                            num_timesteps=23, eval_mode=False)

        # Assert
        assert task_1._continual_eval_task_spec.task_id == "mock_1", "Task id not created properly"
        assert task_1._task_spec.task_id == "mock_1", "Task id not created properly"
        assert task_2._continual_eval_task_spec.task_id == "mock_2", "Task id not created properly"
        assert task_2._task_spec.task_id == "mock_2", "Task id not created properly"
