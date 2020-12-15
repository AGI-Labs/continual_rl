import numpy as np
from continual_rl.experiments.environment_runners.environment_runner_sync import EnvironmentRunnerSync
from continual_rl.experiments.tasks.task_base import TaskSpec
from tests.common_mocks.mock_policy.mock_policy import MockPolicy
from tests.common_mocks.mock_policy.mock_policy_config import MockPolicyConfig
from tests.common_mocks.mock_policy.mock_timestep_data import MockTimestepData
from tests.common_mocks.mock_preprocessor import MockPreprocessor


class MockEnv(object):
    def __init__(self):
        self.actions_executed = []
        self.reset_count = 0
        self.observation_space = [1, 2, 3]
        self.action_space = [4, 5]

    def seed(self, seed):
        self.seed_set = seed

    def reset(self):
        self.reset_count += 1
        return np.array([0, 1, 2])

    def step(self, action):
        self.actions_executed.append(action)
        observation = np.array([12, 13, 14])
        reward = 1.5
        done = action == 4  # Simple way to force the done state we want
        return observation, reward, done, {"info": "unused"}


class TestEnvironmentRunnerSync(object):

    def test_collect_data_simple_success(self, monkeypatch):
        """
        Setup the simple happy-path for collect_data, and make sure things are being populated correctly.
        Simple: no done=True, no rewards returned, etc.
        """
        # Arrange
        def mock_compute_action(_, observation, action_space_id, last_timestep_data, eval_mode):
            action = [3]
            timestep_data = MockTimestepData(data_to_store=(observation, action_space_id, eval_mode))

            if last_timestep_data is None:
                timestep_data.memory = 0
            else:
                timestep_data.memory = last_timestep_data.memory + 1

            return action, timestep_data

        # Mock the policy we're running; action_space and observation_space not used.
        mock_policy = MockPolicy(MockPolicyConfig(), action_spaces=None, observation_space=None)
        monkeypatch.setattr(MockPolicy, "compute_action", mock_compute_action)

        # The object under test
        runner = EnvironmentRunnerSync(policy=mock_policy, timesteps_per_collection=123)

        # Arguments to collect_data
        # Normally should create a new one each time, but doing this for spying
        mock_env = MockEnv()
        mock_env_spec = lambda: mock_env

        # MockEnv is used for determining that parameters are getting generated and passed correctly
        task_spec = TaskSpec(action_space_id=3, preprocessor=MockPreprocessor(), env_spec=mock_env_spec,
                             time_batch_size=6, num_timesteps=9718, eval_mode=1964)

        # Act
        timesteps, collected_data, rewards_reported, _ = runner.collect_data(task_spec)

        # Assert
        # Basic return checks
        assert timesteps == 123, f"Number of timesteps returned inaccurate. Got {timesteps}."
        assert len(collected_data) == 1, f"Amount of collected data unexpected. Got {len(collected_data)}."
        assert len(collected_data[0]) == 123, f"Amount of collected data unexpected. Got {len(collected_data[0])}."
        assert len(rewards_reported) == 0, "Rewards were reported when none were expected."

        # Check that MockTimestepData is getting properly updated
        collected_data = collected_data[0]
        assert isinstance(collected_data[0], MockTimestepData), "Unexpected TimestepData returned."
        assert np.all(np.array([entry.reward for entry in collected_data]) == 1.5), \
            "MockTimestepData not correctly populated with reward."
        assert not np.any(np.array([entry.done for entry in collected_data])), \
            "MockTimestepData not correctly populated with done."
        assert collected_data[0].memory == 0, "compute_action not correctly receiving last_timestep_data."
        assert collected_data[1].memory == 1, "compute_action not correctly receiving last_timestep_data."
        assert collected_data[78].memory == 78, "compute_action not correctly receiving last_timestep_data."

        # Check that the observation is being created correctly
        observation_to_policy, received_action_space_id, observed_eval_mode = collected_data[0].data_to_store
        assert received_action_space_id == 3, "action_space_id getting intercepted somehow."
        assert observation_to_policy.shape[0] == 1, "'Fake' batch missing"
        assert observation_to_policy.shape[1] == 6, "Time not being batched correctly"
        assert observed_eval_mode == 1964, "Eval_mode not passed correctly"

        # 3 is from how MockEnv is written, which returns observations of length 3
        assert observation_to_policy.shape[2] == 3, "Incorrect obs shape"

        # Use our environment spy to check it's being called correctly
        assert mock_env.reset_count == 1, f"Mock env reset an incorrect number of times: {mock_env.reset_count}"
        assert len(mock_env.actions_executed) == 123, "Mock env.step not called a sufficient number of times"
        assert np.all(np.array(mock_env.actions_executed) == 3), "Incorrect action taken"
        assert mock_env.seed_set is not None, "Seed not being set"

    def test_collect_data_with_intermediate_dones(self, monkeypatch):
        """
        Setup an environment that gives "done" at some point during the run
        """
        # Arrange
        current_step = 0

        def mock_compute_action(_, observation, action_space_id, last_timestep_data, eval_mode):
            nonlocal current_step
            action = [4] if current_step == 73 else [3]  # 4 is the "done" action, 3 is arbitrary
            current_step += 1
            timestep_data = MockTimestepData(data_to_store=(observation, action_space_id, eval_mode))

            if last_timestep_data is None:
                timestep_data.memory = 0
            else:
                timestep_data.memory = last_timestep_data.memory + 1

            return action, timestep_data

        # Mock the policy we're running. action_space and observation_space not used.
        mock_policy = MockPolicy(MockPolicyConfig(), action_spaces=None, observation_space=None)
        monkeypatch.setattr(MockPolicy, "compute_action", mock_compute_action)

        # The object under test
        runner = EnvironmentRunnerSync(policy=mock_policy, timesteps_per_collection=123)

        # Arguments to collect_data
        # Normally should create a new one each time, but doing this for spying
        mock_env = MockEnv()
        mock_env_spec = lambda: mock_env

        # MockEnv is used for determining that parameters are getting generated and passed correctly
        task_spec = TaskSpec(action_space_id=6, preprocessor=MockPreprocessor(), env_spec=mock_env_spec,
                             time_batch_size=7, num_timesteps=9718, eval_mode=1964)

        # Act
        timesteps, collected_data, rewards_reported, _ = runner.collect_data(task_spec)

        # Assert
        # Basic return checks
        assert timesteps == 123, f"Number of timesteps returned inaccurate. Got {timesteps}."
        assert len(collected_data) == 1, f"Amount of collected data unexpected. Got {len(collected_data)}."
        assert len(collected_data[0]) == 123, f"Amount of collected data unexpected. Got {len(collected_data[0])}."
        assert len(rewards_reported) == 1, "Rewards were not reported when one was expected."
        assert rewards_reported[0] == 74 * 1.5, f"Value of reward reported unexpected {rewards_reported}"

        # Check that MockTimestepData is getting properly updated
        collected_data = collected_data[0]
        assert isinstance(collected_data[0], MockTimestepData), "Unexpected TimestepData returned."
        assert not np.any(np.array([entry.done for entry in collected_data[:73]])), \
            "MockTimestepData not correctly populated with done."
        assert not np.any(np.array([entry.done for entry in collected_data[74:]])), \
            "MockTimestepData not correctly populated with done."
        assert collected_data[73].done, "MockTimestepData not correctly populated with done."
        assert collected_data[78].memory == 78, "compute_action not correctly receiving last_timestep_data. " \
                                                "(Always populated, even if a done occurred.)"

        # Check that the observation is being created correctly
        observation_to_policy, received_action_space_id, observed_eval_mode = collected_data[0].data_to_store
        assert received_action_space_id == 6, "action_space_id getting intercepted somehow."
        assert observation_to_policy.shape[0] == 1, "'Fake' batch appearing in correctly"
        assert observation_to_policy.shape[1] == 7, "Time not being batched correctly"
        assert observed_eval_mode == 1964, "Eval_mode not passed correctly"

        # 3 is from how MockEnv is written, which returns observations of length 3
        assert observation_to_policy.shape[2] == 3, "Incorrect obs shape"

        # Use our environment spy to check it's being called correctly
        assert mock_env.reset_count == 2, f"Mock env reset an incorrect number of times: {mock_env.reset_count}"
        assert len(mock_env.actions_executed) == 123, "Mock env.step not called a sufficient number of times"
        assert np.all(np.array(mock_env.actions_executed[:73]) == 3), "Incorrect action taken, first half"
        assert np.all(np.array(mock_env.actions_executed[74:]) == 3), "Incorrect action taken, second half"
        assert np.array(mock_env.actions_executed)[73] == 4, "Incorrect action taken at the 'done' step."

    def test_collect_data_multi_collect_before_done(self, monkeypatch):
        """
        Run two data collections, and
        """
        # Arrange
        # Mock methods
        current_step = 0

        def mock_compute_action(_, observation, action_space_id, last_timestep_data, eval_mode):
            nonlocal current_step
            action = [4] if current_step == 73 else [3]  # 4 is the "done" action, 3 is arbitrary
            current_step += 1
            timestep_data = MockTimestepData(data_to_store=(observation, action_space_id, eval_mode))

            if last_timestep_data is None:
                timestep_data.memory = 0
            else:
                timestep_data.memory = last_timestep_data.memory + 1
            return action, timestep_data

        # Mock the policy we're running. action_space and observation_space not used.
        mock_policy = MockPolicy(MockPolicyConfig(), action_spaces=None, observation_space=None)
        monkeypatch.setattr(MockPolicy, "compute_action", mock_compute_action)

        # The object under test
        runner = EnvironmentRunnerSync(policy=mock_policy, timesteps_per_collection=50)

        # Arguments to collect_data
        # Normally should create a new one each time, but doing this for spying
        mock_env = MockEnv()
        mock_env_spec = lambda: mock_env

        # MockEnv is used for determining that parameters are getting generated and passed correctly
        task_spec = TaskSpec(action_space_id=6, preprocessor=MockPreprocessor(), env_spec=mock_env_spec,
                             time_batch_size=7, num_timesteps=9718, eval_mode=1964)

        # Act
        timesteps_0, collected_data_0, rewards_reported_0, _ = runner.collect_data(task_spec)
        timesteps_1, collected_data_1, rewards_reported_1, _ = runner.collect_data(task_spec)

        # Assert
        # Basic return checks
        assert timesteps_0 == timesteps_1 == 50, f"Number of timesteps returned inaccurate. " \
                                                 f"Got {(timesteps_0, timesteps_1)}."
        assert len(collected_data_0[0]) == len(collected_data_1[0]) == 50, f"Amount of collected data unexpected. " \
                                                                     f"Got {(len(collected_data_0), len(collected_data_1))}."
        assert len(rewards_reported_0) == 0, "Rewards were reported when none were expected."
        assert len(rewards_reported_1) == 1, "Rewards were not reported when one was expected."
        assert rewards_reported_1[0] == 74 * 1.5, f"Value of reward reported unexpected {rewards_reported_1}"

        # Check that MockTimestepData is getting properly updated
        collected_data_0 = collected_data_0[0]
        collected_data_1 = collected_data_1[0]
        assert not np.any(np.array([entry.done for entry in collected_data_0])), \
            "MockTimestepData not correctly populated with done."
        assert not np.any(np.array([entry.done for entry in collected_data_1[:23]])), \
            "MockTimestepData not correctly populated with done."
        assert not np.any(np.array([entry.done for entry in collected_data_1[24:]])), \
            "MockTimestepData not correctly populated with done."
        assert collected_data_1[23].done, "MockTimestepData not correctly populated with done."
        assert collected_data_1[45].memory == 95, "MockTimestepData not correctly populated with done."

        # Use our environment spy to check it's being called correctly
        assert mock_env.reset_count == 2, f"Mock env reset an incorrect number of times: {mock_env.reset_count}"
        assert len(mock_env.actions_executed) == 100, "Mock env.step not called a sufficient number of times"
        assert np.all(np.array(mock_env.actions_executed[:73]) == 3), "Incorrect action taken, first half"
        assert np.all(np.array(mock_env.actions_executed[74:]) == 3), "Incorrect action taken, second half"
        assert np.array(mock_env.actions_executed)[73] == 4, "Incorrect action taken at the 'done' step."

    def test_collect_data_time_batching_correct(self, monkeypatch):
        """
        Make sure that the time batching is doing the right thing - collecting sequential data, and correctly
        throwing out the old index.
        """
        # Arrange
        current_step = 0

        # A mock that gets sequential, easily predictable observations
        # Reset() produces [0, 1, 2], so we offset by 100+ to easily distinguish
        def mock_step(_, action):
            nonlocal current_step
            observation = np.array([current_step+100, current_step+101, current_step+102])
            reward = 1.5
            done = action == 4  # Simple way to force the done state we want

            current_step += 1
            return observation, reward, done, {"info": "unused"}

        # A mock that spies on the observations we've seen (and puts them in the DataToStore)
        def mock_compute_action(_, observation, action_space_id, last_timestep_data, eval_mode):
            # Since we're using the Batch runner, it expects a vector
            action = [3]
            return action, MockTimestepData(data_to_store=(observation, action_space_id, eval_mode))

        mock_policy = MockPolicy(MockPolicyConfig(), action_spaces=None, observation_space=None)
        monkeypatch.setattr(MockPolicy, "compute_action", mock_compute_action)

        # The object under test
        runner = EnvironmentRunnerSync(policy=mock_policy, timesteps_per_collection=10)

        # Normally should create a new one each time, but doing this for spying
        mock_env = MockEnv()
        monkeypatch.setattr(MockEnv, "step", mock_step)
        mock_env_spec = lambda: mock_env

        # MockEnv is used for determining that parameters are getting generated and passed correctly
        task_spec = TaskSpec(action_space_id=0, preprocessor=MockPreprocessor(), env_spec=mock_env_spec,
                             time_batch_size=4, num_timesteps=9718, eval_mode=1964)

        # Act
        timesteps, collected_data, rewards_reported, _ = runner.collect_data(task_spec)

        # Assert
        # From the reset()
        collected_data = collected_data[0]  # Sync is a list of lists, where the outer is of len 1
        assert np.all(collected_data[0].data_to_store[0].numpy() == np.array([[0,1,2], [0,1,2], [0,1,2], [0,1,2]]))
        assert np.all(collected_data[1].data_to_store[0].numpy() == np.array([[0,1,2], [0,1,2], [0,1,2], [100, 101, 102]]))
        assert np.all(collected_data[2].data_to_store[0].numpy() == np.array([[0,1,2], [0,1,2], [100, 101, 102], [101, 102, 103]]))
        assert np.all(collected_data[3].data_to_store[0].numpy() == np.array([[0,1,2], [100, 101, 102], [101, 102, 103], [102, 103, 104]]))
        assert np.all(collected_data[4].data_to_store[0].numpy() == np.array([[100, 101, 102], [101, 102, 103], [102, 103, 104], [103, 104, 105]]))
        assert np.all(collected_data[5].data_to_store[0].numpy() == np.array([[101, 102, 103], [102, 103, 104], [103, 104, 105], [104, 105, 106]]))
        assert np.all(collected_data[6].data_to_store[0].numpy() == np.array([[102, 103, 104], [103, 104, 105], [104, 105, 106], [105, 106, 107]]))
