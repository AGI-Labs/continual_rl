import torch
import numpy as np
from continual_rl.experiments.environment_runners.environment_runner_batch import EnvironmentRunnerBatch
from tests.common_mocks.mock_policy.mock_policy import MockPolicy
from tests.common_mocks.mock_policy.mock_policy_config import MockPolicyConfig
from tests.common_mocks.mock_policy.mock_info_to_store import MockInfoToStore


class MockEnv(object):
    def __init__(self):
        self.actions_executed = []
        self.reset_count = 0
        self.observation_space = 4
        self.action_space = 4

    def reset(self):
        self.reset_count += 1
        return np.array([0, 1, 2])

    def step(self, action):
        self.actions_executed.append(action)
        observation = np.array([12, 13, 14])
        reward = 1.5
        done = action == 4  # Simple way to force the done state we want
        return observation, reward, done, {"info": "unused"}


class TestEnvironmentRunnerBatch(object):

    def test_collect_data_simple_success(self, monkeypatch):
        """
        Setup the simple happy-path for collect_data, and make sure things are being populated correctly.
        Simple: no done=True, no rewards returned, etc.
        """
        # Arrange
        def mock_compute_action(_, observation, action_space_id):
            # Since we're using the Batch runner, it expects a vector
            action = [3] * len(observation)
            return action, MockInfoToStore(data_to_store=(observation, action_space_id))

        # Mock the policy we're running; action_space and observation_size not used.
        mock_policy = MockPolicy(MockPolicyConfig(), action_spaces=None, observation_size=None)
        monkeypatch.setattr(MockPolicy, "compute_action", mock_compute_action)

        # The object under test
        runner = EnvironmentRunnerBatch(policy=mock_policy, num_parallel_envs=12, timesteps_per_collection=123)

        # Arguments to collect_data
        time_batch_size = 6

        mock_env = MockEnv()  # Useful for determining that parameters are getting generated and passed correctly
        mock_env_spec = lambda: mock_env  # Normally should create a new one each time, but doing this for spying
        mock_preprocessor = lambda x: torch.Tensor(x)
        action_space_id = 3

        # Act
        timesteps, collected_data, rewards_reported = runner.collect_data(time_batch_size=time_batch_size,
                                                                          env_spec=mock_env_spec,
                                                                          preprocessor=mock_preprocessor,
                                                                          action_space_id=action_space_id)

        # Assert
        # Basic return checks
        assert timesteps == 123 * 12, f"Number of timesteps returned inaccurate. Got {timesteps}."
        assert len(collected_data) == 123, f"Amount of collected data unexpected. Got {len(collected_data)}."
        assert len(rewards_reported) == 0, "Rewards were reported when none were expected."

        # Check that MockInfoToStore is getting properly updated
        assert isinstance(collected_data[0], MockInfoToStore), "Unexpected InfoToStore returned."
        assert np.all(np.array([entry.reward for entry in collected_data]) == 1.5), \
            "MockInfoToStore not correctly populated with reward."
        assert not np.any(np.array([entry.done for entry in collected_data])), \
            "MockInfoToStore not correctly populated with done."

        # Check that the observation is being created correctly
        observation_to_policy, received_action_space_id = collected_data[0].data_to_store
        assert received_action_space_id == action_space_id, "action_space_id getting intercepted somehow."
        assert observation_to_policy.shape[0] == 12, "Envs not being batched correctly"
        assert observation_to_policy.shape[1] == time_batch_size, "Time not being batched correctly"

        # 3 is from how MockEnv is written, which returns observations of length 3
        assert observation_to_policy.shape[2] == 3, "Incorrect obs shape"

        # Use our environment spy to check it's being called correctly
        # All env params are *1 not *12 because the first env is done local to the current process, so this is only
        # one env's worth, even though technically we're returning the same env every time (for spying purposes)
        # It's odd.
        assert mock_env.reset_count == 1, f"Mock env reset an incorrect number of times: {mock_env.reset_count}"
        assert len(mock_env.actions_executed) == 123, "Mock env.step not called a sufficient number of times"
        assert np.all(np.array(mock_env.actions_executed) == 3), "Incorrect action taken"

    def test_collect_data_with_intermediate_dones(self, monkeypatch):
        """
        Setup an environment that gives "done" at some point during the run
        """
        # Arrange
        current_step = 0

        def mock_compute_action(_, observation, action_space_id):
            nonlocal current_step
            action = [4 if current_step == 73 else 3] * len(observation)  # 4 is the "done" action, 3 is arbitrary

            current_step += 1
            return action, MockInfoToStore(data_to_store=(observation, action_space_id))

        # Mock the policy we're running. action_space and observation_size not used.
        mock_policy = MockPolicy(MockPolicyConfig(), action_spaces=None, observation_size=None)
        monkeypatch.setattr(MockPolicy, "compute_action", mock_compute_action)

        # The object under test
        runner = EnvironmentRunnerBatch(policy=mock_policy, num_parallel_envs=12, timesteps_per_collection=123)

        # Arguments to collect_data
        time_batch_size = 7

        mock_env = MockEnv()
        mock_env_spec = lambda: mock_env  # Normally should create a new one each time, but doing this for spying
        mock_preprocessor = lambda x: torch.Tensor(x)
        action_space_id = 6

        # Act
        timesteps, collected_data, rewards_reported = runner.collect_data(time_batch_size=time_batch_size,
                                                                          env_spec=mock_env_spec,
                                                                          preprocessor=mock_preprocessor,
                                                                          action_space_id=action_space_id)

        # Assert
        # Basic return checks
        assert timesteps == 123 * 12, f"Number of timesteps returned inaccurate. Got {timesteps}."
        assert len(collected_data) == 123, f"Amount of collected data unexpected. Got {len(collected_data)}."
        assert len(rewards_reported) == 12, "Rewards were not reported when multiple were expected."
        assert np.all(np.array(rewards_reported) == 74 * 1.5), f"Value of reward reported unexpected {rewards_reported}"

        # Check that MockInfoToStore is getting properly updated
        assert isinstance(collected_data[0], MockInfoToStore), "Unexpected InfoToStore returned."
        assert not np.any(np.array([entry.done for entry in collected_data[:73]])), \
            "MockInfoToStore not correctly populated with done."
        assert not np.any(np.array([entry.done for entry in collected_data[74:]])), \
            "MockInfoToStore not correctly populated with done."
        assert np.all(collected_data[73].done), "MockInfoToStore not correctly populated with done."

        # Check that the observation is being created correctly
        observation_to_policy, received_action_space_id = collected_data[0].data_to_store
        assert received_action_space_id == action_space_id, "action_space_id getting intercepted somehow."
        assert observation_to_policy.shape[0] == 12, "Envs not being batched correctly"
        assert observation_to_policy.shape[1] == time_batch_size, "Time not being batched correctly"

        # 3 is from how MockEnv is written, which returns observations of length 3
        assert observation_to_policy.shape[2] == 3, "Incorrect obs shape"

        # Use our environment spy to check it's being called correctly
        # All env params are *1 not *12 because the first env is done local to the current process, so this is only
        # one env's worth, even though technically we're returning the same env every time (for spying purposes)
        # It's odd.
        assert mock_env.reset_count == 2, f"Mock env reset an incorrect number of times: {mock_env.reset_count}"
        assert len(mock_env.actions_executed) == 123, "Mock env.step not called a sufficient number of times"
        assert np.all(np.array(mock_env.actions_executed[:73]) == 3), "Incorrect action taken, first half"
        assert np.all(np.array(mock_env.actions_executed[74:]) == 3), "Incorrect action taken, second half"
        assert np.array(mock_env.actions_executed)[73] == 4, "Incorrect action taken at the 'done' step."

    def test_collect_data_multi_collect_before_done(self, monkeypatch):
        """
        Run two data collections, and make sure the rewards are collected successfully.
        """
        # Arrange
        # Mock methods
        current_step = 0

        def mock_compute_action(_, observation, action_space_id):
            nonlocal current_step
            action = [4 if current_step == 73 else 3] * len(observation)  # 4 is the "done" action, 3 is arbitrary

            current_step += 1
            return action, MockInfoToStore(data_to_store=(observation, action_space_id))

        # Mock the policy we're running. action_space and observation_size not used.
        mock_policy = MockPolicy(MockPolicyConfig(), action_spaces=None, observation_size=None)
        monkeypatch.setattr(MockPolicy, "compute_action", mock_compute_action)

        # The object under test
        runner = EnvironmentRunnerBatch(policy=mock_policy, num_parallel_envs=12, timesteps_per_collection=50)

        # Arguments to collect_data
        time_batch_size = 7

        mock_env = MockEnv()
        mock_env_spec = lambda: mock_env  # Normally should create a new one each time, but doing this for spying
        mock_preprocessor = lambda x: torch.Tensor(x)
        action_space_id = 6

        # Act
        timesteps_0, collected_data_0, rewards_reported_0 = runner.collect_data(time_batch_size=time_batch_size,
                                                                                env_spec=mock_env_spec,
                                                                                preprocessor=mock_preprocessor,
                                                                                action_space_id=action_space_id)
        timesteps_1, collected_data_1, rewards_reported_1 = runner.collect_data(time_batch_size=time_batch_size,
                                                                                env_spec=mock_env_spec,
                                                                                preprocessor=mock_preprocessor,
                                                                                action_space_id=action_space_id)

        # Assert
        # Basic return checks
        assert timesteps_0 == timesteps_1 == 50 * 12, f"Number of timesteps returned inaccurate. " \
                                                 f"Got {(timesteps_0, timesteps_1)}."
        assert len(collected_data_0) == len(collected_data_1) == 50, f"Amount of collected data unexpected. " \
                                                                     f"Got {(len(collected_data_0), len(collected_data_1))}."
        assert len(rewards_reported_0) == 0, "Rewards were reported when none were expected."
        assert len(rewards_reported_1) == 12, "Rewards were not reported when multiple were expected."
        assert np.all(np.array(rewards_reported_1) == 74 * 1.5), f"Value of reward reported unexpected {rewards_reported_1}"

        # Check that MockInfoToStore is getting properly updated
        assert not np.any(np.array([entry.done for entry in collected_data_0])), \
            "MockInfoToStore not correctly populated with done."
        assert not np.any(np.array([entry.done for entry in collected_data_1[:23]])), \
            "MockInfoToStore not correctly populated with done."
        assert not np.any(np.array([entry.done for entry in collected_data_1[24:]])), \
            "MockInfoToStore not correctly populated with done."
        assert np.all(collected_data_1[23].done), "MockInfoToStore not correctly populated with done."

        # Use our environment spy to check it's being called correctly
        # All env params are *1 not *12 because the first env is done local to the current process, so this is only
        # one env's worth, even though technically we're returning the same env every time (for spying purposes)
        # It's odd. But kinda convenient I suppose.
        assert mock_env.reset_count == 2, f"Mock env reset an incorrect number of times: {mock_env.reset_count}"
        assert len(mock_env.actions_executed) == 100, "Mock env.step not called a sufficient number of times"
        assert np.all(np.array(mock_env.actions_executed[:73]) == 3), "Incorrect action taken, first half"
        assert np.all(np.array(mock_env.actions_executed[74:]) == 3), "Incorrect action taken, second half"
        assert np.array(mock_env.actions_executed)[73] == 4, "Incorrect action taken at the 'done' step."

    def test_collect_data_multi_collect_before_done_envs_finish_at_different_times(self, monkeypatch):
        """
        Run two data collections, and make sure the rewards are collected successfully even if not all envs finish together.
        """
        # Arrange
        # Mock methods
        current_step = 0

        def mock_compute_action(_, observation, action_space_id):
            nonlocal current_step
            action = [3] * len(observation)  # 4 is the "done" action, 3 is arbitrary

            # Make a pseudo-random environment finish, but not the others
            if current_step == 73:
                action[8] = 4

            current_step += 1
            return action, MockInfoToStore(data_to_store=(observation, action_space_id))

        # Mock the policy we're running. action_space and observation_size not used.
        mock_policy = MockPolicy(MockPolicyConfig(), action_spaces=None, observation_size=None)
        monkeypatch.setattr(MockPolicy, "compute_action", mock_compute_action)

        # The object under test
        runner = EnvironmentRunnerBatch(policy=mock_policy, num_parallel_envs=12, timesteps_per_collection=50)

        # Arguments to collect_data
        time_batch_size = 7

        mock_env = MockEnv()
        mock_env_spec = lambda: mock_env  # Normally should create a new one each time, but doing this for spying
        mock_preprocessor = lambda x: torch.Tensor(x)
        action_space_id = 6

        # Act
        timesteps_0, collected_data_0, rewards_reported_0 = runner.collect_data(time_batch_size=time_batch_size,
                                                                                env_spec=mock_env_spec,
                                                                                preprocessor=mock_preprocessor,
                                                                                action_space_id=action_space_id)
        timesteps_1, collected_data_1, rewards_reported_1 = runner.collect_data(time_batch_size=time_batch_size,
                                                                                env_spec=mock_env_spec,
                                                                                preprocessor=mock_preprocessor,
                                                                                action_space_id=action_space_id)

        # Assert
        # Basic return checks
        assert timesteps_0 == timesteps_1 == 50 * 12, f"Number of timesteps returned inaccurate. " \
                                                      f"Got {(timesteps_0, timesteps_1)}."
        assert len(collected_data_0) == len(collected_data_1) == 50, f"Amount of collected data unexpected. " \
                                                                     f"Got {(len(collected_data_0), len(collected_data_1))}."
        assert len(rewards_reported_0) == 0, "Rewards were reported when none were expected."
        assert len(rewards_reported_1) == 1, "Rewards were not reported when one was expected."
        assert rewards_reported_1[0] == 74 * 1.5, f"Value of reward reported unexpected {rewards_reported_1}"

        # Check that MockInfoToStore is getting properly updated
        assert not np.any(np.array([entry.done for entry in collected_data_0])), \
            "MockInfoToStore not correctly populated with done."
        assert not np.any(np.array([entry.done for entry in collected_data_1[:23]])), \
            "MockInfoToStore not correctly populated with done."
        assert not np.any(np.array([entry.done for entry in collected_data_1[24:]])), \
            "MockInfoToStore not correctly populated with done."
        assert not np.any(collected_data_1[23].done[:8]), "MockInfoToStore not correctly populated with done."
        assert not np.any(collected_data_1[23].done[9:]), "MockInfoToStore not correctly populated with done."
        assert collected_data_1[23].done[8], "MockInfoToStore not correctly populated with done."

        # Use our environment spy to check it's being called correctly
        # All env params are *1 not *12 because the first env is done local to the current process, so this is only
        # one env's worth, even though technically we're returning the same env every time (for spying purposes)
        # It's odd. But kinda convenient I suppose.
        assert mock_env.reset_count == 1, f"Mock env reset an incorrect number of times: {mock_env.reset_count}"
        assert len(mock_env.actions_executed) == 100, "Mock env.step not called a sufficient number of times"
        assert np.all(np.array(mock_env.actions_executed) == 3), "Incorrect action taken by the first env"

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
        def mock_compute_action(_, observation, action_space_id):
            # Since we're using the Batch runner, it expects a vector
            action = [3] * len(observation)
            return action, MockInfoToStore(data_to_store=(observation, action_space_id))

        mock_policy = MockPolicy(MockPolicyConfig(), action_spaces=None, observation_size=None)
        monkeypatch.setattr(MockPolicy, "compute_action", mock_compute_action)

        # The object under test
        runner = EnvironmentRunnerBatch(policy=mock_policy, num_parallel_envs=1, timesteps_per_collection=10)

        mock_env = MockEnv()
        monkeypatch.setattr(MockEnv, "step", mock_step)

        mock_env_spec = lambda: mock_env  # Normally should create a new one each time, but doing this for spying
        mock_preprocessor = lambda x: torch.Tensor(x)
        time_batch_size = 4
        action_space_id = 0

        # Act
        timesteps, collected_data, rewards_reported = runner.collect_data(time_batch_size=time_batch_size,
                                                                          env_spec=mock_env_spec,
                                                                          preprocessor=mock_preprocessor,
                                                                          action_space_id=action_space_id)

        # Assert
        # From the reset()
        assert np.all(collected_data[0].data_to_store[0].numpy() == np.array([[0,1,2], [0,1,2], [0,1,2], [0,1,2]]))
        assert np.all(collected_data[1].data_to_store[0].numpy() == np.array([[0,1,2], [0,1,2], [0,1,2], [100, 101, 102]]))
        assert np.all(collected_data[2].data_to_store[0].numpy() == np.array([[0,1,2], [0,1,2], [100, 101, 102], [101, 102, 103]]))
        assert np.all(collected_data[3].data_to_store[0].numpy() == np.array([[0,1,2], [100, 101, 102], [101, 102, 103], [102, 103, 104]]))
        assert np.all(collected_data[4].data_to_store[0].numpy() == np.array([[100, 101, 102], [101, 102, 103], [102, 103, 104], [103, 104, 105]]))
        assert np.all(collected_data[5].data_to_store[0].numpy() == np.array([[101, 102, 103], [102, 103, 104], [103, 104, 105], [104, 105, 106]]))
        assert np.all(collected_data[6].data_to_store[0].numpy() == np.array([[102, 103, 104], [103, 104, 105], [104, 105, 106], [105, 106, 107]]))
