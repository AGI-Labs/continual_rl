from collections import deque
from .environment_runner_base import EnvironmentRunnerBase
from continual_rl.utils.utils import Utils


class EnvironmentRunnerSync(EnvironmentRunnerBase):
    """
    An episode collection class that will collect the data synchronously, using one environment.
    """
    def __init__(self, policy, timesteps_per_collection):
        super().__init__()
        self._policy = policy
        self._timesteps_per_collection = timesteps_per_collection
        self._env = None
        self._observations = None

    def _reset_env(self, time_batch_size, preprocessor):
        # Initialize the observation time-batch with n of the first observation.
        raw_observation = self._env.reset()
        processed_observation = preprocessor(raw_observation)

        self._observations = deque(maxlen=time_batch_size)
        for _ in range(time_batch_size):
            self._observations.append(processed_observation)

    def collect_data(self, time_batch_size, env_spec, preprocessor, task_action_count):
        environment_data = []

        if self._env is None:
            self._env = Utils.make_env(env_spec)

        for timestep_id in range(self._timesteps_per_collection):
            # The input to the policy is a collection of the latest time_batch_size observations
            # Assumes that if the observations are None, we should do a reset.
            if self._observations is None:
                self._reset_env(time_batch_size, preprocessor)

            action, info_to_store = self._policy.compute_action(self._observations, task_action_count)
            next_obs, reward, done, _ = self._env.step(action)

            self._observations.append(preprocessor(next_obs))

            # Store the data in the currently active episode data store
            environment_data.append((info_to_store, reward, done))

            if done:
                self._observations = None  # Triggers a reset

        # The expected return is a list of lists (all data collected across all environments).
        # Since we are only running over one environment, put it in a list accordingly.
        return [environment_data]
