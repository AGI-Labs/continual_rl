from collections import deque
from .environment_runner_base import EnvironmentRunnerBase


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

    def collect_data(self, time_batch_size, env_spec, preprocessor, task_action_count):
        if self._env is None:
            self._env = self.make_env(env_spec)

        # The input to the policy is a collection of the latest time_batch_size observations
        # Initialize the observation time-batch with n of the first observation.
        # Assumes that if the observations are None, we should do a reset.
        if self._observations is None:
            raw_observation = self._env.reset()
            processed_observation = preprocessor(raw_observation)

            self._observations = deque(maxlen=time_batch_size)
            for _ in range(time_batch_size):
                self._observations.append(processed_observation)

        for timestep_id in range(self._timesteps_per_collection):
            action, info_to_store = self._policy.compute_action(self._observations, task_action_count)
            next_obs, reward, done, _ = self._env.step(action)

