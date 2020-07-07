import torch
from collections import deque
from .environment_runner_base import EnvironmentRunnerBase
from continual_rl.utils.utils import Utils


class EnvironmentRunnerSync(EnvironmentRunnerBase):
    """
    An episode collection class that will collect the data synchronously, using one environment.

    The arguments provided to __init__ are from the policy.
    The arguments provided to collect_data are from the task.
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

        observations = deque(maxlen=time_batch_size)
        for _ in range(time_batch_size):
            observations.append(processed_observation)

        return observations

    def collect_data(self, time_batch_size, env_spec, preprocessor, task_action_count):
        """
        Provides actions to the policy in the form [time, **env.obs_shape]
        """
        environment_data = []

        if self._env is None:
            self._env = Utils.make_env(env_spec)

        for timestep_id in range(self._timesteps_per_collection):
            # The input to the policy is a collection of the latest time_batch_size observations
            # Assumes that if the observations are None, we should do a reset.
            if self._observations is None:
                self._observations = self._reset_env(time_batch_size, preprocessor)

            stacked_observations = torch.stack(list(self._observations))
            action, info_to_store = self._policy.compute_action(stacked_observations, task_action_count)
            next_obs, reward, done, _ = self._env.step(action)

            self._observations.append(preprocessor(next_obs))

            # Finish populating the info to store with the collected data
            info_to_store.reward = reward
            info_to_store.done = done

            # Store the data in the currently active episode data store
            environment_data.append(info_to_store)

            if done:
                self._observations = None  # Triggers a reset

        return self._timesteps_per_collection, environment_data
