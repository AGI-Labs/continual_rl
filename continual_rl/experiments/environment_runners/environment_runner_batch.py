import torch
from collections import deque
from torch_ac.utils.penv import ParallelEnv
from .environment_runner_base import EnvironmentRunnerBase
from continual_rl.utils.utils import Utils


class EnvironmentRunnerBatch(EnvironmentRunnerBase):
    """
    Passes a batch of observations into the policy, gets a batch of actions out, and runs the environments in parallel.

    The arguments provided to __init__ are from the policy.
    The arguments provided to collect_data are from the task.
    """
    def __init__(self, policy, num_parallel_envs, timesteps_per_collection):
        super().__init__()
        self._policy = policy
        self._num_parallel_envs = num_parallel_envs
        self._timesteps_per_collection = timesteps_per_collection

        self._parallel_env = None
        self._observations = None

    def _preprocess_raw_observations(self, preprocessor, raw_observations):
        return torch.stack([preprocessor(raw_observation) for raw_observation in raw_observations])

    def _reset_env(self, time_batch_size, preprocessor):
        # Initialize the observation time-batch with n of the first observation.
        raw_observations = self._parallel_env.reset()
        processed_observations = self._preprocess_raw_observations(preprocessor, raw_observations)

        observations = deque(maxlen=time_batch_size)
        for _ in range(time_batch_size):
            observations.append(processed_observations)

        return observations

    def collect_data(self, time_batch_size, env_spec, preprocessor, task_action_count):
        """
        Passes observations to the policy of shape [time, envs, **env.obs_shape]
        """
        environment_data = []

        if self._parallel_env is None:
            envs = [Utils.make_env(env_spec) for _ in range(self._num_parallel_envs)]
            self._parallel_env = ParallelEnv(envs)

        for timestep_id in range(self._timesteps_per_collection):
            if self._observations is None:
                self._observations = self._reset_env(time_batch_size, preprocessor)

            stacked_observations = torch.stack(list(self._observations))
            actions, info_to_store = self._policy.compute_action(stacked_observations, task_action_count)

            result = self._parallel_env.step(actions)
            raw_observations, rewards, dones, infos = list(result)

            self._observations.append(self._preprocess_raw_observations(preprocessor, raw_observations))

            environment_data.append((info_to_store, rewards, dones))

        return environment_data
