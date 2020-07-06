from abc import ABC, abstractmethod
import numpy as np


class TaskBase(ABC):
    def __init__(self, env_spec, observation_size, action_size, time_batch_size, num_timesteps, eval_mode, output_dir):
        self.observation_size = observation_size
        self.action_size = action_size
        self.time_batch_size = time_batch_size
        self._num_timesteps = num_timesteps
        self._env_spec = env_spec

    @abstractmethod
    def preprocess(self, observation):
        pass

    def run(self, policy, task_id, summary_writer):
        total_timesteps = 0
        environment_runner = policy.get_environment_runner()

        while total_timesteps < self._num_timesteps:
            # all_env_data is a list of lists: [[(info_to_store[], rewards, done)]]
            all_env_data = environment_runner.collect_data(self.time_batch_size, self._env_spec,
                                                           self.preprocess, self.action_size)

            policy.train(all_env_data)

            # Compute the number of timesteps just by taking the total amount of data we just collected
            timesteps = np.array([len(env_data) for env_data in all_env_data]).sum()
            total_timesteps += timesteps
