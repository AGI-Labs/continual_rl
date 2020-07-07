from abc import ABC, abstractmethod
import numpy as np


class TaskBase(ABC):
    def __init__(self, env_spec, observation_size, action_size, time_batch_size, num_timesteps, eval_mode, output_dir):
        """
        Subclasses of TaskBase contain all information that should be consistent within a task for everyone
        trying to use it for a baseline. In other words anything that should be kept comparable, should be specified
        here.
        :param env_spec: A gym environment name OR a lambda that creates an environment.
        :param observation_size: The observation size that will be passed to the policy,
        not including batch, if applicable, or time_batch_size.
        :param action_size: The action_size the environment of this task uses.
        :param time_batch_size: The number of steps in time that will be concatenated together
        :param num_timesteps: The total number of timesteps this task should run
        :param eval_mode: Whether this environment is being run in eval_mode (i.e. training should not occur)
        :param output_dir: The output location for any logs or artefacts
        """
        self.observation_size = [time_batch_size, *observation_size]
        self.action_size = action_size
        self.time_batch_size = time_batch_size
        self._num_timesteps = num_timesteps
        self._env_spec = env_spec
        self._eval_mode = eval_mode

    @abstractmethod
    def preprocess(self, observation):
        pass

    def run(self, policy, task_id, summary_writer):
        total_timesteps = 0
        environment_runner = policy.get_environment_runner()

        while total_timesteps < self._num_timesteps:
            # all_env_data is a list info_to_stores
            timesteps, all_env_data, rewards_to_report = environment_runner.collect_data(self.time_batch_size,
                                                                                         self._env_spec,
                                                                                         self.preprocess,
                                                                                         self.action_size)

            if not self._eval_mode:
                policy.train(all_env_data)

            total_timesteps += timesteps

            if len(rewards_to_report) > 0:
                print(f"{total_timesteps}: {np.array(rewards_to_report).mean()}")
