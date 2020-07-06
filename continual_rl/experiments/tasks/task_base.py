from abc import ABC, abstractmethod
import types
import gym


class TaskBase(ABC):
    def __init__(self, env_spec, obs_size, action_size, num_timesteps, time_batch_size, eval_mode, output_dir):
        self.obs_size = obs_size
        self.action_size = action_size
        self._num_timesteps = num_timesteps
        self._env_spec = env_spec

    @abstractmethod
    def preprocess(self, observation):
        pass

    def run(self, policy, task_id, summary_writer):
        episode_runner = policy.get_episode_runner()

        # all_env_data is a list of tuples, (timesteps, info_to_store[], rewards)
        # TODO: is it reasonable to assume timesteps is just len(info_to_store)?
        all_env_data = episode_runner.collect_data(self._env_spec, self.preprocess)
