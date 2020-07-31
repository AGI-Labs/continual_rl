from abc import ABC, abstractmethod
import numpy as np


class TaskBase(ABC):
    def __init__(self, action_space_id, env_spec, observation_size, action_space, time_batch_size, num_timesteps,
                 eval_mode, early_stopping_condition=None):
        """
        Subclasses of TaskBase contain all information that should be consistent within a task for everyone
        trying to use it for a baseline. In other words anything that should be kept comparable, should be specified
        here.
        :param action_space_id: An identifier that is consistent between all times we run any tasks that share an
        action space. This is basically how we identify that two tasks are intended to be the same.
        :param env_spec: A gym environment name OR a lambda that creates an environment.
        :param observation_size: The observation size that will be passed to the policy,
        not including batch, if applicable, or time_batch_size.
        :param action_space: The action_space the environment of this task uses.
        :param time_batch_size: The number of steps in time that will be concatenated together
        :param num_timesteps: The total number of timesteps this task should run
        :param eval_mode: Whether this environment is being run in eval_mode (i.e. training should not occur)
        :param early_stopping_condition: Lambda that takes (timestep, episode_info) and returns True if the episode
        should end.
        """
        self.action_space_id = action_space_id
        self.observation_size = [time_batch_size, *observation_size]
        self.action_space = action_space
        self.time_batch_size = time_batch_size
        self._num_timesteps = num_timesteps
        self._env_spec = env_spec
        self._eval_mode = eval_mode
        self._early_stopping_condition = early_stopping_condition

    @abstractmethod
    def preprocess(self, observation):
        pass

    @abstractmethod
    def render_episode(self, episode_observations):
        """
        Turn a list of observations gathered from the episode into a video tensor (N, T, C, H, W) that can be saved off
        to view behavior.
        """
        pass

    def _report_log(self, summary_writer, log, default_timestep):
        type = log["type"]
        tag = log["tag"]
        value = log["value"]
        timestep = log["timestep"] or default_timestep

        if type == "video":
            summary_writer.add_video(tag, value, global_step=timestep)
        elif type == "scalar":
            summary_writer.add_scalar(tag, value, global_step=timestep)

        summary_writer.flush()

    def run(self, run_id, policy, summary_writer):
        total_timesteps = 0
        environment_runner = policy.get_environment_runner()

        while total_timesteps < self._num_timesteps:
            # all_env_data is a list of info_to_stores
            timesteps, all_env_data, rewards_to_report, logs_to_report = environment_runner.collect_data(
                self.time_batch_size,
                self._env_spec,
                self.preprocess,
                self.action_space_id,
                self.render_episode,
                self._early_stopping_condition)

            if not self._eval_mode:
                policy.train(all_env_data)

            total_timesteps += timesteps

            if len(rewards_to_report) > 0:
                mean_rewards = np.array(rewards_to_report).mean()
                print(f"{total_timesteps}: {mean_rewards}")
                logs_to_report.append({"type": "scalar", "tag": f"reward/{run_id}", "value": mean_rewards,
                                       "timestep": total_timesteps})

            for log in logs_to_report:
                self._report_log(summary_writer, log, default_timestep=total_timesteps)
