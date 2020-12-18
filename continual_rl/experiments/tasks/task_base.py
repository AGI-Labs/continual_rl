from abc import ABC
import numpy as np
import gym
from continual_rl.experiments.tasks.task_spec import TaskSpec
from continual_rl.utils.utils import Utils


class TaskBase(ABC):
    def __init__(self, action_space_id, preprocessor, env_spec, observation_space, action_space,
                 num_timesteps, eval_mode):
        """
        Subclasses of TaskBase contain all information that should be consistent within a task for everyone
        trying to use it for a baseline. In other words anything that should be kept comparable, should be specified
        here.
        :param action_space_id: An identifier that is consistent between all times we run any tasks that share an
        action space. This is basically how we identify that two tasks are intended to be the same.
        :param preprocessor: A subclass of PreprocessBase that handles the input type of this task.
        :param env_spec: A gym environment name OR a lambda that creates an environment.
        :param observation_space: The observation space that will be passed to the policy,
        not including batch, if applicable, or time_batch_size.
        :param action_space: The action_space the environment of this task uses.
        :param num_timesteps: The total number of timesteps this task should run
        :param eval_mode: Whether this environment is being run in eval_mode (i.e. training should not occur)
        should end.
        """
        self.action_space_id = action_space_id
        self.action_space = action_space
        self.observation_space = observation_space

        # We keep running mean of rewards so the average is less dependent on how many episodes completed
        # in the last update
        self._rolling_reward_count = 100  # The number OpenAI baselines uses. Represents # rewards to keep between logs

        # How many episodes to run while doing continual evaluation. It will be at least this number, but might be more
        # (e.g. 8 returns in the first collection, then 6 in the next), as it is used as the max within a collection
        # (i.e. by environment_runner_batch) as well.
        continual_eval_num_returns = 10

        # The set of task parameters that the environment runner gets access to.
        self._task_spec = TaskSpec(action_space_id, preprocessor, env_spec, num_timesteps, eval_mode)

        # A version of the task spec to use if we're in forced-eval mode. The collection will end when
        # the first reward is logged, so the num_timesteps just needs to be long enough to allow for that.
        self._continual_eval_task_spec = TaskSpec(action_space_id, preprocessor, env_spec,
                                                  num_timesteps=100000, eval_mode=True,
                                                  return_after_episode_num=continual_eval_num_returns)

    def _report_log(self, summary_writer, log, run_id, default_timestep):
        type = log["type"]
        tag = f"{log['tag']}/{run_id}"
        value = log["value"]
        timestep = log.get("timestep", None) or default_timestep

        if type == "video":
            summary_writer.add_video(tag, value, global_step=timestep)
        elif type == "scalar":
            summary_writer.add_scalar(tag, value, global_step=timestep)
        elif type == "image":
            summary_writer.add_image(tag, value, global_step=timestep)

        summary_writer.flush()

    def logger(self, output_dir):
        logger = Utils.create_logger(f"{output_dir}/core_process.log")
        return logger

    def run(self, run_id, policy, summary_writer, output_dir, timestep_log_offset=0):
        """
        Run the task as a "primary" task.
        """
        return self._run(self._task_spec, run_id, policy, summary_writer, output_dir,
                         timestep_log_offset, wait_to_report=False)

    def continual_eval(self, run_id, policy, summary_writer, output_dir, timestep_log_offset=0):
        """
        Run the task as a "continual eval" task. In other words brief samples during the running of another task.
        """
        return self._run(self._continual_eval_task_spec, run_id, policy, summary_writer, output_dir,
                         timestep_log_offset, wait_to_report=True)

    def _complete_logs(self, run_id, collected_returns, output_dir, timestep, logs_to_report, summary_writer):
        if len(collected_returns) > 0:
            # Note that we're logging at the offset - any steps taken during collection don't matter
            mean_rewards = np.array(collected_returns).mean()
            self.logger(output_dir).info(f"{timestep}: {mean_rewards}")
            logs_to_report.append({"type": "scalar", "tag": f"reward", "value": mean_rewards,
                                   "timestep": timestep})

        for log in logs_to_report:
            if summary_writer is not None:
                self._report_log(summary_writer, log, run_id, default_timestep=timestep)
            else:
                self.logger(output_dir).info(log)

    def _run(self, task_spec, run_id, policy, summary_writer, output_dir, timestep_log_offset, wait_to_report):
        """
        Run a task according to its task spec.
        :param task_spec: Specifies how the task should be handled as it runs. E.g. the number of timesteps, or
        what preprocessor to use.
        :param run_id: The identifier used to group results. All calls to run with the same run_id will be plotted as
        one task.
        :param policy: The policy used to run the task.
        :param summary_writer: Used to log tensorboard files.
        :param output_dir: The location to write logs to.
        :param timestep_log_offset: How many (global) timesteps have been run prior to the execution of this task, for
        the purposes of alignment.
        :param wait_to_report: If true, the result will be logged after all results are in, otherwise it will be
        logged whenever any result comes in.
        :yields: The number of timesteps executed so far in this task.
        """
        task_timesteps = 0
        environment_runner = policy.get_environment_runner()  # Getting a new one will cause the envs to be re-created
        collected_returns = []
        collected_logs_to_report = []

        while task_timesteps < task_spec.num_timesteps:
            # all_env_data is a list of timestep_datas
            timesteps, all_env_data, rewards_to_report, logs_to_report = environment_runner.collect_data(task_spec)

            if not task_spec.eval_mode:
                train_logs = policy.train(all_env_data)

                if train_logs is not None:
                    logs_to_report.extend(train_logs)

            task_timesteps += timesteps
            total_log_timesteps = timestep_log_offset + task_timesteps

            # Aggregate our results
            collected_returns.extend(rewards_to_report)
            collected_logs_to_report.extend(logs_to_report)

            # If we're logging continuously, do so and clear the log list, but only if we actually have something new
            if len(rewards_to_report) > 0 and not wait_to_report:
                self._complete_logs(run_id, collected_returns, output_dir, total_log_timesteps,
                                    collected_logs_to_report, summary_writer)

                # We only truncate/clear our aggregators if we've logged the information they contain
                collected_logs_to_report.clear()
                collected_returns = collected_returns[:self._rolling_reward_count]

            yield task_timesteps

            if (task_spec.return_after_episode_num is not None and
                    len(collected_returns) >= task_spec.return_after_episode_num):
                self.logger(output_dir).info(f"Ending task early at task step {task_timesteps}")
                break

        # If we waited, report everything now. The main reason for this is to log the average over all timesteps
        # in the run, instead of doing the rolling mean
        if wait_to_report:
            total_log_timesteps = timestep_log_offset + task_timesteps
            self._complete_logs(run_id, collected_returns, output_dir, total_log_timesteps, collected_logs_to_report,
                                summary_writer)

        environment_runner.cleanup()
