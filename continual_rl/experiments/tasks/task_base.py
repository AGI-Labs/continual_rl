from abc import ABC
import numpy as np
from continual_rl.experiments.tasks.task_spec import TaskSpec
from continual_rl.utils.utils import Utils


class TaskBase(ABC):
    ALL_TASK_IDS = set()

    def __init__(self, task_id, action_space_id, preprocessor, env_spec, observation_space, action_space,
                 num_timesteps, eval_mode, continual_eval=True, rolling_return_count=100,
                 continual_eval_num_returns=10):
        """
        Subclasses of TaskBase contain all information that should be consistent within a task for everyone
        trying to use it for a baseline. In other words anything that should be kept comparable, should be specified
        here.
        :param task_id: Each task_id must be unique, which is verified upon task initialization. The task id can be
        used by the policies for saving/loading, so it needs to be both unique and consistent (e.g. as the task set changes)
        :param action_space_id: An identifier that is consistent between all times we run any tasks that share an
        action space.
        :param preprocessor: A subclass of PreprocessBase that handles the input type of this task.
        :param env_spec: A gym environment name OR a lambda that creates an environment.
        :param observation_space: The observation space that will be passed to the policy,
        not including batch, if applicable, or time_batch_size.
        :param action_space: The action_space the environment of this task uses.
        :param num_timesteps: The total number of timesteps this task should run
        :param eval_mode: Whether this environment is being run in eval_mode (i.e. training should not occur)
        should end.
        :param continual_eval: Whether the task should be run during continual evaluation collections
        :param rolling_return_count: How many returns in the rolling mean (Default is the number OpenAI baselines uses.)
        :param continual_eval_num_returns: How many episodes to run while doing continual evaluation.
        These should be collected by a single environment: see note in policy_base.get_environment_runner
        """
        self.action_space_id = action_space_id
        self.action_space = action_space
        self.observation_space = observation_space
        self.task_id = task_id

        self._verify_and_save_task_id(task_id)

        # We keep running mean of rewards so the average is less dependent on how many episodes completed
        # in the last update
        self._rolling_return_count = rolling_return_count

        # The set of task parameters that the environment runner gets access to.
        self._task_spec = TaskSpec(self.task_id, action_space_id, preprocessor, env_spec, num_timesteps, eval_mode,
                                   with_continual_eval=continual_eval)

        # A version of the task spec to use if we're in forced-eval mode. The collection will end when
        # the first reward is logged, so the num_timesteps just needs to be long enough to allow for that.
        self._continual_eval_task_spec = TaskSpec(self.task_id, action_space_id, preprocessor, env_spec,
                                                  num_timesteps=100000, eval_mode=True,
                                                  return_after_episode_num=continual_eval_num_returns)

    @classmethod
    def _verify_and_save_task_id(cls, task_id):
        assert task_id not in cls.ALL_TASK_IDS, f"Task with task id {task_id} failed to be created due to task id already in use. Use a different id."
        cls.ALL_TASK_IDS.add(task_id)

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

    def run(self, run_id, policy, summary_writer, output_dir, task_timestep_start=0, timestep_log_offset=0):
        """
        Run the task as a "primary" task.
        """
        return self._run(self._task_spec, run_id, policy, summary_writer, output_dir,
                         timestep_log_offset, wait_to_report=False, log_with_task_timestep=True,
                         reward_tag="train_reward", task_timestep_start=task_timestep_start)

    def continual_eval(self, run_id, policy, summary_writer, output_dir, timestep_log_offset=0):
        """
        Run the task as a "continual eval" task. In other words brief samples during the running of another task.
        """
        return self._run(self._continual_eval_task_spec, run_id, policy, summary_writer, output_dir,
                         timestep_log_offset, wait_to_report=True, log_with_task_timestep=False,
                         reward_tag="eval_reward")

    def _complete_logs(self, run_id, collected_returns, output_dir, timestep, logs_to_report, summary_writer,
                       reward_tag):
        if len(collected_returns) > 0:
            # Note that we're logging at the offset - any steps taken during collection don't matter
            mean_rewards = np.array(collected_returns).mean()
            self.logger(output_dir).info(f"{timestep}: {mean_rewards}")
            logs_to_report.append({"type": "scalar", "tag": reward_tag, "value": mean_rewards,
                                   "timestep": timestep})

        for log in logs_to_report:
            if summary_writer is not None:
                self._report_log(summary_writer, log, run_id, default_timestep=timestep)
            else:
                self.logger(output_dir).info(log)

    def _compute_timestep_to_log(self, offset, task_timestep, log_with_task_timestep):
        total_timesteps = offset
        if log_with_task_timestep:
            total_timesteps += task_timestep
        return total_timesteps

    def _run(self, task_spec, run_id, policy, summary_writer, output_dir, timestep_log_offset, wait_to_report,
             log_with_task_timestep, reward_tag, task_timestep_start=0):
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
        :param log_with_task_timestep: Whether or not the timestep of logging should include the current task's
        timestep.
        :param reward_tag: What tag rewards will be logged under in the tensorboard
        :param task_timestep_start: The timestep to start collection at (for loading from existing)
        :yields: (task_timesteps, reported_data): The number of timesteps executed so far in this task,
        and a tuple of what was collected (rewards, logs) since the last returned data
        """
        task_timesteps = task_timestep_start
        environment_runner = policy.get_environment_runner(task_spec)  # Getting a new one will cause the envs to be re-created
        collected_returns = []
        collected_logs_to_report = []

        while task_timesteps < task_spec.num_timesteps:
            # all_env_data is a list of timestep_datas
            timesteps, all_env_data, returns_to_report, logs_to_report = environment_runner.collect_data(task_spec)

            if not task_spec.eval_mode:
                train_logs = policy.train(all_env_data)

                if train_logs is not None:
                    logs_to_report.extend(train_logs)

            task_timesteps += timesteps

            # Aggregate our results
            collected_returns.extend(returns_to_report)
            collected_logs_to_report.extend(logs_to_report)
            data_to_return = None

            # If we're logging continuously, do so and clear the log list, but only if we actually have something new
            if len(returns_to_report) > 0 and not wait_to_report:
                total_log_timesteps = self._compute_timestep_to_log(timestep_log_offset, task_timesteps,
                                                                    log_with_task_timestep)
                self._complete_logs(run_id, collected_returns, output_dir, total_log_timesteps,
                                    collected_logs_to_report, summary_writer, reward_tag)
                # Only return the new data, not the full rolling aggregation, since we're not waiting in this case
                data_to_return = (returns_to_report, logs_to_report)

                # We only truncate/clear our aggregators if we've logged the information they contain
                collected_logs_to_report.clear()
                collected_returns = collected_returns[-self._rolling_return_count:]

            yield task_timesteps, data_to_return

            if (task_spec.return_after_episode_num is not None and
                    len(collected_returns) >= task_spec.return_after_episode_num):
                # The collection time frame may have over-shot. Just take the first n.
                collected_returns = collected_returns[:task_spec.return_after_episode_num]
                self.logger(output_dir).info(f"Ending task {task_spec.task_id}, eval_mode {task_spec.eval_mode}, early at task step {task_timesteps}")
                break

        # If we waited, report everything now. The main reason for this is to log the average over all timesteps
        # in the run, instead of doing the rolling mean
        data_to_return = None
        if wait_to_report:
            total_log_timesteps = self._compute_timestep_to_log(timestep_log_offset, task_timesteps,
                                                                log_with_task_timestep)
            self._complete_logs(run_id, collected_returns, output_dir, total_log_timesteps, collected_logs_to_report,
                                summary_writer, reward_tag)

            # Return everything, since we waited
            data_to_return = (collected_returns, collected_logs_to_report)

        environment_runner.cleanup(task_spec)
        yield task_timesteps, data_to_return
