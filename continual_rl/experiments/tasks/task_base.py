from abc import ABC
import numpy as np
import gym
from continual_rl.experiments.tasks.task_spec import TaskSpec
from continual_rl.utils.utils import Utils


class TaskBase(ABC):
    def __init__(self, action_space_id, preprocessor, env_spec, observation_space, action_space, time_batch_size,
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
        :param time_batch_size: The number of steps in time that will be concatenated together
        :param num_timesteps: The total number of timesteps this task should run
        :param eval_mode: Whether this environment is being run in eval_mode (i.e. training should not occur)
        should end.
        """
        self.action_space_id = action_space_id
        self.action_space = action_space
        self.time_batch_size = time_batch_size

        # We stack frames in the first dimension, so update the observation to include this.
        old_space = observation_space
        self.observation_space = gym.spaces.Box(
            low=old_space.low.min(),  # .min to turn the array back to a scalar
            high=old_space.high.max(),  # .max to turn the array back to a scalar
            shape=[time_batch_size, *old_space.shape],
            dtype=old_space.dtype,
        )

        # A running mean of rewards so the average is less dependent on how many episodes completed in the last update
        self._rewards_to_report = []
        self._rolling_reward_count = 100  # The number OpenAI baselines uses. Represents # rewards to keep between logs

        # How many episodes to run while doing continual evaluation. It will be at least this number, but might be more
        # (e.g. 8 returns in the first collection, then 6 in the next), as it is used as the max within a collection
        # (i.e. by environment_runner_batch) as well.
        continual_eval_num_returns = 10

        # The set of task parameters that the environment runner gets access to.
        self._task_spec = TaskSpec(action_space_id, preprocessor, env_spec, time_batch_size, num_timesteps, eval_mode)

        # A version of the task spec to use if we're in forced-eval mode. The collection will end when
        # the first reward is logged, so the num_timesteps just needs to be long enough to allow for that.
        self._force_eval_task_spec = TaskSpec(action_space_id, preprocessor, env_spec, time_batch_size,
                                              num_timesteps=100000, eval_mode=True,
                                              return_after_reward_num=continual_eval_num_returns)

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

    def run(self, run_id, policy, summary_writer, output_dir, force_eval=False, timestep_log_offset=0):
        task_timesteps = 0
        environment_runner = policy.get_environment_runner()  # Getting a new one will cause the envs to be re-created

        task_spec = self._force_eval_task_spec if force_eval else self._task_spec
        force_eval_rewards = []
        total_timesteps = task_spec.num_timesteps  # Don't change this even if we switch to eval part way through

        # We collect rewards for force_eval and put them in force_eval_rewards, and return after we have
        # return_after_reward_num of them. This means force_eval is NOT exactly eval_mode
        assert (force_eval and task_spec.return_after_reward_num is not None) or \
               (not force_eval and task_spec.return_after_reward_num is None), \
            "return_after_reward_num must only be specified in force_eval mode"

        while task_timesteps < total_timesteps:
            # all_env_data is a list of timestep_datas
            timesteps, all_env_data, rewards_to_report, logs_to_report = environment_runner.collect_data(task_spec)

            if not task_spec.eval_mode:
                train_logs = policy.train(all_env_data)

                if train_logs is not None:
                    logs_to_report.extend(train_logs)

            task_timesteps += timesteps
            total_log_timesteps = timestep_log_offset + task_timesteps

            # Compute reward results so far
            if not force_eval:
                # Only include the new rewards into the rolling total if we're not in "force eval" mode
                self._rewards_to_report.extend(rewards_to_report)

                if len(self._rewards_to_report) > 0:  # TODO: and new rewards > 0 (same below).
                    mean_rewards = np.array(self._rewards_to_report).mean()
                    self.logger(output_dir).info(f"{total_log_timesteps}: {mean_rewards}")
                    logs_to_report.append({"type": "scalar", "tag": f"reward", "value": mean_rewards,
                                           "timestep": total_log_timesteps})

                self._rewards_to_report = self._rewards_to_report[-self._rolling_reward_count:]
            else:
                force_eval_rewards.extend(rewards_to_report)

                if len(force_eval_rewards) >= task_spec.return_after_reward_num:
                    # Note that we're logging at the offset - any steps taken during collection don't matter
                    mean_rewards = np.array(force_eval_rewards).mean()
                    self.logger(output_dir).info(f"EVAL {total_log_timesteps}: {mean_rewards}")
                    logs_to_report.append({"type": "scalar", "tag": f"reward", "value": mean_rewards,
                                           "timestep": timestep_log_offset})

            for log in logs_to_report:
                if summary_writer is not None:
                    self._report_log(summary_writer, log, run_id, default_timestep=total_log_timesteps)
                else:
                    self.logger(output_dir).info(log)

            yield task_timesteps

            if (task_spec.return_after_reward_num is not None and
                    len(force_eval_rewards) >= task_spec.return_after_reward_num):
                self.logger(output_dir).info(f"Ending task early at task step {task_timesteps}")
                break

        environment_runner.cleanup()
