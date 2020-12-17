import os
import yappi
from continual_rl.utils.utils import Utils
from continual_rl.utils.common_exceptions import OutputDirectoryNotSetException


class InvalidTaskAttributeException(Exception):
    def __init__(self, error_msg):
        super().__init__(error_msg)


class Experiment(object):
    def __init__(self, tasks, continual_testing_freq=None, cycle_count=1):
        """
        The Experiment class contains everything that should be held consistent when the experiment is used as a
        setting for a baseline.

        A single experiment can cover tasks with a variety of action spaces. It is up to the policy on how they wish
        to handle this, but what the Experiment does is create a dictionary mapping action_space_id to action space, and
        ensures that all tasks claiming the same id use the same action space.

        The observation space and time batch sizes are both restricted to being the same for all tasks. This
        initialization will assert if this is violated.

        :param tasks: A list of subclasses of TaskBase. These need to have a consistent observation space.
        :param output_dir: The directory in which logs will be stored.
        :param continual_testing_freq: The number of timesteps between evaluation steps on the not-currently-training
        tasks.
        :param cycle count: The number of times to cycle through the list of tasks.
        """
        self.tasks = tasks
        self.action_spaces = self._get_action_spaces(self.tasks)
        self.observation_space = self._get_common_attribute([task.observation_space for task in self.tasks])
        self.time_batch_size = self._get_common_attribute([task.time_batch_size for task in self.tasks])
        self._output_dir = None
        self._continual_testing_freq = continual_testing_freq
        self._cycle_count = cycle_count

    def set_output_dir(self, output_dir):
        self._output_dir = output_dir

    @property
    def output_dir(self):
        if self._output_dir is None:
            raise OutputDirectoryNotSetException("Output directory not set, but is attempting to be used. "
                                                 "Call set_output_dir.")
        return self._output_dir

    @property
    def _logger(self):
        return Utils.create_logger(f"{self.output_dir}/core_process.log")

    @classmethod
    def _get_action_spaces(self, tasks):
        action_space_map = {}  # Maps task id to its action space

        for task in tasks:
            if task.action_space_id not in action_space_map:
                action_space_map[task.action_space_id] = task.action_space
            elif action_space_map[task.action_space_id] != task.action_space:
                raise InvalidTaskAttributeException(f"Action sizes were mismatched for task {task.action_space_id}")

        return action_space_map

    @classmethod
    def _get_common_attribute(self, task_attributes):
        common_attribute = None

        for task_attribute in task_attributes:
            if common_attribute is None:
                common_attribute = task_attribute

            if task_attribute != common_attribute:
                raise InvalidTaskAttributeException("Tasks do not have a common attribute.")

        return common_attribute

    def _run_continual_eval(self, task_run_id, policy, summary_writer, total_timesteps):
        # Run a small amount of eval on all non-eval, not-currently-running tasks
        for test_task_run_id, test_task in enumerate(self.tasks):
            if test_task_run_id != task_run_id and not test_task._task_spec.eval_mode:
                self._logger.info(f"Continual eval for task: {test_task_run_id}")

                # Don't increment the total_timesteps counter for continual tests
                test_task_runner = self.tasks[test_task_run_id].continual_eval(test_task_run_id, policy, summary_writer,
                                                                    output_dir=self.output_dir,
                                                                    timestep_log_offset=total_timesteps)
                test_complete = False
                while not test_complete:
                    try:
                        next(test_task_runner)
                    except StopIteration:
                        test_complete = True

                self._logger.info(f"Completed continual eval for task: {test_task_run_id}")

    def _run(self, policy, summary_writer):
        # Only updated after a task is complete. To get the current within-task number, add task_timesteps
        total_train_timesteps = 0
        yappi.start()

        for task_run_id, task in enumerate(self.tasks):
            # Run the current task as a generator so we can intersperse testing tasks during the run
            self._logger.info(f"Starting task {task_run_id}")
            task_complete = False
            task_runner = task.run(task_run_id, policy, summary_writer, self.output_dir,
                                   timestep_log_offset=total_train_timesteps)
            task_timesteps = 0  # What timestep the task is currently on. Cumulative during a task.
            last_continual_testing_step = -1e6  # Make it very negative so the first update gets a CL run
            continual_freq = self._continual_testing_freq

            while not task_complete:
                try:
                    task_timesteps = next(task_runner)
                except StopIteration:
                    task_complete = True

                # If we're already doing eval, don't do a forced eval run (nothing has trained to warrant it anyway)
                # Evaluate intermittently. Every time is too slow
                if continual_freq is not None and not task._task_spec.eval_mode and \
                        total_train_timesteps + task_timesteps > last_continual_testing_step + continual_freq:
                    self._run_continual_eval(task_run_id, policy, summary_writer,
                                             total_train_timesteps + task_timesteps)
                    last_continual_testing_step = total_train_timesteps + task_timesteps

            # Log out some info about the just-completed task
            self._logger.info(f"Task {task_run_id} complete")
            profiling_path = os.path.join(self.output_dir, "profile.log")
            with open(profiling_path, "a") as profile_file:
                yappi.get_func_stats().print_all(out=profile_file)
            yappi.clear_stats()  # Prep for the next task, which we'll profile separately

            # Only increment the global counter for training (it's supposed to represent number of frames *trained on*)
            if not task._task_spec.eval_mode:
                total_train_timesteps += task_timesteps

    def try_run(self, policy, summary_writer):
        try:
            self._run(policy, summary_writer)
        except Exception as e:
            self._logger.exception(f"Failed with exception: {e}")
            policy.shutdown()

            raise e
