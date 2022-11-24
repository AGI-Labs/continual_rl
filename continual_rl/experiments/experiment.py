import os
import json
from continual_rl.experiments.run_metadata import RunMetadata
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
        self.observation_space = self._get_common_attribute(
            [task.observation_space for task in self.tasks]
        )
        self.task_ids = [task.task_id for task in tasks]
        self._output_dir = None
        self._continual_testing_freq = continual_testing_freq
        self._cycle_count = cycle_count

    def set_output_dir(self, output_dir):
        self._output_dir = output_dir

    @property
    def output_dir(self):
        if self._output_dir is None:
            raise OutputDirectoryNotSetException("Output directory not set, but is attempting to be used. Call set_output_dir.")
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
            # not checking test_task._task_spec.eval_mode anymore since some eval tasks
            # (for train/test pairs) should be continual eval
            if not test_task._task_spec.with_continual_eval:
                continue

            self._logger.info(f"Continual eval for task: {test_task_run_id}")

            # Don't increment the total_timesteps counter for continual tests
            test_task_runner = self.tasks[test_task_run_id].continual_eval(
                test_task_run_id,
                policy,
                summary_writer,
                output_dir=self.output_dir,
                timestep_log_offset=total_timesteps,
            )
            test_complete = False
            while not test_complete:
                try:
                    next(test_task_runner)
                except StopIteration:
                    test_complete = True

            self._logger.info(f"Completed continual eval for task: {test_task_run_id}")

    def _run(self, policy, summary_writer):
        # Load as necessary
        policy.load(self.output_dir)
        run_metadata = RunMetadata(self._output_dir)
        start_cycle_id = run_metadata.cycle_id
        start_task_id = run_metadata.task_id
        start_task_timesteps = run_metadata.task_timesteps

        # Only updated after a task is complete. To get the current within-task number, add task_timesteps
        total_train_timesteps = run_metadata.total_train_timesteps

        timesteps_per_save = policy.config.timesteps_per_save

        for cycle_id in range(start_cycle_id, self._cycle_count):
            for task_run_id, task in enumerate(self.tasks[start_task_id:], start=start_task_id):
                # Run the current task as a generator so we can intersperse testing tasks during the run
                self._logger.info(f"Starting cycle {cycle_id} task {task_run_id}")
                task_complete = False
                task_runner = task.run(
                    task_run_id,
                    policy,
                    summary_writer,
                    self.output_dir,
                    timestep_log_offset=total_train_timesteps,
                    task_timestep_start=start_task_timesteps,
                )
                task_timesteps = start_task_timesteps  # What timestep the task is currently on. Cumulative during a task.
                continual_freq = self._continual_testing_freq
                last_timestep_saved = None  # Ensures a save at the beginning of every new task (after one train step)

                # The last step at which continual testing was done. Initializing to be more negative
                # than the frequency we collect at, to ensure we do a collection right away
                last_continual_testing_step = -10 * continual_freq if continual_freq is not None else None

                while not task_complete:
                    try:
                        task_timesteps, _ = next(task_runner)
                    except StopIteration:
                        task_complete = True

                    if not task._task_spec.eval_mode:
                        if last_timestep_saved is None or task_timesteps - last_timestep_saved >= timesteps_per_save or \
                                task_complete:
                            # Save the metadata that allows us to resume where we left off.
                            # This will not copy files in large_file_path such as 
                            # replay buffers, and is intended for debugging model changes
                            # at task boundaries.
                            run_metadata.save(cycle_id, task_run_id, task_timesteps, total_train_timesteps)
                            policy.save(self.output_dir, cycle_id, task_run_id, task_timesteps)
                            if task_complete:
                                task_boundary_dir = os.path.join(self.output_dir, f'cycle{cycle_id}_task{task_run_id}')
                                os.makedirs(task_boundary_dir, exist_ok=True)

                                policy.save(task_boundary_dir, cycle_id, task_run_id, task_timesteps)

                            last_timestep_saved = task_timesteps

                    # If we're already doing eval, don't do a forced eval run (nothing has trained to warrant it anyway)
                    # Evaluate intermittently. Every time is too slow
                    if continual_freq is not None and not task._task_spec.eval_mode and \
                            total_train_timesteps + task_timesteps > last_continual_testing_step + continual_freq:
                        self._run_continual_eval(
                            task_run_id,
                            policy,
                            summary_writer,
                            total_train_timesteps + task_timesteps,
                        )
                        last_continual_testing_step = total_train_timesteps + task_timesteps

                # Log out some info about the just-completed task
                self._logger.info(f"Task {task_run_id} complete")

                # Only increment the global counter for training (it's supposed to represent number of frames *trained on*)
                if not task._task_spec.eval_mode:
                    total_train_timesteps += task_timesteps

                # On the next task, start from the beginning (regardless of where we loaded from)
                start_task_timesteps = 0

            # On the next cycle, start from the beginning again (regardless of where we loaded from)
            start_task_id = 0

    def try_run(self, policy, summary_writer):
        try:
            self._run(policy, summary_writer)
        except Exception as e:
            self._logger.exception(f"Failed with exception: {e}")
            policy.shutdown()

            raise e
