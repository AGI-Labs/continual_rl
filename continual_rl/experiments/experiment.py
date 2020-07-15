from continual_rl.utils.utils import Utils
from continual_rl.utils.common_exceptions import OutputDirectoryNotSetException


class InvalidTaskAttributeException(Exception):
    def __init__(self, error_msg):
        super().__init__(error_msg)


class Experiment(object):
    def __init__(self, tasks):
        """
        The Experiment class contains everything that should be held consistent when the experiment is used as a
        setting for a baseline.

        A single experiment can cover tasks with a variety of action spaces. It is up to the policy on how they wish
        to handle this, but what the Experiment does is create a dictionary mapping task id to action space, and
        ensures that all tasks claiming the same id use the same action space.

        The observation size and time batch sizes are both restricted to being the same for all tasks. This
        initialization will assert if this is violated.

        :param tasks: A list of subclasses of TaskBase. These need to have a consistent observation size.
        :param output_dir: The directory in which logs will be stored.
        """
        self.tasks = tasks
        self.action_sizes = self._get_action_sizes(self.tasks)
        self.observation_size = self._get_common_attribute([task.observation_size for task in self.tasks])
        self.time_batch_size = self._get_common_attribute([task.time_batch_size for task in self.tasks])

        self._output_dir = None

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
        return Utils.create_logger(f"{self.output_dir}/core_process.log", name="core_logger")

    @classmethod
    def _get_action_sizes(self, tasks):
        action_size_map = {}  # Maps task id to its action space

        for task in tasks:
            if task.task_id not in action_size_map:
                action_size_map[task.task_id] = task.action_size
            elif action_size_map[task.task_id] != task.action_size:
                raise InvalidTaskAttributeException(f"Action sizes were mismatched for task {task.task_id}")

        return action_size_map

    @classmethod
    def _get_common_attribute(self, task_attributes):
        common_attribute = None

        for task_attribute in task_attributes:
            if common_attribute is None:
                common_attribute = task_attribute

            if task_attribute != common_attribute:
                raise InvalidTaskAttributeException("Tasks do not have a common attribute.")

        return common_attribute

    def _run(self, policy, summary_writer):
        for task_run_id, task in enumerate(self.tasks):
            self._logger.info(f"Starting task {task_run_id}")
            task.run(policy, summary_writer)
            self._logger.info(f"Task {task_run_id} complete")

    def try_run(self, policy, summary_writer):
        try:
            self._run(policy, summary_writer)
        except Exception as e:
            self._logger.exception(f"Failed with exception: {e}")
            policy.shutdown()

            raise e
