import numpy as np
from continual_rl.utils.utils import Utils


class Experiment(object):
    def __init__(self, tasks, output_dir):
        self.tasks = tasks
        self.action_size = self._get_common_action_size(self.tasks)
        self.obs_size = self._get_common_obs_size(self.tasks)

        self._output_dir = output_dir

    @property
    def _logger(self):
        return Utils.create_logger(f"{self._output_dir}/core_process.log", name="core_logger")

    def _get_common_obs_size(self, tasks):
        common_obs_size = None

        for task in tasks:
            if common_obs_size is None:
                common_obs_size = task.obs_size
            else:
                assert common_obs_size == task.obs_size, "Tasks must share a common observation size."

        return common_obs_size

    def _get_common_action_size(self, tasks):
        action_sizes = [task.action_size for task in tasks]
        return np.array(action_sizes).max()

    def _run(self, policy, summary_writer):
        for task_id, task in enumerate(self.tasks):
            self._logger.info(f"Starting task {task_id}")
            task.run(policy, task_id, summary_writer)
            self._logger.info(f"Task {task_id} complete")

    def try_run(self, policy, summary_writer):
        try:
            self._run(policy, summary_writer)
        except Exception as e:
            self._logger.exception(f"Failed with exception: {e}")
            policy.shutdown()

            for task in self.tasks:
                task.cleanup_processes()

            raise e
