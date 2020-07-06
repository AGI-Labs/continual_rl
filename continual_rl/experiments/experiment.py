import numpy as np
from continual_rl.utils.utils import Utils


class Experiment(object):
    def __init__(self, tasks, output_dir):
        self.tasks = tasks
        self.action_size = self._get_common_action_size(self.tasks)
        self.observation_size = self._get_common_observation_size(self.tasks)
        self.time_batch_size = self._get_common_time_batch_size(self.tasks)

        self._output_dir = output_dir

    @property
    def _logger(self):
        return Utils.create_logger(f"{self._output_dir}/core_process.log", name="core_logger")

    def _get_common_observation_size(self, tasks):
        common_obs_size = None

        for task in tasks:
            if common_obs_size is None:
                common_obs_size = task.observation_size
            else:
                assert common_obs_size == task.observation_size, "Tasks must share a common observation size."

        return common_obs_size

    def _get_common_action_size(self, tasks):
        action_sizes = [task.action_size for task in tasks]
        return np.array(action_sizes).max()

    def _get_common_time_batch_size(self, tasks):
        assert len(tasks) > 0, "At least one task must be specified."
        time_batch_size = tasks[0].time_batch_size

        for task in tasks:
            assert time_batch_size == task.time_batch_size, "All tasks must use the same time batch size " \
                                                            "(Number of timesteps worth of observations that get " \
                                                            "concatenated and passed to the policy)."

        return time_batch_size

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

            raise e
