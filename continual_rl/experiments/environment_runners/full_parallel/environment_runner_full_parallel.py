import torch.multiprocessing as multiprocessing
import cloudpickle
from continual_rl.experiments.environment_runners.environment_runner_base import EnvironmentRunnerBase
from continual_rl.experiments.environment_runners.full_parallel.collection_process import CollectionProcess


class EnvironmentRunnerFullParallel(EnvironmentRunnerBase):
    """
    Runs the entirety of collection on separate processes. Uses pytorch multiprocessing so share_memory() can be used.
    """
    def __init__(self, policy, num_parallel_processes, timesteps_per_collection, render_collection_freq=None,
                 create_update_process_bundle=None, receive_update_process_bundle=None, output_dir=None):
        """
        create_update_process_bundle is a callback that creates a list of arbitrary data bundles, one per process, to
        be used to update the states of the processes. This callback is executed on the main process, and the data is
        passed to each appropriate process.
        receive_update_process_bundle is the callback that will be called per-process, to consume the data bundle
        and update accordingly.
        The form of the data bundles is entirely up to the creator of this EnvironmentRunner.
        """
        super().__init__()
        self._create_update_process_bundle = create_update_process_bundle
        self._process_managers = [CollectionProcess(policy, timesteps_per_collection,
                                                    render_collection_freq=render_collection_freq,
                                                    receive_update_process_bundle=receive_update_process_bundle,
                                                    output_dir=output_dir)
                                  for _ in range(num_parallel_processes)]

        for manager in self._process_managers:
            process = multiprocessing.Process(target=manager.try_process_queue)  # TODO: if it takes too long, don't do in constructor...also should CollectionProcess have this?
            process.start()

    def _send_message_to_process(self, process_id, message_id, message_content):
        self._process_managers[process_id].incoming_queue.put((message_id, message_content))

    def _send_message_to_all(self, message_id, message_content):
        for process_id in range(len(self._process_managers)):
            self._send_message_to_process(process_id, message_id, message_content)

    def collect_data(self, task_spec):
        """
        If you use this EnvironmentRunner, note that pytorch won't let tensors with require_grad=True be sent across
        process boundaries. So all elements of your TimestepDatas must be detach()'d.
        """

        # Allow the policy to do anything it needs to do before collection (e.g. update the policy on the processes)
        # This will trigger the receive_update_process_bundle callback on each process
        if self._create_update_process_bundle is not None:
            process_bundles = self._create_update_process_bundle()

            if process_bundles is not None:
                for process_id, bundle in enumerate(process_bundles):
                    self._send_message_to_process(process_id, "update_state", bundle)

        # All lambdas need to be cloudpickle'd because regular pickle can't handle them
        # Note that env_spec, within task_spec, is optionally a lambda
        task_spec = cloudpickle.dumps(task_spec)

        total_timesteps = 0
        all_timestep_data = []
        all_rewards_to_report = []
        all_logs_to_report = []

        self._send_message_to_all("start_episode", task_spec)

        for process in self._process_managers:
            timesteps, per_timestep_data, rewards_to_report, logs_to_report = process.outgoing_queue.get()
            total_timesteps += timesteps
            all_timestep_data.append(per_timestep_data[0])  # Since batch returns a list of lists, with outer len = 1
            all_rewards_to_report.extend(rewards_to_report)
            all_logs_to_report.extend(logs_to_report)

        return total_timesteps, all_timestep_data, all_rewards_to_report, all_logs_to_report

    def cleanup(self, task_spec):
        self._send_message_to_all("kill", None)
