import torch.multiprocessing as multiprocessing
from continual_rl.experiments.environment_runners.environment_runner_base import EnvironmentRunnerBase
from continual_rl.experiments.environment_runners.full_parallel.collection_process import CollectionProcess


class EnvironmentRunnerFullParallel(EnvironmentRunnerBase):
    """
    Runs the entirety of collection on separate processes. Uses pytorch multiprocessing so share_memory() can be used.
    """
    def __init__(self, policy, num_parallel_processes, timesteps_per_collection, render_collection_freq=None):
        super().__init__()
        self._processes = [CollectionProcess(policy, timesteps_per_collection, render_collection_freq)
                           for _ in range(num_parallel_processes)]

        for process in self._processes:
            multiprocessing.Process(target=process.process_queue)  # TODO: if it takes too long, don't do in constructor

    def collect_data(self, time_batch_size, env_spec, preprocessor, task_id, episode_renderer=None,
                     early_stopping_condition=None):
        collection_request_data = (time_batch_size, env_spec, preprocessor, task_id,
                                   episode_renderer, early_stopping_condition)

        total_timesteps = 0
        all_timestep_data = []
        all_rewards_to_report = []
        all_logs_to_report = []

        for process in self._processes:
            # TODO: update policy
            process.incoming_queue.put(("start_episode", collection_request_data))

        for process in self._processes:
            timesteps, per_timestep_data, rewards_to_report, logs_to_report = process.outgoing_queue.get()
            total_timesteps += timesteps
            all_timestep_data.append(per_timestep_data)
            all_rewards_to_report.extend(rewards_to_report)
            all_logs_to_report.extend(logs_to_report)

        return total_timesteps, all_timestep_data, all_rewards_to_report, all_rewards_to_report, all_logs_to_report
