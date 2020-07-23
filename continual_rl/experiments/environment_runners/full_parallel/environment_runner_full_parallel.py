import torch.multiprocessing as multiprocessing
import cloudpickle
from continual_rl.experiments.environment_runners.environment_runner_base import EnvironmentRunnerBase
from continual_rl.experiments.environment_runners.full_parallel.collection_process import CollectionProcess


class EnvironmentRunnerFullParallel(EnvironmentRunnerBase):
    """
    Runs the entirety of collection on separate processes. Uses pytorch multiprocessing so share_memory() can be used.
    """
    def __init__(self, policy, num_parallel_processes, timesteps_per_collection, render_collection_freq=None,
                 before_collection_handler=None):
        super().__init__()
        self._before_collection_handler = before_collection_handler
        self._process_managers = [CollectionProcess(policy, timesteps_per_collection, render_collection_freq)
                           for _ in range(num_parallel_processes)]

        for manager in self._process_managers:
            process = multiprocessing.Process(target=manager.process_queue)  # TODO: if it takes too long, don't do in constructor...also should CollectionProcess have this?
            process.start()

    def send_message_to_all(self, message_id, message_content):
        for process in self._process_managers:
            process.incoming_queue.put((message_id, message_content))

    def collect_data(self, time_batch_size, env_spec, preprocessor, task_id, episode_renderer=None,
                     early_stopping_condition=None):  # TODO: common naming pattern for my lambdas for clarity? (handler?)
        """
        If you use this EnvironmentRunner, note that pytorch won't let tensors with require_grad=True be sent across
        process boundaries. So all elements of your InfoToStores must be detach()'d.
        """

        # Allow the policy to do anything it needs to do before collection (e.g. update the policy on the processes)
        if self._before_collection_handler is not None:
            self._before_collection_handler(self)  # Passing self in gives the caller a lot of power (TODO?)

        # All lambdas need to be cloudpickle'd because regular pickle can't handle them
        # Note that env_spec is optionally a lambda
        env_spec = cloudpickle.dumps(env_spec)
        preprocessor = cloudpickle.dumps(preprocessor)
        episode_renderer = cloudpickle.dumps(episode_renderer)
        early_stopping_condition = cloudpickle.dumps(early_stopping_condition)

        collection_request_data = (time_batch_size, env_spec, preprocessor, task_id,
                                   episode_renderer, early_stopping_condition)

        total_timesteps = 0
        all_timestep_data = []
        all_rewards_to_report = []
        all_logs_to_report = []

        self.send_message_to_all("start_episode", collection_request_data)

        for process in self._process_managers:
            timesteps, per_timestep_data, rewards_to_report, logs_to_report = process.outgoing_queue.get()
            total_timesteps += timesteps
            all_timestep_data.append(per_timestep_data)
            all_rewards_to_report.extend(rewards_to_report)
            all_logs_to_report.extend(logs_to_report)

        return total_timesteps, all_timestep_data, all_rewards_to_report, all_logs_to_report
