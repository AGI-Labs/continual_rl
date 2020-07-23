import uuid
from torch.multiprocessing import Queue
from continual_rl.experiments.environment_runners.environment_runner_batch import EnvironmentRunnerBatch


class CollectionProcess():
    def __init__(self, policy, timesteps_per_collection, render_collection_freq=None):
        self.incoming_queue = Queue()  # Requests to process
        self.outgoing_queue = Queue()  # Pass data back to caller

        # Only supporting 1 parallel env for now
        self._episode_runner = EnvironmentRunnerBatch(policy, num_parallel_envs=1,
                                                      timesteps_per_collection=timesteps_per_collection,
                                                      render_collection_freq=render_collection_freq)

    def process_queue(self):
        while True:
            next_message = self.incoming_queue.get()
            action_id, content = next_message

            if action_id == "kill":
                break

            elif action_id == "start_episode":
                results = self._episode_runner.collect_data(*content)
                self.outgoing_queue.put(results)
