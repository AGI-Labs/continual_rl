from torch.multiprocessing import Queue
import cloudpickle as pickle
from continual_rl.experiments.environment_runners.environment_runner_batch import EnvironmentRunnerBatch
import traceback


class StateUpdateUnexpectedError(Exception):
    pass


class CollectionProcess():
    def __init__(self, policy, timesteps_per_collection, render_collection_freq=None,
                 receive_update_process_bundle=None, output_dir=None):
        self.incoming_queue = Queue()  # Requests to process
        self.outgoing_queue = Queue()  # Pass data back to caller

        # Only supporting 1 parallel env for now
        self._episode_runner = EnvironmentRunnerBatch(policy, num_parallel_envs=1,
                                                      timesteps_per_collection=timesteps_per_collection,
                                                      render_collection_freq=render_collection_freq,
                                                      output_dir=output_dir)

        self._receive_update_process_bundle = receive_update_process_bundle

    def try_process_queue(self):
        try:
            self._process_queue()
        except Exception as e:
            print(f"Failed with exception: {e}")
            traceback.print_exc()
            self.outgoing_queue.put(None)  # Kill signal

    def _process_queue(self):
        while True:
            next_message = self.incoming_queue.get()
            action_id, content = next_message

            if action_id == "kill":
                self._episode_runner.cleanup()
                break

            elif action_id == "start_episode":
                task_spec = pickle.loads(content)
                results = self._episode_runner.collect_data(task_spec)
                self.outgoing_queue.put(results)

            elif action_id == "update_state":
                if self._receive_update_process_bundle is not None:
                    self._receive_update_process_bundle(content)
                else:
                    raise StateUpdateUnexpectedError("State update received when none was expected")
