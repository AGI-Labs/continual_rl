from torch.multiprocessing import Queue
import cloudpickle
import torch
import numpy as np
from continual_rl.experiments.environment_runners.environment_runner_batch import EnvironmentRunnerBatch


class StateUpdateUnexpectedError(Exception):
    pass


class CollectionProcess():
    def __init__(self, policy, timesteps_per_collection, worker_id, seed=None, render_collection_freq=None,
                 receive_update_process_bundle=None):
        self.incoming_queue = Queue()  # Requests to process
        self.outgoing_queue = Queue()  # Pass data back to caller

        # Only supporting 1 parallel env for now
        self._episode_runner = EnvironmentRunnerBatch(policy, num_parallel_envs=1,
                                                      timesteps_per_collection=timesteps_per_collection,
                                                      render_collection_freq=render_collection_freq)

        if seed is not None:  # TODO test setting a seed and the numpy seeding
            self._seed = seed + worker_id
        else:
            self._seed = worker_id  # TODO This makes the seed non-random.... Seems to be required by https://github.com/pytorch/pytorch/issues/33546

        self._receive_update_process_bundle = receive_update_process_bundle

    def process_queue(self):
        if self._seed is not None:
            torch.manual_seed(self._seed)
            np.random.seed(self._seed)
        else:
            torch.seed()
            np.random.seed()

        while True:
            next_message = self.incoming_queue.get()
            action_id, content = next_message

            if action_id == "kill":
                break

            elif action_id == "start_episode":
                time_batch_size, env_spec, preprocessor, task_id, episode_renderer, early_stopping_condition = content

                env_spec = cloudpickle.loads(env_spec)
                preprocessor = cloudpickle.loads(preprocessor)
                episode_renderer = cloudpickle.loads(episode_renderer)
                early_stopping_condition = cloudpickle.loads(early_stopping_condition)

                results = self._episode_runner.collect_data(time_batch_size, env_spec, preprocessor, task_id,
                                                            episode_renderer, early_stopping_condition)
                self.outgoing_queue.put(results)

            elif action_id == "update_state":
                if self._receive_update_process_bundle is not None:
                    self._receive_update_process_bundle(content)
                else:
                    raise StateUpdateUnexpectedError("State update received when none was expected")
