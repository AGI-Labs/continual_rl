# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Taken from https://raw.githubusercontent.com/facebookresearch/torchbeast/3f3029cf3d6d488b8b8f952964795f451a49048f/torchbeast/monobeast.py
# and modified

import os
import logging
import pprint
import time
import timeit
import traceback
import typing
import copy
import psutil
import numpy as np
import queue
import cloudpickle
from torch.multiprocessing import Pool
import threading
import json
import shutil
import signal

import torch
import multiprocessing as py_mp
from torch import multiprocessing as mp

from continual_rl.policies.impala.torchbeast.core import environment
from continual_rl.policies.impala.torchbeast.core import prof
from continual_rl.utils.utils import Utils
from continual_rl.policies.impala.vtrace_loss_handler import VtraceLossHandler
from continual_rl.policies.impala.ddpg_loss_handler import DdpgLossHandler


Buffers = typing.Dict[str, typing.List[torch.Tensor]]


class LearnerThreadState():
    STARTING, RUNNING, STOP_REQUESTED, STOPPED = range(4)

    def __init__(self):
        """
        This class is a helper class to manage communication of state between threads. For now I'm assuming just
        setting state is atomic enough to not require further thread safety.
        """
        self.state = self.STARTING
        self.lock = threading.Lock()

    def wait_for(self, desired_state_list, timeout=300):
        time_passed = 0
        delta = 0.1  # seconds

        while self.state not in desired_state_list and time_passed < timeout:
            time.sleep(delta)
            time_passed += delta

        if time_passed > timeout:
            print(f"Gave up on waiting due to timeout. Desired list: {desired_state_list}, current state: {self.state}")  # TODO: not print


class Monobeast():
    def __init__(self, model_flags, observation_space, action_spaces, policy_class):
        self._model_flags = model_flags

        # The latest full episode's set of observations generated by actor with actor_index == 0
        self._videos_to_log = py_mp.Manager().Queue(maxsize=1)

        # Moved some of the original Monobeast code into a setup function, to make class objects
        self.buffers, self.actor_model, self.learner_model, self.plogger, self.logger, self.checkpointpath \
            = self.setup(model_flags, observation_space, action_spaces, policy_class)

        if model_flags.continuous_actions:
            self._loss_handler = DdpgLossHandler(model_flags, self.learner_model)
        else:
            self._loss_handler = VtraceLossHandler(model_flags, self.learner_model)

        # Keep track of our threads/processes so we can clean them up.
        self._learner_thread_states = []
        self._actor_processes = []

        # train() will get called multiple times (once per task, per cycle). The current assumption is that only
        # one train() should be running a time, and that all others have been cleaned up. These parameters help us
        # ensure this is true.
        self._train_loop_id_counter = 0
        self._train_loop_id_running = None

        # If we're reloading a task, we need to start from where we left off. This gets populated by load, if
        # applicable
        self.last_timestep_returned = 0

        # Created during train, saved so we can die cleanly
        self.free_queue = None
        self.full_queue = None

        # Pillow sometimes pollutes the logs, see: https://github.com/python-pillow/Pillow/issues/5096
        logging.getLogger("PIL.PngImagePlugin").setLevel(logging.CRITICAL + 1)

    # Functions designed to be overridden by subclasses of Monobeast
    def on_act_unroll_complete(self, task_flags, actor_index, agent_output, env_output, new_buffers):
        """
        Called after every unroll in every process running act(). Note that this happens in separate processes, and
        data will need to be shepherded accordingly.
        """
        pass

    def get_batch_for_training(self, batch):
        """
        Create a new batch based on the old, with any modifications desired. (E.g. augmenting with entries from
        a replay buffer.) This is run in each learner thread.
        """
        return batch

    def custom_loss(self, task_flags, model, initial_agent_state):
        """
        Create a new loss. This is added to the existing losses before backprop. Any returned stats will be added
        to the logged stats. If a stat's key ends in "_loss", it'll automatically be plotted as well.
        This is run in each learner thread.
        :return: (loss, dict of stats)
        """
        return 0, {}

    # Core Monobeast functionality
    def setup(self, model_flags, observation_space, action_spaces, policy_class):
        os.environ["OMP_NUM_THREADS"] = "1"
        logging.basicConfig(
            format=(
                "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
            ),
            level=0,
        )

        logger = Utils.create_logger(os.path.join(model_flags.savedir, "impala_logs.log"))
        plogger = Utils.create_logger(os.path.join(model_flags.savedir, "impala_results.log"))

        checkpointpath = os.path.join(model_flags.savedir, "model.tar")

        if model_flags.num_buffers is None:  # Set sensible default for num_buffers.
            model_flags.num_buffers = max(2 * model_flags.num_actors, model_flags.batch_size)
        if model_flags.num_actors >= model_flags.num_buffers:
            raise ValueError("num_buffers should be larger than num_actors")
        if model_flags.num_buffers < model_flags.batch_size:
            raise ValueError("num_buffers should be larger than batch_size")

        # Convert the device string into an actual device
        model_flags.device = torch.device(model_flags.device)

        model = policy_class(observation_space, action_spaces, model_flags)
        buffers = self.create_buffers(model_flags, observation_space.shape, model.num_actions)

        model.share_memory()

        learner_model = policy_class(
            observation_space, action_spaces, model_flags
        ).to(device=model_flags.device)

        return buffers, model, learner_model, plogger, logger, checkpointpath

    def act(
            self,
            model_flags,
            task_flags,
            actor_index: int,
            free_queue: py_mp.Queue,
            full_queue: py_mp.Queue,
            model: torch.nn.Module,
            buffers: Buffers,
            initial_agent_state_buffers,
    ):
        env = None
        try:
            self.logger.info("Actor %i started.", actor_index)
            timings = prof.Timings()  # Keep track of how fast things are.

            gym_env, seed = Utils.make_env(task_flags.env_spec, create_seed=True)
            self.logger.info(f"Environment and libraries setup with seed {seed}")

            # Parameters involved in rendering behavior video
            observations_to_render = []  # Only populated by actor 0

            env = environment.Environment(gym_env)
            env_output = env.initial()
            agent_state = model.initial_state(batch_size=1)
            agent_output, unused_state = model(env_output, task_flags.action_space_id, agent_state)

            # Make sure to kill the env cleanly if a terminate signal is passed. (Will not go through the finally)
            def end_task(*args):
                env.close()

            signal.signal(signal.SIGTERM, end_task)

            while True:
                index = free_queue.get()
                if index is None:
                    break

                # Write old rollout end.
                for key in env_output:
                    buffers[key][index][0, ...] = env_output[key]
                for key in agent_output:
                    buffers[key][index][0, ...] = agent_output[key]
                for i, tensor in enumerate(agent_state):
                    initial_agent_state_buffers[index][i][...] = tensor

                # Do new rollout.
                for t in range(model_flags.unroll_length):
                    timings.reset()

                    with torch.no_grad():
                        agent_output, agent_state = model(env_output, task_flags.action_space_id, agent_state)

                    timings.time("model")

                    env_output = env.step(agent_output["action"])

                    timings.time("step")

                    for key in env_output:
                        buffers[key][index][t + 1, ...] = env_output[key]
                    for key in agent_output:
                        buffers[key][index][t + 1, ...] = agent_output[key]

                    # Save off video if appropriate
                    if actor_index == 0:
                        if env_output['done'].squeeze():
                            # If we have a video in there, replace it with this new one
                            try:
                                self._videos_to_log.get(timeout=1)
                            except queue.Empty:
                                pass
                            except (FileNotFoundError, ConnectionRefusedError, ConnectionResetError, RuntimeError) as e:
                                # Sometimes it seems like the videos_to_log socket fails. Since video logging is not
                                # mission-critical, just let it go.
                                self.logger.warning(
                                    f"Video logging socket seems to have failed with error {e}. Aborting video log.")
                                pass

                            self._videos_to_log.put(copy.deepcopy(observations_to_render))
                            observations_to_render.clear()

                        observations_to_render.append(env_output['frame'].squeeze(0).squeeze(0)[-1])

                    timings.time("write")

                new_buffers = {key: buffers[key][index] for key in buffers.keys()}
                self.on_act_unroll_complete(task_flags, actor_index, agent_output, env_output, new_buffers)
                full_queue.put(index)

            if actor_index == 0:
                self.logger.info("Actor %i: %s", actor_index, timings.summary())

        except KeyboardInterrupt:
            pass  # Return silently.
        except Exception as e:
            self.logger.error(f"Exception in worker process {actor_index}: {e}")
            traceback.print_exc()
            print()
            raise e
        finally:
            self.logger.info(f"Finalizing actor {actor_index}")
            if env is not None:
                env.close()

    def get_batch(
            self,
            flags,
            free_queue: py_mp.Queue,
            full_queue: py_mp.Queue,
            buffers: Buffers,
            initial_agent_state_buffers,
            timings,
            lock,
    ):
        with lock:
            timings.time("lock")
            indices = [full_queue.get() for _ in range(flags.batch_size)]
            timings.time("dequeue")
        batch = {
            key: torch.stack([buffers[key][m] for m in indices], dim=1) for key in buffers
        }
        initial_agent_state = (
            torch.cat(ts, dim=1)
            for ts in zip(*[initial_agent_state_buffers[m] for m in indices])
        )
        timings.time("batch")
        for m in indices:
            free_queue.put(m)
        timings.time("enqueue")

        batch = {k: t.to(device=flags.device, non_blocking=True) for k, t in batch.items()}
        initial_agent_state = tuple(
            t.to(device=flags.device, non_blocking=True) for t in initial_agent_state
        )
        timings.time("device")
        return batch, initial_agent_state

    def compute_loss(self, task_flags, batch, initial_agent_state, with_custom_loss=True):
        custom_loss_fn = self.custom_loss if with_custom_loss else None
        return self._loss_handler.compute_loss(task_flags, batch, initial_agent_state, custom_loss_fn=custom_loss_fn)

    def learn(
            self,
            task_flags,
            batch,
            initial_agent_state,
            lock,
    ):
        """Performs a learning (optimization) step."""
        with lock:
            # Only log the real batch of new data, not the manipulated version for training, so save it off
            batch_for_logging = copy.deepcopy(batch)

            # Prepare the batch for training (e.g. augmenting with more data)
            batch = self.get_batch_for_training(batch)

            stats = self.compute_loss(task_flags, batch, initial_agent_state)  # TODO: poorly named since it does do the gradient steps

            self.actor_model.load_state_dict(self.learner_model.state_dict())

            # The episode_return may be nan if we're using an EpisodicLifeEnv (for Atari), where episode_return is nan
            # until the end of the game, where a real return is produced.
            batch_done_flags = batch_for_logging["done"] * ~torch.isnan(batch_for_logging["episode_return"])
            episode_returns = batch_for_logging["episode_return"][batch_done_flags]
            stats.update({
                "episode_returns": tuple(episode_returns.cpu().numpy()),
                "mean_episode_return": torch.mean(episode_returns).item()
            })

            return stats

    def create_buffer_specs(self, unroll_length, obs_shape, num_actions):
        T = unroll_length
        action_spec = dict(size=(T + 1, num_actions), dtype=torch.float32) if self._model_flags.continuous_actions else \
            dict(size=(T + 1,), dtype=torch.int64)
        frame_dtype = torch.uint8 if self._model_flags.encode_frame_as_uint8 else torch.float32
        specs = dict(
            frame=dict(size=(T + 1, *obs_shape), dtype=frame_dtype),
            reward=dict(size=(T + 1,), dtype=torch.float32),
            done=dict(size=(T + 1,), dtype=torch.bool),
            episode_return=dict(size=(T + 1,), dtype=torch.float32),
            episode_step=dict(size=(T + 1,), dtype=torch.int32),
            policy_logits=dict(size=(T + 1, num_actions), dtype=torch.float32),
            baseline=dict(size=(T + 1,), dtype=torch.float32),
            last_action=dict(size=(T + 1,), dtype=torch.int64),
            action=action_spec,
        )
        return specs

    def create_buffers(self, flags, obs_shape, num_actions) -> Buffers:
        specs = self.create_buffer_specs(flags.unroll_length, obs_shape, num_actions)
        buffers: Buffers = {key: [] for key in specs}
        for _ in range(flags.num_buffers):
            for key in buffers:
                buffers[key].append(torch.empty(**specs[key]).share_memory_())
        return buffers

    def create_learn_threads(self, batch_and_learn, stats_lock, thread_free_queue, thread_full_queue):
        learner_thread_states = [LearnerThreadState() for _ in range(self._model_flags.num_learner_threads)]
        batch_lock = threading.Lock()
        learn_lock = threading.Lock()
        threads = []
        for i in range(self._model_flags.num_learner_threads):
            thread = threading.Thread(
                target=batch_and_learn, name="batch-and-learn-%d" % i, args=(i, stats_lock, learner_thread_states[i], batch_lock, learn_lock, thread_free_queue, thread_full_queue)
            )
            thread.start()
            threads.append(thread)
        return threads, learner_thread_states

    def cleanup(self):
        # We've finished the task, so reset the appropriate counter
        self.logger.info("Finishing task, setting timestep_returned to 0")
        self.last_timestep_returned = 0

        # Ensure the training loop will end
        self._train_loop_id_running = None

        self._cleanup_parallel_workers()

    def _cleanup_parallel_workers(self):
        self.logger.info("Cleaning up actors")

        # Send the signal to the actors to die, and resume them so they can (if they're not already dead)
        for actor_index, actor in enumerate(self._actor_processes):
            self.free_queue.put(None)
            try:
                actor_process = psutil.Process(actor.pid)
                actor_process.resume()
            except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
                # If it's already dead, just let it go
                pass

        # Try wait for the actors to end cleanly. If they do not, try to force a termination
        for actor_index, actor in enumerate(self._actor_processes):
            try:
                actor.join(30)  # Give up on waiting eventually

                if actor.exitcode is None:
                    actor.terminate()

                actor.close()
                self.logger.info(f"[Actor {actor_index}] Cleanup complete")
            except ValueError:  # if actor already killed
                pass

        # Pause the learner so we don't keep churning out results when we're done (or something died)
        self.logger.info("Cleaning up learners")
        for thread_state in self._learner_thread_states:
            thread_state.state = LearnerThreadState.STOP_REQUESTED

        self.logger.info("Cleaning up parallel workers complete")

    def resume_actor_processes(self, ctx, task_flags, actor_processes, free_queue, full_queue, initial_agent_state_buffers):
        # Copy, so iterator and what's being updated are separate
        actor_processes_copy = actor_processes.copy()
        for actor_index, actor in enumerate(actor_processes_copy):
            allowed_statuses = ["running", "sleeping", "disk-sleep"]
            actor_pid = None  # actor.pid fails with ValueError if the process is already closed

            try:
                actor_pid = actor.pid
                actor_process = psutil.Process(actor_pid)
                actor_process.resume()
                recreate_actor = not actor_process.is_running() or actor_process.status() not in allowed_statuses
            except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
                recreate_actor = True

            if recreate_actor:
                # Kill the original ctx.Process object, rather than the one attached to by pid
                # Attempting to fix an issue where the actor processes are hanging, CPU util shows zero
                try:
                    actor_processes[actor_index].kill()
                    actor_processes[actor_index].join()
                    actor_processes[actor_index].close()
                except ValueError:  # if actor already killed
                    pass

                self.logger.warn(
                    f"Actor with pid {actor_pid} in actor index {actor_index} was unable to be restarted. Recreating...")
                new_actor = ctx.Process(
                    target=self.act,
                    args=(
                        self._model_flags,
                        task_flags,
                        actor_index,
                        free_queue,
                        full_queue,
                        self.actor_model,
                        self.buffers,
                        initial_agent_state_buffers,
                    ),
                )
                new_actor.start()
                actor_processes[actor_index] = new_actor

    def save(self, output_path):
        if self._model_flags.disable_checkpoint:
            return

        model_file_path = os.path.join(output_path, "model.tar")

        # Back up previous model (sometimes they can get corrupted)
        if os.path.exists(model_file_path):
            shutil.copyfile(model_file_path, os.path.join(output_path, "model_bak.tar"))

        # Save the model
        self.logger.info(f"Saving model to {output_path}")

        checkpoint_data = {
                "model_state_dict": self.actor_model.state_dict(),
            }

        checkpoint_data.update(self._loss_handler.get_save_data())

        torch.save(checkpoint_data, model_file_path)

        # Save metadata
        metadata_path = os.path.join(output_path, "impala_metadata.json")
        metadata = {"last_timestep_returned": self.last_timestep_returned}
        with open(metadata_path, "w+") as metadata_file:
            json.dump(metadata, metadata_file)

    def load(self, output_path):
        model_file_path = os.path.join(output_path, "model.tar")
        if os.path.exists(model_file_path):
            self.logger.info(f"Loading model from {output_path}")
            checkpoint = torch.load(model_file_path, map_location="cpu")

            self.actor_model.load_state_dict(checkpoint["model_state_dict"])
            self.learner_model.load_state_dict(checkpoint["model_state_dict"])
            self._loss_handler.load_save_data(checkpoint)
        else:
            self.logger.info("No model to load, starting from scratch")

        # Load metadata
        metadata_path = os.path.join(output_path, "impala_metadata.json")
        if os.path.exists(metadata_path):
            self.logger.info(f"Loading impala metdata from {metadata_path}")
            with open(metadata_path, "r") as metadata_file:
                metadata = json.load(metadata_file)

            self.last_timestep_returned = metadata["last_timestep_returned"]

    def train(self, task_flags):  # pylint: disable=too-many-branches, too-many-statements
        T = self._model_flags.unroll_length
        B = self._model_flags.batch_size

        self._loss_handler.initialize_for_task(task_flags)

        # Add initial RNN state.
        initial_agent_state_buffers = []
        for _ in range(self._model_flags.num_buffers):
            state = self.actor_model.initial_state(batch_size=1)
            for t in state:
                t.share_memory_()
            initial_agent_state_buffers.append(state)

        # Setup actor processes and kick them off
        self._actor_processes = []
        ctx = mp.get_context("fork")

        # See: https://stackoverflow.com/questions/47085458/why-is-multiprocessing-queue-get-so-slow for why Manager
        self.free_queue = py_mp.Manager().Queue()
        self.full_queue = py_mp.Manager().Queue()

        for i in range(self._model_flags.num_actors):
            actor = ctx.Process(
                target=self.act,
                args=(
                    self._model_flags,
                    task_flags,
                    i,
                    self.free_queue,
                    self.full_queue,
                    self.actor_model,
                    self.buffers,
                    initial_agent_state_buffers,
                ),
            )
            actor.start()
            self._actor_processes.append(actor)

        stat_keys = [
            "total_loss",
            "mean_episode_return",
            "pg_loss",
            "baseline_loss",
            "entropy_loss",
        ]
        self.logger.info("# Step\t%s", "\t".join(stat_keys))

        step, collected_stats = self.last_timestep_returned, {}
        self._stats_lock = threading.Lock()

        def batch_and_learn(i, lock, thread_state, batch_lock, learn_lock, thread_free_queue, thread_full_queue):
            """Thread target for the learning process."""
            try:
                nonlocal step, collected_stats
                timings = prof.Timings()

                while True:
                    # If we've requested a stop, indicate it and end the thread
                    with thread_state.lock:
                        if thread_state.state == LearnerThreadState.STOP_REQUESTED:
                            thread_state.state = LearnerThreadState.STOPPED
                            return

                        thread_state.state = LearnerThreadState.RUNNING

                    timings.reset()
                    batch, agent_state = self.get_batch(
                        self._model_flags,
                        thread_free_queue,
                        thread_full_queue,
                        self.buffers,
                        initial_agent_state_buffers,
                        timings,
                        batch_lock,
                    )
                    stats = self.learn(
                        task_flags, batch, agent_state, learn_lock
                    )
                    timings.time("learn")
                    with lock:
                        step += T * B
                        to_log = dict(step=step)
                        to_log.update({k: stats[k] for k in stat_keys if k in stats})
                        self.plogger.info(to_log)

                        # We might collect stats more often than we return them to the caller, so collect them all
                        for key in stats.keys():
                            if key not in collected_stats:
                                collected_stats[key] = []

                            if isinstance(stats[key], tuple) or isinstance(stats[key], list):
                                collected_stats[key].extend(stats[key])
                            else:
                                collected_stats[key].append(stats[key])
            except Exception as e:
                self.logger.error(f"Learner thread failed with exception {e}")
                raise e

            if i == 0:
                self.logger.info("Batch and learn: %s", timings.summary())

            thread_state.state = LearnerThreadState.STOPPED

        for m in range(self._model_flags.num_buffers):
            self.free_queue.put(m)

        threads, self._learner_thread_states = self.create_learn_threads(batch_and_learn, self._stats_lock, self.free_queue, self.full_queue)

        # Create the id for this train loop, and only loop while it is the active id
        assert self._train_loop_id_running is None, "Attempting to start a train loop while another is active."
        train_loop_id = self._train_loop_id_counter
        self._train_loop_id_counter += 1
        self._train_loop_id_running = train_loop_id
        self.logger.info(f"Starting train loop id {train_loop_id}")

        timer = timeit.default_timer
        try:
            while self._train_loop_id_running == train_loop_id:
                start_step = step
                start_time = timer()
                time.sleep(self._model_flags.seconds_between_yields)

                # Copy right away, because there's a race where stats can get re-set and then certain things set below
                # will be missing (eg "step")
                with self._stats_lock:
                    stats_to_return = copy.deepcopy(collected_stats)
                    collected_stats.clear()

                sps = (step - start_step) / (timer() - start_time)

                # Aggregate our collected values. Do it with mean so it's not sensitive to the number of times
                # learning occurred in the interim
                mean_return = np.array(stats_to_return.get("episode_returns", [np.nan])).mean()
                stats_to_return["mean_episode_return"] = mean_return

                # Make a copy of the keys so we're not updating it as we iterate over it
                for key in list(stats_to_return.keys()).copy():
                    if key.endswith("loss") or key == "total_norm":
                        # Replace with the number we collected and the mean value, otherwise the logs are very verbose
                        stats_to_return[f"{key}_count"] = len(np.array(stats_to_return.get(key, [])))
                        stats_to_return[key] = np.array(stats_to_return.get(key, [np.nan])).mean()

                self.logger.info(
                    "Steps %i @ %.1f SPS. Mean return %f. Stats:\n%s",
                    step,
                    sps,
                    mean_return,
                    pprint.pformat(stats_to_return),
                )
                stats_to_return["step"] = step
                stats_to_return["step_delta"] = step - self.last_timestep_returned

                try:
                    video = self._videos_to_log.get(block=False)
                    stats_to_return["video"] = video
                except queue.Empty:
                    pass
                except (FileNotFoundError, ConnectionRefusedError, ConnectionResetError, RuntimeError) as e:
                    # Sometimes it seems like the videos_to_log socket fails. Since video logging is not
                    # mission-critical, just let it go.
                    self.logger.warning(f"Video logging socket seems to have failed with error {e}. Aborting video log.")
                    pass

                # This block sets us up to yield our results in batches, pausing everything while yielded.
                if self.last_timestep_returned != step:
                    self.last_timestep_returned = step

                    # Stop learn threads, they are recreated after yielding. 
                    # Do this before the actors in case we need to do a last batch
                    self.logger.info("Stopping learners")
                    for thread_id, thread_state in enumerate(self._learner_thread_states):
                        wait = False
                        with thread_state.lock:
                            if thread_state.state != LearnerThreadState.STOPPED and threads[thread_id].is_alive():
                                thread_state.state = LearnerThreadState.STOP_REQUESTED
                                wait = True

                        # Wait for it to stop, otherwise we have training overlapping with eval, and possibly
                        # the thread creation below
                        if wait:
                            thread_state.wait_for([LearnerThreadState.STOPPED], timeout=30)

                    # The actors will keep going unless we pause them, so...do that.
                    if self._model_flags.pause_actors_during_yield:
                        for actor in self._actor_processes:
                            psutil.Process(actor.pid).suspend()

                    # Make sure the queue is empty (otherwise things can get dropped in the shuffle)
                    # (Not 100% sure relevant but:) https://stackoverflow.com/questions/19257375/python-multiprocessing-queue-put-not-working-for-semi-large-data
                    while not self.free_queue.empty():
                        try:
                            self.free_queue.get(block=False)
                        except queue.Empty:
                            # Race between empty check and get, I guess
                            break

                    while not self.full_queue.empty():
                        try:
                            self.full_queue.get(block=False)
                        except queue.Empty:
                            # Race between empty check and get, I guess
                            break

                    yield stats_to_return

                    # Ensure everything is set back up to train
                    self.actor_model.train()
                    self.learner_model.train()

                    # Resume the actors. If one is dead, replace it with a new one
                    if self._model_flags.pause_actors_during_yield:
                        self.resume_actor_processes(ctx, task_flags, self._actor_processes, self.free_queue, self.full_queue,
                                                    initial_agent_state_buffers)

                    # Resume the learners by creating new ones
                    self.logger.info("Restarting learners")
                    threads, self._learner_thread_states = self.create_learn_threads(batch_and_learn, self._stats_lock, self.free_queue, self.full_queue)
                    self.logger.info("Restart complete")

                    for m in range(self._model_flags.num_buffers):
                        self.free_queue.put(m)
                    self.logger.info("Free queue re-populated")

        except KeyboardInterrupt:
            pass

        except Exception as e:
            self.logger.error(f"Exception in main process: {e}")
            traceback.print_exc()
            print()
            raise e

        finally:
            self._cleanup_parallel_workers()
            for thread in threads:
                thread.join()
            self.logger.info("Learning finished after %d steps.", step)

    @staticmethod
    def _collect_test_episode(pickled_args):
        task_flags, logger, model = cloudpickle.loads(pickled_args)

        gym_env, seed = Utils.make_env(task_flags.env_spec, create_seed=True)
        logger.info(f"Environment and libraries setup with seed {seed}")
        env = environment.Environment(gym_env)
        observation = env.initial()
        done = False
        step = 0
        returns = []

        while not done:
            if task_flags.mode == "test_render":
                env.gym_env.render()
            agent_outputs = model(observation, task_flags.action_space_id)
            policy_outputs, _ = agent_outputs
            observation = env.step(policy_outputs["action"])
            step += 1
            done = observation["done"].item() and not torch.isnan(observation["episode_return"])

            # NaN if the done was "fake" (e.g. Atari). We want real scores here so wait for the real return.
            if done:
                returns.append(observation["episode_return"].item())
                logger.info(
                    "Episode ended after %d steps. Return: %.1f",
                    observation["episode_step"].item(),
                    observation["episode_return"].item(),
                )

        env.close()
        return step, returns

    def test(self, task_flags, num_episodes: int = 10):
        if not self._model_flags.no_eval_mode:
            self.actor_model.eval()

        returns = []
        step = 0

        # Break the number of episodes we need to run up into batches of num_parallel, which get run concurrently
        for batch_start_id in range(0, num_episodes, self._model_flags.eval_episode_num_parallel):
            # If we are in the last batch, only do the necessary number, otherwise do the max num in parallel
            batch_num_episodes = min(num_episodes - batch_start_id, self._model_flags.eval_episode_num_parallel)

            with Pool(processes=batch_num_episodes) as pool:
                async_objs = []
                for episode_id in range(batch_num_episodes):
                    pickled_args = cloudpickle.dumps((task_flags, self.logger, self.actor_model))
                    async_obj = pool.apply_async(self._collect_test_episode, (pickled_args,))
                    async_objs.append(async_obj)

                for async_obj in async_objs:
                    episode_step, episode_returns = async_obj.get()
                    step += episode_step
                    returns.extend(episode_returns)

        self.logger.info(
            "Average returns over %i episodes: %.1f", len(returns), sum(returns) / len(returns)
        )
        stats = {"episode_returns": returns, "step": step, "num_episodes": len(returns)}

        yield stats
