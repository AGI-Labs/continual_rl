import torch
from continual_rl.policies.sane.hypothesis.replay_buffer import ReplayBuffer, ReplayEntry
import uuid
import time
from multiprocessing.queues import Empty
import torch.utils.checkpoint
from continual_rl.policies.sane.hypothesis_directory.utils import Utils


"""
This project has three types of process:
1. Core process: owns the overall train loop
2. Usage process: doesn't change the hypotheses, just uses them to get the action as a function of input. One of these exists per environment.
3. Train process: updates the hypotheses. Multiple hypotheses exist on one train process 
    (would have one hypothesis per, except pytorch allocates a minimum gpu usage per process, and there's no easy way to reduce this.

Usage and Train do not talk to each other. Only Core <=> Usage and Core <=> Train
"""


class ProcessTerminatedException(Exception):
    def __init__(self, message):
        super().__init__(message)


class CoreToTrainComms(object):
    """
    This is associated with one hypothesis, and handles all interaction over the wire from core process to the process that handles hypothesis training. 
    """
    # TODO: currently this is the *external* interface. Rename appropriately if I keep this.
    # TODO: it currently uses a lot of Hypothesis-internals - it's manipulating the core-process-local hypothesis primarily to add replay_entries to grab as negative_examples
    # TODO: clean this path up
    def __init__(self, hypothesis_accessor, hypothesis, process_comms):
        self.process_comms = process_comms
        self.process = None
        self._hypothesis = hypothesis
        self._hypothesis_accessor = hypothesis_accessor

        # Caches used in the sending process, for efficiency
        # Compact locally so we cap how much we're sending over the wire
        self._to_add_to_replay_cache = [] #ReplayBuffer(non_permanent_maxlen=self._hypothesis._replay_buffer_size)
        self._bulk_transfer_cache = []

        # Used for grabbing negative examples, so this is sort of best-effort. (Also used for getting the length of the replay buffer.)
        self._cached_replay_buffer = ReplayBuffer(non_permanent_maxlen=self._hypothesis._replay_buffer_size)  # TODO: hacky
        
        self._tensors_in_flight = []  # To make sure tensors we're sending over don't get garbage collected too soon

    @property
    def logger(self):
        logger = Utils.create_logger(f"{self._hypothesis._output_dir}/hypothesis_{self._hypothesis.friendly_name}_core.log")
        return logger

    def close(self):
        self.process_comms.close()

    def check_outgoing_queue(self):
        while not self.process_comms.outgoing_queue.empty():
            received_request_id, task_result = self.process_comms.outgoing_queue.get(timeout=100)

            if received_request_id is None:
                self.logger.error(f"{self._hypothesis.friendly_name} Received request to shut down experiment.")
                raise ProcessTerminatedException("Process terminated, killing experiment.")
            else:
                self.logger.info(f"Received {received_request_id} {task_result}. No action is being taken on this result.")

    def _construct_packet(self, message_id, request_id, object_to_send, response_requested):
        return (message_id, request_id, self._hypothesis.unique_id, object_to_send, response_requested)

    def send_hypothesis_old(self, hypothesis):
        self.logger.info(f"Sending new hypothesis over: {hypothesis.friendly_name}")
        self.send_task_and_await_result("add_hypothesis", hypothesis)
        self.logger.info(f"Hypothesis successfully sent: {hypothesis.friendly_name}")

    def send_hypothesis(self, hypothesis):
        result = None
        for id in range(5):
            self.logger.info(f"Sending new hypothesis over: {hypothesis.friendly_name}")
            result = self.send_task_and_await_result("add_hypothesis", hypothesis, timeout=60)

            if result is not None:
                self.logger.info(f"Hypothesis successfully sent: {hypothesis.friendly_name}")
                break
            else:
                # This happens when we get the Shared Memory Manager timed out error. Seeing if this is intermittent...
                # so just trying again
                self.logger.info(f"Hypothesis {hypothesis.friendly_name} timed out. Trying again. Try: {id}")

        assert result is not None, "Failed to send hypothesis"

    def send_task_and_await_result(self, message_id, object_to_send, request_id=None, timeout=None):
        """
        Anything that uses this should be idempotent, since it may get called multiple times (if the timeout triggers).
        """
        sent_request_id = uuid.uuid4() if request_id is None else request_id
        self.process_comms.incoming_queue.put(self._construct_packet(message_id, sent_request_id, object_to_send, response_requested=True))

        received_request_id = None
        task_result = None

        while sent_request_id != received_request_id:  # TODO: is this automatically non-thread-blocking?
            try:
                received_request_id, task_result = self.process_comms.outgoing_queue.get(timeout=timeout)
            except Empty:
                # Debugging a hang...
                self.logger.warning(f"Get timed out for {message_id} for hypothesis {self._hypothesis.friendly_name}. Just...letting it go")
                return None

            if received_request_id is None:
                self.logger.info(f"{self._hypothesis.friendly_name} Received request to shut down experiment from {message_id}")
                raise ProcessTerminatedException("Process terminated, killing experiment.")

            if sent_request_id != received_request_id:
                self.process_comms.outgoing_queue.put((received_request_id, task_result))  # If this isn't what we were supposed to get, put it back. TODO: this is hacky...
                self.logger.warning(f"Message received out of order, trying again. Expected {sent_request_id}, Received {received_request_id}")
                time.sleep(0.5)  # TODO: this is so the messages above get spammed less, mostly, but we can still see what happened. I should implement proper logging I guess

        return task_result

    def send_kill_message(self):
        self.process_comms.incoming_queue.put(None)

    def send_delete_message(self, hypothesis):
        self.logger.info("Sending deletion message")
        self.send_task_and_await_result("delete_hypothesis", hypothesis.unique_id)
        self.logger.info("Deletion message complete")

    def send_train_message(self, num_samples, id_start_frac, id_end_frac, num_times_to_train):
        object_to_send = ([num_samples, id_start_frac, id_end_frac, num_times_to_train], {})

        # Fire and forget version
        request_id = uuid.uuid4()
        self.process_comms.incoming_queue.put(self._construct_packet("train", request_id, object_to_send, response_requested=False))  # response_requested: False

    def complete_and_register_replay_entry_to_send(self, replay_entry, reward_received, log_probs, selected_action, parent_hypothesis_comms):  # TODO: update Sync
        replay_entry.reward_received = torch.Tensor([reward_received])
        replay_entry.action_log_prob = torch.Tensor([log_probs])
        replay_entry.selected_action = torch.Tensor([selected_action])

        self.register_replay_entry_for_sending(replay_entry)

        if parent_hypothesis_comms is not None:
            parent_hypothesis_comms.register_replay_entry_for_sending(replay_entry)

    def wait_for_train_to_complete(self):  # TODO: rename?
        self.send_task_and_await_result("ping", {}, timeout=30)  # TODO: Hanging indefinitely on ping sometimes...?

        self._tensors_in_flight.clear()  # We know the train process has cleared its queue, so it's safe to delete

    def register_replay_entry_for_sending(self, x):
        self._to_add_to_replay_cache.append(x)  # TODO: later entries might overwrite earlier in such a way that earlier never actually get seen by the pattern_filter

    def send_replay_cache(self):
        self._hypothesis._replay_buffer.add_many(self._to_add_to_replay_cache)
        self._to_add_to_replay_cache.clear()

    def add_many_to_replay(self, entries):
        self._hypothesis._replay_buffer.add_many(entries)

    def get_random_replay_buffer_entries(self, num_to_get):
        return self._hypothesis._replay_buffer.get_random(num_to_get)

    def get_all_replay_buffer_entries(self):
        return self._hypothesis._replay_buffer.get_all()

    # Convenient aliases
    train = send_train_message


class CoreToTrainCommsSync(object):  # TODO: common base, and ..update to non-Sync
    def __init__(self, hypothesis_accessor, hypothesis, process_comms):
        self._hypothesis_accessor = hypothesis_accessor
        self._hypothesis = hypothesis
        self._to_add_to_replay_cache = ReplayBuffer(non_permanent_maxlen=self._hypothesis._replay_buffer_size)

    def check_outgoing_queue(self):
        return

    def send_hypothesis(self, hypothesis):
        return

    def send_delete_message(self, hypothesis):
        return

    def send_kill_message(self):
        return

    def send_train_message(self, num_samples, id_start_frac, id_end_frac, num_times_to_train):
        self._hypothesis_accessor.try_train_pattern_filter(self._hypothesis, num_samples, id_start_frac, id_end_frac, num_times_to_train)

    def complete_and_register_replay_entry_to_send(self, replay_entry, reward_received, log_probs, selected_action, parent_hypothesis_comms):
        #TODO: de-dupe with original
        replay_entry.reward_received = torch.Tensor([reward_received])
        replay_entry.action_log_prob = torch.Tensor([log_probs])
        replay_entry.selected_action = torch.Tensor([selected_action])

        self.register_replay_entry_for_sending(replay_entry)

        if parent_hypothesis_comms is not None:
            parent_hypothesis_comms.register_replay_entry_for_sending(replay_entry)

    def wait_for_train_to_complete(self):
        return

    def register_replay_entry_for_sending(self, x):
        self._to_add_to_replay_cache.add(x)

    def send_replay_cache(self):
        self._hypothesis._replay_buffer.add_many(self._to_add_to_replay_cache)
        self._to_add_to_replay_cache.clear()

    # Convenient aliases
    train = send_train_message  # TODO: this is kind of a confusing name with train_pattern_filter
