import torch
from continual_rl.policies.sane.hypothesis.replay_buffer import ReplayBuffer
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
        self._to_add_to_replay_cache = ReplayBuffer(non_permanent_maxlen=self._hypothesis._replay_buffer_size,
                                                  device_for_quick_compute=self._hypothesis._device, preprocessing_net=self._hypothesis._replay_buffer._reduction_conv_net)

        # Used for grabbing negative examples, so this is sort of best-effort. (Also used for getting the length of the replay buffer.)
        self._cached_replay_buffer = ReplayBuffer(non_permanent_maxlen=self._hypothesis._replay_buffer_size,
                                                  device_for_quick_compute=self._hypothesis._device, preprocessing_net=self._hypothesis._replay_buffer._reduction_conv_net)  # TODO: hacky
        
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

    def send_hypothesis(self, hypothesis):
        self.logger.info(f"Sending new hypothesis over: {hypothesis.friendly_name}")
        self.send_task_and_await_result("add_hypothesis", hypothesis)
        self.logger.info(f"Hypothesis successfully sent: {hypothesis.friendly_name}")

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

        # TODO: currently awaiting result here, so multiple hypotheses aren't being trained at the same time. This is for debug, and kinda defeats some of the purpose of this multiprocessing thing
        #self.send_task_and_await_result("train", object_to_send)  # args, kwargs

    def send_clear_replay_message(self):
        self.send_task_and_await_result("clear_replay", {})

    def send_add_many_to_replay_message(self, x):
        """
        Compress our entries into a tensor. The primary reason for this is because OSes seem to have issues sometimes with how many objects are getting transferred without this.
        """
        #num_to_send = 1024 #len(x) #256
        batch_size = 500  #num_to_send  #256  # TODO: this should not be necessary....ideally remove

        if len(x) > 0:
            # Add the entries to our local buffer
            self.logger.info(f"Adding many: {len(x)}")
            #self._cached_replay_buffer.add_many(x)

            self.logger.info("Preparing to send in batches")
            # TODO: is it faster to put it on the gpu for sending? (This is what I'm currently doing)
            self.logger.info(f"Sending {len(x)} replay entries to {self._hypothesis.friendly_name}")
            entries = x

            for bulk_subset_index in range(0, len(entries), batch_size):  # TODO: ...this shouldn't be necessary if the queue is getting properly cleared
                entries_subset = entries[bulk_subset_index:bulk_subset_index+batch_size]
                bulk_tensor_obj = ReplayBuffer.prepare_for_bulk_transfer(entries_subset)
                #bulk_tensor_obj = bulk_tensor_obj.to(self._hypothesis._device)  # Move it to the gpu because I think (?TODO) this process is faster... TODO not doing it?

                # Fire and forget version -- if the tensor gets garbage collected before this completes, things break, so add it to a list to make sure it stays around.
                # It's important that the get be consistently pumped, otherwise this put might hang
                request_id = uuid.uuid4()
                self.process_comms.incoming_queue.put(self._construct_packet("add_many_to_replay", request_id, bulk_tensor_obj, response_requested=False))  # response_requested: False
                
                self._tensors_in_flight.append(bulk_tensor_obj)

                #self.send_task_and_await_result("add_many_to_replay", bulk_tensor_obj)

            self.logger.info("All batches sent")

    def send_add_many_to_negative_examples_message(self, x):
        if len(x) > 0:
            #entries = [entry.clone() for entry in x]
            bulk_tensor_obj = ReplayBuffer.prepare_for_bulk_transfer(x)

            # Fire and forget version
            request_id = uuid.uuid4()
            self.process_comms.incoming_queue.put(self._construct_packet("add_many_to_negative_examples", request_id, bulk_tensor_obj, response_requested=False))  # response_requested: False

            #self.send_task_and_await_result("add_many_to_negative_examples", bulk_tensor_obj)

    def send_get_replay_buffer_length_message(self):
        #return self._cached_replay_buffer._non_permanent_maxlen  # TODO: obviously not accurate
        #return len(self._cached_replay_buffer)  # TODO: how accurate? (with threading) Currently doing this because it's bottlenecking the training - currently def not accurate, constantly setting to 0
        replay_buffer_length = self.send_task_and_await_result("get_replay_buffer_length", None)
        return replay_buffer_length

    def send_get_random_replay_buffer_entries_message(self, *args, **kwargs):
        """
        Expected args: the signature of replay_buffer.get
        """
        # TODO: currently just using the local version for speed
        #print("Using local get_random_replay cache")
        #replay_entries = self._cached_replay_buffer.get(*args, **kwargs)

        # TODO: send over in batches? (Could have train_process say how many messages it's sending for a given request_id)
        # This is just quick and dirty because very occasionally it hangs on large requests. I think ideally it shouldn't hang at all though
        self.logger.info("Get random replay entries starting")
        bulk_tensor_obj = self.send_task_and_await_result("get_random_replay_entries", (args, kwargs), timeout=300)
        self.logger.info("Get random replay entries complete")

        if bulk_tensor_obj is not None:
            replay_entries = ReplayBuffer.inflate_from_bulk_transfer(bulk_tensor_obj)
        else:
            self.logger.error("Just dropped a get_random on the ground. This is bad.")
            replay_entries = []

        return replay_entries

    def send_get_all_replay_buffer_entries_message(self):
        #print("Using local get_all_replay cache")
        #replay_entries = self._cached_replay_buffer._buffer  # TODO: hacky, and local for speed. Clean all this up

        self.logger.info("Get all replay entries starting")
        bulk_tensor_obj = self.send_task_and_await_result("get_all_replay_entries", None, timeout=300)
        self.logger.info("Get all replay entries complete")

        if bulk_tensor_obj is not None:
            replay_entries = ReplayBuffer.inflate_from_bulk_transfer(bulk_tensor_obj)
            #replay_entries = [entry.clone() for entry in replay_entries]
            del bulk_tensor_obj  # TODO: clean this stuff up
        else:
            self.logger.error("Just dropped a get_all_replay on the ground. This is bad.")
            replay_entries = []

        return replay_entries

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
        self._to_add_to_replay_cache.add(x)  # TODO: later entries might overwrite earlier in such a way that earlier never actually get seen by the pattern_filter

    def send_replay_cache(self):
        if len(self._to_add_to_replay_cache) > 0:
            self.send_add_many_to_replay_message(self._to_add_to_replay_cache)
            self._to_add_to_replay_cache.clear()  # TODO:...this might actually be harmful?
            #self._cached_replay_buffer._buffer.clear()  # TODO: thus *only* ever sending the top k per the priority replay buffer

    # Convenient aliases
    clear_replay = send_clear_replay_message
    get_replay_buffer_length = send_get_replay_buffer_length_message
    get_random_replay_buffer_entries = send_get_random_replay_buffer_entries_message
    get_all_replay_buffer_entries = send_get_all_replay_buffer_entries_message
    add_many_to_replay = send_add_many_to_replay_message
    add_many_to_negative_examples = send_add_many_to_negative_examples_message
    train = send_train_message  # TODO: this is kind of a confusing name with train_pattern_filter


class CoreToTrainCommsSync(object):  # TODO: common base
    def __init__(self, hypothesis_accessor, hypothesis, process_comms):
        self._hypothesis_accessor = hypothesis_accessor
        self._hypothesis = hypothesis
        self._to_add_to_replay_cache = ReplayBuffer(non_permanent_maxlen=self._hypothesis._replay_buffer_size,
                                                    device_for_quick_compute=self._hypothesis._device,
                                                    preprocessing_net=self._hypothesis._replay_buffer._reduction_conv_net)


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

    def send_add_many_to_replay_message(self, x):
        """
        Compress our entries into a tensor. The primary reason for this is because OSes seem to have issues sometimes with how many objects are getting transferred without this.
        """
        self._hypothesis._replay_buffer.add_many(x)

    def send_add_many_to_negative_examples_message(self, x):
        # TODO: de-dupe with train_process
        self._hypothesis._negative_examples.add_many(x)

    def send_get_replay_buffer_length_message(self):
        return len(self._hypothesis._replay_buffer)

    def send_get_random_replay_buffer_entries_message(self, *args, **kwargs):
        """
        Expected args: the signature of replay_buffer.get(num_non_permanent_to_get, id_min, id_max)
        """
        replay_entries = self._hypothesis._replay_buffer.get(*args, **kwargs)
        return replay_entries

    def send_get_all_replay_buffer_entries_message(self):
        # TODO: remove non-permanent
        return self._hypothesis._replay_buffer._buffer

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
        if len(self._to_add_to_replay_cache) > 0:
            self.send_add_many_to_replay_message(self._to_add_to_replay_cache)
            self._to_add_to_replay_cache.clear()

    # Convenient aliases
    get_replay_buffer_length = send_get_replay_buffer_length_message
    get_random_replay_buffer_entries = send_get_random_replay_buffer_entries_message
    get_all_replay_buffer_entries = send_get_all_replay_buffer_entries_message
    add_many_to_replay = send_add_many_to_replay_message
    add_many_to_negative_examples = send_add_many_to_negative_examples_message
    train = send_train_message  # TODO: this is kind of a confusing name with train_pattern_filter
