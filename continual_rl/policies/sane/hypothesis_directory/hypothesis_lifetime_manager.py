import torch
import torch.multiprocessing as multiprocessing
import copy
from continual_rl.policies.sane.hypothesis.hypothesis import Hypothesis
from continual_rl.policies.sane.hypothesis.core_accessor import CoreAccessor
from continual_rl.policies.sane.hypothesis.train_accessor import TrainAccessor
from continual_rl.policies.sane.hypothesis.long_term_gate_hypothesis import LongTermGateHypothesis
from continual_rl.policies.sane.process_comms.core_to_train_comms import CoreToTrainComms, CoreToTrainCommsSync
from continual_rl.policies.sane.process_comms.process_comms import ProcessComms
from continual_rl.policies.sane.process_comms.train_process import TrainProcess
from continual_rl.policies.sane.hypothesis_directory.utils import Utils
from continual_rl.utils.utils import Utils as CommonUtils


class HypothesisLifetimeManager(object):
    """
    This class manages the creation and deletion of hypotheses. There are multiple processes that run hypothesis
    training, each of which manages multiple hypotheses. This class sets that up.
    Since it owns the creation and deletion of hypotheses, it also owns the communication channel to them.
    """

    def __init__(self, data):
        self._data = data
        self.hypothesis_accessor = CoreAccessor
        self._hypothesis_comms = {}  # A map of hypothesis to the core-to-train comms used to talk to it.
        self._available_comms = []  # The set of comms available to be used (each one corresponds to a separate process, but each process can handle multiple hypotheses)
        self._next_comm_id = 0

    @property
    def logger(self):
        logger = Utils.create_logger(f"{self._data._output_dir}/core_process.log")
        return logger

    def get_hypothesis_from_id(self, id):
        # TODO: de-dupe with core_accessor's version
        selected_hypothesis = self._data.get_hypothesis_from_id(id)

        if selected_hypothesis is None:
            self.logger.warning(f"get_hypothesis_from_id: Attempted to find hypothesis {id} that no longer exists")

        # May return None if the hypothesis was not found.
        return selected_hypothesis

    def get_comms(self, hypothesis):
        return self._hypothesis_comms[hypothesis]

    def _process_creation_buffer(self, creation_buffer):
        if len(creation_buffer) > 0:
            creation_data_to_remove = []

            for hypothesis_to_duplicate_id, all_creation_data in creation_buffer.items():
                for creation_data_id, creation_data in all_creation_data.items():
                    self._process_creation_buffer_entry(hypothesis_to_duplicate_id, creation_data)
                    creation_data_to_remove.append((hypothesis_to_duplicate_id, creation_data_id))

            # TODO: at this point could probably just .clear()
            for hypothesis_to_duplicate_id, creation_data_id in creation_data_to_remove:
                del creation_buffer[hypothesis_to_duplicate_id][creation_data_id]
                if len(creation_buffer[hypothesis_to_duplicate_id]) == 0:
                    del creation_buffer[hypothesis_to_duplicate_id]

    def _duplicate_hypothesis(self, hypothesis_to_duplicate, parent, pattern_filter_source,
                              random_policy, keep_non_decayed=False):  # TODO: let each hypothesis handle this for itself?
        # Gates don't have policies, so grab it from the prototype
        if hypothesis_to_duplicate.is_long_term:
            policy = hypothesis_to_duplicate._policy
            self.logger.info(f"Creating new hypothesis from LONG TERM {hypothesis_to_duplicate.friendly_name}.")
        else:
            policy = hypothesis_to_duplicate._policy
            self.logger.info(f"Creating new hypothesis from SHORT TERM {hypothesis_to_duplicate.friendly_name}.")

        policy = copy.deepcopy(policy)

        # In some cases, we want the prototype's pattern filter (if ST < prototype)
        # Other times we want the gate's pattern filter (e.g. if promoting a gate). Thus it must get passed in
        # (specified when added to the creation buffer)
        pattern_filter = copy.deepcopy(pattern_filter_source.pattern_filter.state_dict())  # TODO: deep copy still necessary? which is more efficient?

        entries = self.get_comms(pattern_filter_source).get_all_replay_buffer_entries() if self._data._duplicate_uses_replay else []

        if random_policy and policy is not None:  # TODO: not true random
            noise = 2 * torch.rand(policy.shape) - 1  # Noise between -1 and 1
            policy.data = policy.data + 0.1 * noise

        new_hypothesis = self._create_hypothesis(parent, pattern_filter_state_dict=pattern_filter, policy=policy,
                                                 replay_entries=entries)

        new_hypothesis.non_decayed_usage_count = hypothesis_to_duplicate.non_decayed_usage_count if keep_non_decayed else 0
        new_hypothesis.usage_count = 0  # Must get used more before it gets duplicated again

        if new_hypothesis.is_long_term:
            new_hypothesis.prototype.usage_count = hypothesis_to_duplicate.usage_count  # TODO: ??
            new_hypothesis.prototype.non_decayed_usage_count = hypothesis_to_duplicate.non_decayed_usage_count
            self.logger.info(
                f"Hypothesis {new_hypothesis.friendly_name} with prototype {new_hypothesis.prototype.friendly_name} has prototype usage count: {new_hypothesis.prototype.usage_count}")

        hypothesis_to_duplicate.usage_count = 0  # To prevent infinite creation against _usage_count_min_to_convert_to_long_term

        parent_id = parent.friendly_name if parent is not None else None
        self.logger.info(f"Created hypothesis {new_hypothesis.friendly_name} with policy {new_hypothesis.policy}. "
                              f"and parent {parent_id} Made random? {random_policy} Usage count: {new_hypothesis.usage_count} "
                              f"Non-decayed: {new_hypothesis.non_decayed_usage_count}")
        return new_hypothesis

    def _process_creation_buffer_entry(self, hypothesis_to_duplicate_id, creation_data):
        self.logger.info(f"Choosing to create {hypothesis_to_duplicate_id} with count {creation_data['count']}")

        parent_id = creation_data["parent_id"]
        pattern_filter_source_id = creation_data["pattern_filter_source_id"]
        random_policy = creation_data["random_policy"]
        count = creation_data["count"]

        if count > 0:
            if hypothesis_to_duplicate_id is not None:
                hypothesis_to_duplicate = self.get_hypothesis_from_id(hypothesis_to_duplicate_id)
                parent_hypothesis = self.get_hypothesis_from_id(parent_id) if parent_id is not None else None
                pattern_filter_source = self.get_hypothesis_from_id(pattern_filter_source_id) if pattern_filter_source_id is not None else None

                if hypothesis_to_duplicate is not None:
                    self.logger.info(f"Duplicating {hypothesis_to_duplicate.friendly_name} and adding to parent "
                                          f"{parent_hypothesis.friendly_name if parent_hypothesis is not None else None}")
                    self._duplicate_hypothesis(hypothesis_to_duplicate, parent=parent_hypothesis,
                                               random_policy=random_policy, pattern_filter_source=pattern_filter_source)
                else:
                    self.logger.warning(
                        f"_process_creation_buffer: Attempted to find hypothesis {hypothesis_to_duplicate_id} that no longer exists")
            else:
                self.logger.info(f"Creating new long-term entry")
                self._create_hypothesis(parent=None)

    def _get_or_create_process_comms(self):  # TODO: move out of directory..?
        # TODO: this is not very synchronous-friendly
        num_devices = torch.cuda.device_count()  # Cuda devices
        comm_id = self._next_comm_id % self._data._max_processes

        if len(self._available_comms) < self._data._max_processes:
            self.logger.info("Creating new process comms")
            cuda_id = self._next_comm_id % max(1, num_devices)
            device_id = torch.device( "cpu" if num_devices == 0 or not self._data._use_cuda
                                      else f"cuda:{cuda_id}")
            process_comms = ProcessComms(device_id)
            self._available_comms.append(process_comms)

            train_process = TrainProcess(hypothesis_accessor=TrainAccessor, output_dir=self._data._output_dir)

            new_process = multiprocessing.Process(target=train_process.try_process_queue, args=(process_comms,))  # The hypothesis is automatically passed as an arg ("self")
            new_process.daemon = True
            new_process.start()

            process_comms.process = new_process  # TODO: needed?
        else:
            self.logger.info("Using existing process comms")
            # TODO: proper load balancing
            process_comms = self._available_comms[comm_id]

        self._next_comm_id += 1

        return process_comms

    def _send_hypothesis_to_process(self, process_comms, hypothesis, replay_entries):
        # Ensure the parameters get shared before sending
        self.logger.info(f"Sharing memory for hypothesis {hypothesis.friendly_name}")
        successful_share = False
        failed_count = 0
        
        while not successful_share:
            try:
                assert failed_count < 10, "Failing out if we've tried 10 times"
                self.logger.info("Attempting to share memory")
                hypothesis.share_memory()  # - I think (?) not necessary because of my share_parameters_memory...
                hypothesis.share_parameters_memory()  # share_memory only captures children, not all parameters, so share those here
                successful_share = True
            except RuntimeError as e:
                assert "unable to open shared memory object" in str(e)
                failed_count += 1
                self.logger.info(f"(Attempt {failed_count}) Unable to open shared memory object, trying again...")

        # Get the wrapper that will handle sending data over to the hypothesis training processes
        if self._data._is_sync:
            new_hypothesis_comms = CoreToTrainCommsSync(TrainAccessor, hypothesis, process_comms)
        else:
            new_hypothesis_comms = CoreToTrainComms(self.hypothesis_accessor, hypothesis, process_comms)

        # Send the hypothesis over to the process that will process its requests
        self.logger.info(f"Sending hypothesis {hypothesis.friendly_name} to process {process_comms.friendly_name}")
        new_hypothesis_comms.send_hypothesis(hypothesis)

        self._hypothesis_comms[hypothesis] = new_hypothesis_comms

        # TODO: sending the replay entries over separately because sending it over in the initial hypothesis causes too many open files issues?
        # And even without that, the other way seems way slower...
        if replay_entries is not None:
            self.logger.info(f"Adding {len(replay_entries)} entries to {hypothesis.friendly_name}")
            self.get_comms(hypothesis).add_many_to_replay(replay_entries)

    def _create_hypothesis(self, parent, pattern_filter_state_dict=None, policy=None, replay_entries=None):
        """
        Create a blank hypothesis and add it to the correct directory. Directories are expected just to be lists.
        """
        layer_id = parent.layer_id + 1 if parent is not None else 0

        self.logger.info(f"Getting or creating new process comms")
        process_comms = self._get_or_create_process_comms()
        self.logger.info(f"Creating hypothesis on process {process_comms.friendly_name} device {process_comms.device_id}")
        new_hypothesis = Hypothesis(config=self._data._config,
                                    device=process_comms.device_id, master_device=self._data._master_device_id,
                                    output_dir=self._data._output_dir,
                                    input_space=self._data._obs_space, output_size=self._data._action_size,
                                    replay_buffer_size=self._data._replay_buffer_size,
                                    filter_learning_rate=self._data._filter_learning_rate,
                                    pattern_filter=None, policy=policy,
                                    layer_id=layer_id, parent_hypothesis=parent)

        #self.logger.info(f"SANE Hypothesis: filter num parameters: {CommonUtils.count_trainable_parameters(new_hypothesis.pattern_filter)}")

        self.logger.info(f"Loading state dict")
        if pattern_filter_state_dict is not None:
            self.hypothesis_accessor.load_pattern_filter_from_state_dict(new_hypothesis, pattern_filter_state_dict)
            new_hypothesis.usage_count_since_creation = 0
        self.logger.info(f"Created hypothesis {new_hypothesis.friendly_name} that will be on process {process_comms.friendly_name}")

        # Do this before sending to the process, so we don't have to do it as a second step
        #if replay_entries is not None:
        #    self.logger.info(f"Adding entries to {new_hypothesis.friendly_name}")
        #    new_hypothesis._replay_buffer.add_many(replay_entries)

        # new_hypothesis = new_hypothesis.to(process_comms.device_id) - TODO: technically made unnecessary by the to() in load_pattern_filter_from_state_dict (including in the __init__)
        new_hypothesis._policy.data = new_hypothesis._policy.data.to(self._data._master_device_id)  # TODO: hackily just putting all policies on the same gpu (for the policy update)

        self.logger.info(f"Hypothesis {new_hypothesis.friendly_name} moved to device {process_comms.device_id} and master {self._data._master_device_id}")

        # Send the hypothesis over to the process that will process its requests
        self._send_hypothesis_to_process(process_comms, new_hypothesis, replay_entries)

        # If the hypothesis is long-term, it still has a pattern filter, but duplication happens from the prototype (copy of the source), and a few other changes, so wrap it
        make_long_term = layer_id < len(self._data._max_hypotheses_per_layer) - 1  # Last layer is short-term, all above it are gates
        if make_long_term:
            new_hypothesis = LongTermGateHypothesis(source_hypothesis=new_hypothesis, parent_hypothesis=parent)
            self._send_hypothesis_to_process(process_comms, new_hypothesis, replay_entries)

            if parent is None:
                assert layer_id == 0, "None parent only allowed at layer 0"
                self._data._long_term_directory.append(new_hypothesis)

        self.logger.info(f"Hypothesis {new_hypothesis.friendly_name} finished initializing")

        return new_hypothesis

    def delete_hypothesis(self, hypothesis, kill_process):
        self.logger.info(f"Deleting {hypothesis.friendly_name}. Kill process? {kill_process}")

        if hypothesis.parent_hypothesis is not None:
            # This hypothesis is a regular short term (or intermediate gate) ... TODO: this is not a great way to check "am I a prototype"
            hypothesis.parent_hypothesis.remove_short_term(hypothesis)
        elif hypothesis in self._data._long_term_directory:  # TODO: slow?
            self._data._long_term_directory.remove(hypothesis)
        # TODO: prototypes have no way of accessing their parent, so they can't be auto-removed... but why was having prototype have the parent causing errors in sending to proc?

        if kill_process:  # Kill process name is now a bit out of date
            assert not hypothesis.is_long_term or len(hypothesis.short_term_versions) == 0, "Attempting to delete a gate that still has children."

            self.get_comms(hypothesis).send_delete_message(hypothesis)
            self.logger.info("Deleting hypothesis comms")
            del self._hypothesis_comms[hypothesis]

            self.logger.info("Deleting hypothesis")  # Debugging infrequent SEGFAULTS -is it the hypothesis or the comm?
            del hypothesis  # The core process seems to be continuing to keep some file descriptors open... TODO? (seems not to help, keeping for now).

    def shutdown(self):
        self.logger.info("Shutting down processes.")
        for hypothesis, comms in self._hypothesis_comms.items():
            comms.send_delete_message(hypothesis)  # TODO: hanging sometimes?
            comms.send_kill_message()  # TODO: will kill before all hypotheses are deleted...I think this is okay though

        for comms in self._available_comms:
            comms.close()
