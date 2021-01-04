import torch
import torch.nn as nn
import numpy as np
import time
from continual_rl.utils.utils import Utils


class ReplayEntry(nn.Module):
    """
    Replay entries are owned per-hypothesis, as part of a replay buffer it serves to reinforce the boundary of that hypothesis.
    """
    def __init__(self, input_state, reward_received=None):
        super().__init__()
        # TODO: might be better to use *255 -> uint8. Inputs are usually uint8 by default anyway...
        self.input_state = input_state.detach().half()  # We don't cascade our changes up the pipeline. We make it half to minimize RAM usage
        self.reward_received = reward_received  # Populated after this entry "resolves" - gets discounted reward. (Except in clustering)
        self.action_log_prob = None
        self.selected_action = None

    def clone(self):
        # TODO: not currently cloning the random number? since it's random anyway...? TODO: is there a case where it should really be kept
        cloned_entry = ReplayEntry(self.input_state.clone())

        if self.reward_received is not None:
            cloned_entry.reward_received = self.reward_received.clone()

        if self.action_log_prob is not None:
            cloned_entry.action_log_prob = self.action_log_prob.clone()

        if self.selected_action is not None:
            cloned_entry.selected_action = self.selected_action.clone()

        return cloned_entry


class ReplayBufferFileBacked(object):
    def __init__(self, maxlen, observation_space, large_file_path):
        self._maxlen = maxlen
        self._buffers, self._file_paths = self._construct_buffers(maxlen, observation_space, large_file_path)

    def _construct_buffers(self, maxlen, observation_space, large_file_path):
        buffers = {}
        file_paths = []

        # Store observations
        observation_buffers, file_path = Utils.create_file_backed_tensor(large_file_path,
                                                                         shape=(maxlen, *observation_space.shape),
                                                                         dtype=observation_space.dtype.type)
        buffers["input_state"] = observation_buffers
        file_paths.append(file_path)

        # Store rewards received:
        reward_received_buffers, file_path = Utils.create_file_backed_tensor(large_file_path,
                                                                             shape=(maxlen, 1),
                                                                             dtype=torch.float32)
        buffers["reward_received"] = reward_received_buffers
        file_paths.append(file_path)

        # Store action_log_probs:
        action_log_prob_buffers, file_path = Utils.create_file_backed_tensor(large_file_path,
                                                                             shape=(maxlen, 1),
                                                                             dtype=torch.float32)
        buffers["action_log_prob"] = action_log_prob_buffers
        file_paths.append(file_path)

        # Store selected action:  (TODO: support non-discrete)
        selected_action_buffers, file_path = Utils.create_file_backed_tensor(large_file_path,
                                                                             shape=(maxlen, 1),
                                                                             dtype=torch.uint8)
        buffers["selected_action"] = selected_action_buffers
        file_paths.append(file_path)

        # Store whether this entry has been set:
        been_set_buffers, file_path = Utils.create_file_backed_tensor(large_file_path,
                                                                      shape=(maxlen, 1),
                                                                      dtype=torch.bool)
        buffers["been_set"] = been_set_buffers
        file_paths.append(file_path)

        return buffers, file_paths

    def _get_last_unset(self):
        unset_indices = torch.where(self._buffers["been_sets"])
        return unset_indices

    def add_many(self, entries):
        unset_indices = self._get_last_unset()

        for entry in entries:
            index_to_populate = unset_indices.pop(0) if len(unset_indices) > 0 else np.random.uniform(self._maxlen)

            for replay_key, replay_value in entry.__dict__.items():
                self._buffers[replay_key][index_to_populate].copy_(replay_value)
                self._buffers["been_set"][index_to_populate] = True
        pass


class ReplayBuffer(object):
    def __init__(self, buffer=None, non_permanent_maxlen=None):
        self._non_permanent_maxlen = non_permanent_maxlen  # TODO: rename

        if buffer is not None:
            self._buffer = buffer
        else:
            self._buffer = []
            #self._buffer = deque(maxlen=self._non_permanent_maxlen)

    def clear(self):
        self._buffer.clear()

    def clone(self):
        # TODO: for many use cases, a full deep clone is probably not necessary... (just need a new storage unit, not separate entries)
        entries = [entry.clone() for entry in self._buffer]
        cloned_buffer = ReplayBuffer(buffer=entries, non_permanent_maxlen=self._non_permanent_maxlen)
        return cloned_buffer

    def add(self, x):
        if isinstance(x, ReplayEntry):
            replay_entry = x
        else:
            assert False, "Deprecated path"
            replay_entry = ReplayEntry(x)

        self._buffer.append(replay_entry)

        self.try_compact()

        return replay_entry

    def add_many(self, entries):
        self._buffer.extend(entries)
        self.try_compact()

    def _compact_random(self):
        with torch.no_grad():
            if len(self._buffer) > self._non_permanent_maxlen:
                self._buffer = self.get(self._non_permanent_maxlen, 0, 1)

    def compact(self):
        with torch.no_grad():
            if len(self._buffer) > self._non_permanent_maxlen:
                self._compact_random()

    def try_compact(self):
        error = None
        for _ in range(10):
            try:
                self.compact()
                return  # If we succeed, bail out
            except RuntimeError as e:  # TODO: check explicitly for CUDA issue
                print("Compacting failed with error: {}".format(e))
                error = e
                time.sleep(0.1)

        # If we tried n times and got an error each time, raise it
        raise error

    def get(self, num_non_permanent_to_get, id_start_frac, id_end_frac):

        id_min = int(id_start_frac * len(self))
        id_max = int(id_end_frac * len(self))

        entries = []

        indices = np.array(list(range(id_min, id_max)))
        np.random.shuffle(indices)
        indices_subset = indices[:num_non_permanent_to_get]

        for index in indices_subset:  # TODO: more efficiently
            entries.append(self._buffer[index])

        return entries

    @classmethod
    def prepare_for_bulk_transfer(cls, entries):  # TODO: unit test
        input_states = []
        rewards_received = []
        action_log_probs = []
        selected_actions = []

        # I think this only needs to happen during inflate. Let's find out... (TODO)
        #entries = [entry.clone() for entry in entries]

        for entry in entries:
            input_states.append(entry.input_state)
            rewards_received.append(entry.reward_received)
            action_log_probs.append(entry.action_log_prob)
            selected_actions.append(entry.selected_action)

        if len(input_states) > 0:
            input_states = torch.stack(input_states)
            rewards_received = torch.stack(rewards_received)
            action_log_probs = torch.stack(action_log_probs)
            selected_actions = torch.stack(selected_actions)

        return input_states, rewards_received, action_log_probs, selected_actions

    @classmethod
    def inflate_from_bulk_transfer(cls, bulk_tensor_obj):  # TODO: unit test
        input_states, rewards_received, action_log_probs, selected_actions = bulk_tensor_obj
        entries = []

        for entry_id, input_state in enumerate(input_states):
            entry = ReplayEntry(input_state)
            entry.reward_received = rewards_received[entry_id]
            entry.action_log_prob = action_log_probs[entry_id]
            entry.selected_action = selected_actions[entry_id]

            entries.append(entry.clone())   # TODO: testing clone to see if it fixes my experiments dying (e.g. from SIGABRT)

        del bulk_tensor_obj

        return entries

    @property
    def maxlen(self):
        return self._non_permanent_maxlen

    def __iter__(self):
        for elem in self._buffer:
            yield elem

    def __len__(self):
        return len(self._buffer)

    def __getitem__(self, item):
        return self._buffer[item]

    def __setitem__(self, key, item):  # TODO: unit test me
        self._buffer[key] = item

    def __delitem__(self, key):
        del self._buffer[key]


class ReplayBufferDataLoaderWrapper(object):
    """
    Wraps a replay buffer such that it can be used by a DataLoader
    """
    def __init__(self, replay_buffer):
        self._replay_buffer = replay_buffer

    def __len__(self):
        return len(self._replay_buffer)

    def __getitem__(self, item):
        # TODO: is it ever not just an int? (e.g. a slice?)
        if isinstance(item, int):
            selected_items = [self._replay_buffer[item]]
        else:
            selected_items = self._replay_buffer[item]
        data_tensors = self._replay_buffer.prepare_for_bulk_transfer(selected_items)
        return data_tensors
