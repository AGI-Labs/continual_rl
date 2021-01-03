import torch
import torch.nn as nn
import numpy as np
import time


class ReplayEntry(nn.Module):
    """
    Replay entries are owned per-hypothesis, as part of a replay buffer it serves to reinforce the boundary of that hypothesis.
    """
    def __init__(self, input_state, reward_received=None):
        super().__init__()
        # TODO: might be better to use *255 -> uint8
        self.input_state = input_state.detach().half()  # We don't cascade our changes up the pipeline. We make it half to minimize RAM usage
        self.reward_received = reward_received  # Populated after this entry "resolves" - gets discounted reward. (Except in clustering)
        self.action_log_prob = None
        self.selected_action = None
        #self.keep_value = np.random.uniform(0, 1.0)  # The buffer keeps its current "minimum required" to maintain the correct size

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


class ReplayBuffer(object):
    def __init__(self, device_for_quick_compute, preprocessing_net, buffer=None, non_permanent_maxlen=None):
        self._device_for_quick_compute = device_for_quick_compute

        self._non_permanent_maxlen = non_permanent_maxlen  # TODO: rename

        if buffer is not None:
            self._buffer = buffer
        else:
            self._buffer = []
            #self._buffer = deque(maxlen=self._non_permanent_maxlen)

        self._reduction_conv_net = preprocessing_net #ConvNet().to(self._device)  # The hypothesis net - TODO randomized or trained

    def clear(self):
        self._buffer.clear()

    def clone(self):
        # TODO: for many use cases, a full deep clone is probably not necessary... (just need a new storage unit, not separate entries)
        entries = [entry.clone() for entry in self._buffer]
        cloned_buffer = ReplayBuffer(self._device_for_quick_compute, preprocessing_net=self._reduction_conv_net,
                                     buffer=entries, non_permanent_maxlen=self._non_permanent_maxlen)
        return cloned_buffer

    def _get_closest_replay_entries(self, directory, num_to_get):  # TODO unit test
        dir_size = len(directory)
        dir_size = min(dir_size, self._non_permanent_maxlen)//2  # If the directory has grown too large, we can blow up our memory... TODO...if the dir_size is less than num_to_get, we'll just...remove the first n basically
        input_states = torch.stack([entry.input_state for entry in directory[:dir_size]])
        difference = input_states.unsqueeze(1).cpu() - input_states.cpu()  # Taking advantage of the auto-dim matching thing. Moving onto CPU because otherwise this can get huge
        square_distances = torch.flatten(difference ** 2, start_dim=2).sum(dim=-1).cpu().numpy()

        assert len(square_distances.shape) == 2, "Failed to successfully compute the distances. Double check logic"

        indices = np.argsort(square_distances, axis=None)
        indices_x = indices // dir_size
        indices_y = indices % dir_size

        # TODO: just truncating the dir_size because those are (probably) the diagonal. TODO: checking is slow, but still...

        # Returns two lists of indices. indices_x[0] pairs with indices_y[0]. TODO: make clearer
        return indices_x[dir_size:dir_size+num_to_get], indices_y[dir_size:dir_size+num_to_get]

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

    def _compact_random_value_threshold(self, buffer, max_count):
        buffer.sort(key=lambda entry: entry.keep_value, reverse=True)
        buffer = buffer[:max_count]
        return buffer

    def _compact_random_value_threshold_split_new_and_old(self):
        max_new_count = self._non_permanent_maxlen//2
        max_old_count = self._non_permanent_maxlen - max_new_count

        num_old = len(self._buffer) - max_new_count

        # Split the buffer into "old" and "new".
        # We'll compact the "old" and leave the "new" as-is.
        old_buffer = self._buffer[:num_old]
        new_buffer = self._buffer[num_old:]

        old_buffer = self._compact_random_value_threshold(old_buffer, max_old_count)

        self._buffer = old_buffer
        self._buffer.extend(new_buffer)

    def _compact_random_split_new_and_old(self):
        max_num_new = int(self._non_permanent_maxlen * .5)
        max_num_old = self._non_permanent_maxlen - max_num_new

        # Get old entries starting at 0, and ending at the place where we start getting *all* entries.
        fraction_new = max_num_new/len(self._buffer)
        new_entries_buffer = self._buffer[-max_num_new:]
        old_buffer = self.get(max_num_old, 0, fraction_new)

        self._buffer = new_entries_buffer
        self._buffer.extend(old_buffer)

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
