import numpy as np
import torch
import tempfile
import threading
from torch.nn import functional as F
import queue
from continual_rl.policies.impala.torchbeast.monobeast import Monobeast, Buffers


class ClearMonobeast(Monobeast):
    """
    An implementation of Experience Replay for Continual Learning (Rolnick et al, 2019):
    https://arxiv.org/pdf/1811.11682.pdf
    """
    def __init__(self, model_flags, observation_space, action_space, policy_class):
        super().__init__(model_flags, observation_space, action_space, policy_class)

        # LSTMs not supported largely because they have not been validated; nothing extra is stored for them.
        assert not model_flags.use_lstm, "CLEAR does not presently support using LSTMs."

        self._model_flags = model_flags
        self._entries_per_buffer = int(model_flags.replay_buffer_frames // (model_flags.unroll_length * model_flags.num_actors))
        self._replay_buffers, self._temp_files = self._create_replay_buffers(model_flags, observation_space.shape,
                                                                             action_space.n, self._entries_per_buffer)
        self._replay_lock = threading.Lock()

        # Each replay buffer needs to also have cloning losses applied to it
        # Keep track of them as they're generated, to ensure we apply losses to all. This doesn't currently
        # guarantee order - i.e. one learner thread might get one replay batch for training and a different for cloning
        self._replay_batches_for_loss = queue.Queue()

    def _create_file_backed_tensor(self, file_path, shape, dtype):
        temp_file = tempfile.NamedTemporaryFile(dir=file_path)

        size = 1
        for dim in shape:
            size *= dim

        storage_type = None
        tensor_type = None
        if dtype == torch.uint8:
            storage_type = torch.ByteStorage
            tensor_type = torch.ByteTensor
        elif dtype == torch.int32:
            storage_type = torch.IntStorage
            tensor_type = torch.IntTensor
        elif dtype == torch.int64:
            storage_type = torch.LongStorage
            tensor_type = torch.LongTensor
        elif dtype == torch.bool:
            storage_type = torch.BoolStorage
            tensor_type = torch.BoolTensor
        elif dtype == torch.float32:
            storage_type = torch.FloatStorage
            tensor_type = torch.FloatTensor

        shared_file_storage = storage_type.from_file(temp_file.name, shared=True, size=size)
        new_tensor = tensor_type(shared_file_storage).view(shape)

        return new_tensor, temp_file

    def _create_replay_buffers(self, model_flags, obs_shape, num_actions, entries_per_buffer):
        """
        Key differences from normal buffers:
        1. File-backed, so we can store more at a time
        2. Structured so that there are num_actors buffers, each with entries_per_buffer entries

        Each buffer entry has unroll_length size, so the number of frames stored is (roughly, because of integer
        rounding): num_actors * entries_per_buffer * unroll_length
        """
        # Get the standard specs, and also add the CLEAR-specific reservoir value
        specs = self.create_buffer_specs(model_flags.unroll_length, obs_shape, num_actions)
        specs["reservoir_val"] = dict(size=(1,), dtype=torch.float32)  # Note: one reservoir value per row
        buffers: Buffers = {key: [] for key in specs}

        # Hold on to the file handle so it does not get deleted. Technically optional, as at least linux will
        # keep the file open even after deletion, but this way it is still visible in the location it was created
        temp_files = []

        for _ in range(model_flags.num_actors):
            for key in buffers:
                shape = (entries_per_buffer, *specs[key]["size"])
                new_tensor, temp_file = self._create_file_backed_tensor(model_flags.large_file_path, shape,
                                                                        specs[key]["dtype"])
                new_tensor.zero_()  # Ensure our new tensor is zero'd out
                buffers[key].append(new_tensor.share_memory_())
                temp_files.append(temp_file)

        return buffers, temp_files

    def _get_replay_buffer_filled_indices(self, replay_buffers, actor_index):
        """
        Get the indices in the replay buffer corresponding to the actor_index. If actor_index is None, get the
        index as though the actors were all concatenated together. (This is so we can sample across all of them.)
        """
        # We know that the reservoir value > 0 if it's been filled, so check for entries where it == 0
        if actor_index is not None:
            buffer_indicator = replay_buffers['reservoir_val'][actor_index].squeeze(1)
        else:
            buffer_indicator = torch.cat(replay_buffers['reservoir_val']).squeeze(1)

        replay_indices = np.where(buffer_indicator != 0)[0]
        return replay_indices

    def _get_actor_unfilled_indices(self, actor_index, entries_per_buffer):
        """
        Get the unfilled entries in the actor's subset of the replay buffer using a set difference.
        """
        filled_indices = set(self._get_replay_buffer_filled_indices(self._replay_buffers, actor_index))
        actor_id_set = set(range(0, entries_per_buffer))
        unfilled_indices = actor_id_set - filled_indices
        return unfilled_indices

    def _compute_policy_cloning_loss(self, old_logits, curr_logits):
        # KLDiv requires inputs to be log-probs, and targets to be probs
        old_policy = F.softmax(old_logits, dim=-1)
        curr_log_policy = F.log_softmax(curr_logits, dim=-1)
        kl_loss = torch.nn.KLDivLoss(reduction='sum')(curr_log_policy, old_policy.detach())
        return kl_loss

    def _compute_value_cloning_loss(self, old_value, curr_value):
        return torch.sum((curr_value - old_value.detach()) ** 2)

    def on_act_unroll_complete(self, actor_index, agent_output, env_output, new_buffers):
        """
        Every step, update the replay buffer using reservoir sampling.
        """
        # Compute a reservoir_val for the new entry, then, if the buffer is filled, throw out the entry with the lowest
        # reservoir_val and replace it with the new one. If the buffer it not filled, simply put it in the next spot
        # Using a new RandomState() because using np.random directly is not thread-safe
        random_state = np.random.RandomState()
        new_entry_reservoir_val = random_state.uniform(0.001, 1.0)  # > 0 so we can use reservoir_val==0 to indicate unfilled
        to_populate_replay_index = None
        unfilled_indices = self._get_actor_unfilled_indices(actor_index, self._entries_per_buffer)

        actor_replay_reservoir_vals = self._replay_buffers['reservoir_val'][actor_index]

        if len(unfilled_indices) > 0:
            current_replay_index = min(unfilled_indices)
            to_populate_replay_index = current_replay_index
        else:
            # If we've filled our quota, we need to find something to throw out.
            reservoir_threshold = actor_replay_reservoir_vals.min()

            # If our new value is higher than our existing minimum, replace that one with this new data
            if new_entry_reservoir_val > reservoir_threshold:
                to_populate_replay_index = np.argmin(actor_replay_reservoir_vals)

        # Do the replacement into the buffer, and update the reservoir_vals list
        if to_populate_replay_index is not None:
            with self._replay_lock:
                actor_replay_reservoir_vals[to_populate_replay_index][0] = new_entry_reservoir_val
                for key in new_buffers.keys():
                    if key == 'reservoir_val':
                        continue
                    self._replay_buffers[key][actor_index][to_populate_replay_index][...] = new_buffers[key]

    def get_batch_for_training(self, batch):
        """
        Augment the batch with entries from our replay buffer.
        """
        # Select a random batch set of replay buffers to add also. Only select from ones that have been filled
        with self._replay_lock:
            replay_indices = self._get_replay_buffer_filled_indices(self._replay_buffers, actor_index=None)
            replay_entry_count = int(self._model_flags.batch_size * self._model_flags.replay_ratio)

            # Get the shuffled subset. The defaults to replace=True, which is convenient when we have fewer replay
            # entries than the batch size, and shouldn't matter as the buffer fills. (TODO?)
            shuffled_subset = np.random.choice(replay_indices, replay_entry_count)

            replay_batch = {
                # Get the actor_index and entry_id from the raw id
                key: torch.stack([self._replay_buffers[key][m // self._entries_per_buffer][m % self._entries_per_buffer]
                                  for m in shuffled_subset], dim=1) for key in self._replay_buffers
            }

            assert torch.sum(replay_batch["reservoir_val"] > 0) == replay_entry_count, "Incorrect replay entries retrieved"

        replay_batch = {k: t.to(device=self._model_flags.device, non_blocking=True) for k, t in replay_batch.items()}

        # Combine the replay in with the recent entries
        combo_batch = {
            key: torch.cat((batch[key], replay_batch[key]), dim=1) for key in batch
        }

        # Store the batch so we can generate some losses with it
        self._replay_batches_for_loss.put(replay_batch)

        return combo_batch

    def custom_loss(self, model, initial_agent_state):
        """
        Compute the policy and value cloning losses
        """
        replay_batch = self._replay_batches_for_loss.get()
        replay_learner_outputs, unused_state = model(replay_batch, initial_agent_state)

        replay_batch_policy = replay_batch['policy_logits']
        current_policy = replay_learner_outputs['policy_logits']
        policy_cloning_loss = self._model_flags.policy_cloning_cost * self._compute_policy_cloning_loss(
            replay_batch_policy,
            current_policy)

        replay_batch_baseline = replay_batch['baseline']
        current_baseline = replay_learner_outputs['baseline']
        value_cloning_loss = self._model_flags.value_cloning_cost * self._compute_value_cloning_loss(
            replay_batch_baseline,
            current_baseline)

        cloning_loss = policy_cloning_loss + value_cloning_loss
        stats = {"policy_cloning_loss": policy_cloning_loss.item(),
                 "value_cloning_loss": value_cloning_loss.item()}

        return cloning_loss, stats