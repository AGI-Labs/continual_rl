import numpy as np
import torch
import threading
import os
from torch.nn import functional as F
import queue
from continual_rl.policies.impala.torchbeast.monobeast import Monobeast, Buffers
from continual_rl.utils.utils import Utils


class ClearMonobeast(Monobeast):
    """
    An implementation of Experience Replay for Continual Learning (Rolnick et al, 2019):
    https://arxiv.org/pdf/1811.11682.pdf
    """

    def __init__(self, model_flags, observation_space, action_spaces, policy_class):
        super().__init__(model_flags, observation_space, action_spaces, policy_class)
        common_action_space = Utils.get_max_discrete_action_space(action_spaces)

        torch.multiprocessing.set_sharing_strategy(model_flags.torch_multiprocessing_sharing_strategy)

        # LSTMs not supported largely because they have not been validated; nothing extra is stored for them.
        assert not model_flags.use_lstm, "CLEAR does not presently support using LSTMs."
        assert self._model_flags.always_reuse_actor_indices or self._model_flags.num_actors >= int(self._model_flags.batch_size * self._model_flags.batch_replay_ratio), \
            "Each actor only gets sampled from once during training, so we need at least as many actors as batch_size"
        self._model_flags = model_flags

        # We want the replay buffers to be created in the large_file_path,
        # but in a place characteristic to this experiment.
        # Be careful if the output_dir specified is very nested
        # (ie. Windows has max path length of 260 characters)
        # Could hash output_dir_str if this is a problem.
        output_dir_str = os.path.normpath(model_flags.output_dir).replace(os.path.sep, '-')
        permanent_path = os.path.join(
            model_flags.large_file_path,
            "file_backed",
            output_dir_str,
            f"{model_flags.policy_unique_id}"
        )
        buffers_existed = os.path.exists(permanent_path)
        os.makedirs(permanent_path, exist_ok=True)

        self._entries_per_buffer = int(
            model_flags.replay_buffer_frames // (model_flags.unroll_length * model_flags.num_actors)
        )
        self._replay_buffers, self._temp_files = self._create_replay_buffers(
            model_flags,
            observation_space.shape,
            common_action_space.n,
            self._entries_per_buffer,
            permanent_path,
            buffers_existed,
        )
        self._replay_lock = threading.Lock()

        # Each replay batch needs to also have cloning losses applied to it
        # Keep track of them as they're generated, to ensure we apply losses to all. This doesn't currently
        # guarantee order - i.e. one learner thread might get one replay batch for training and a different for cloning
        self._replay_batches_for_loss = queue.Queue()

    def permanent_delete(self):
        for file_path in self._temp_files:
            os.remove(file_path)

        del self._replay_buffers
        del self.buffers

        super().cleanup()

    def _create_replay_buffers(
        self,
        model_flags,
        obs_shape,
        num_actions,
        entries_per_buffer,
        permanent_path,
        buffers_existed,
    ):
        """
        Key differences from normal buffers:
        1. File-backed, so we can store more at a time
        2. Structured so that there are num_actors buffers, each with entries_per_buffer entries

        Each buffer entry has unroll_length size, so the number of frames stored is (roughly, because of integer
        rounding): num_actors * entries_per_buffer * unroll_length
        """
        # Get the standard specs, and also add the CLEAR-specific reservoir value
        specs = self.create_buffer_specs(model_flags.unroll_length, obs_shape, num_actions)
        # Note: one reservoir value per row
        specs["reservoir_val"] = dict(size=(1,), dtype=torch.float32)
        buffers: Buffers = {key: [] for key in specs}

        # Hold on to the file handle so it does not get deleted. Technically optional, as at least linux will
        # keep the file open even after deletion, but this way it is still visible in the location it was created
        temp_files = []

        for actor_id in range(model_flags.num_actors):
            for key in buffers:
                shape = (entries_per_buffer, *specs[key]["size"])
                permanent_file_name = f"replay_{actor_id}_{key}.fbt"
                new_tensor, file_name, temp_file = Utils.create_file_backed_tensor(
                    permanent_path,
                    shape,
                    specs[key]["dtype"],
                    permanent_file_name=permanent_file_name,
                )

                # reservoir_val needs to be 0'd out so we can use it to see if a row is filled
                # but this operation is slow, so leave the rest as-is
                # Only do this if we created the buffers anew
                if not buffers_existed and key == "reservoir_val":
                    new_tensor.zero_()

                buffers[key].append(new_tensor)
                temp_files.append(file_name)

        return buffers, temp_files

    def _get_replay_buffer_filled_indices(self, replay_buffers, actor_index):
        """
        Get the indices in the replay buffer corresponding to the actor_index.
        """
        # We know that the reservoir value > 0 if it's been filled, so check for entries where it == 0
        buffer_indicator = replay_buffers['reservoir_val'][actor_index].squeeze(1)
        replay_indices = np.where(buffer_indicator != 0)[0]
        return replay_indices

    def _get_actor_unfilled_indices(self, actor_index, entries_per_buffer):
        """
        Get the unfilled entries in the actor's subset of the replay buffer using a set difference.
        """
        filled_indices = set(
            self._get_replay_buffer_filled_indices(self._replay_buffers, actor_index)
        )
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

    def get_min_reservoir_val_greater_than_zero(self):
        reservoir_vals = torch.stack(self._replay_buffers['reservoir_val'])
        vals_gt_zero = reservoir_vals[reservoir_vals > 0]

        if len(vals_gt_zero) > 0:
            min_val = vals_gt_zero.min()
        else:
            min_val = 0

        return min_val

    def on_act_unroll_complete(self, task_flags, actor_index, agent_output, env_output, new_buffers):
        """
        Every step, update the replay buffer using reservoir sampling.
        """
        # Compute a reservoir_val for the new entry, then, if the buffer is filled, throw out the entry with the lowest
        # reservoir_val and replace it with the new one. If the buffer it not filled, simply put it in the next spot
        # Using a new RandomState() because using np.random directly is not thread-safe
        random_state = np.random.RandomState()

        # > 0 so we can use reservoir_val==0 to indicate unfilled
        new_entry_reservoir_val = random_state.uniform(0.001, 1.0) if "reservoir_val" not in new_buffers.keys() else new_buffers["reservoir_val"].item()
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

    def get_batch_for_training(self, batch, store_for_loss=True, reuse_actor_indices=False, replay_entry_scale=1.0):
        """
        Augment the batch with entries from our replay buffer.
        """
        # Select a random batch set of replay buffers to add also. Only select from ones that have been filled
        shuffled_subset = []  # Will contain a list of tuples of (actor_index, buffer_index)

        # We only allow each actor to be sampled from once, to reduce variance, and for parity with the original
        # paper
        actor_indices = list(range(self._model_flags.num_actors))
        replay_entry_count = int(self._model_flags.batch_size * self._model_flags.batch_replay_ratio * replay_entry_scale)
        assert replay_entry_count > 0, "Attempting to run CLEAR without actually using any replay buffer entries."

        random_state = np.random.RandomState()

        with self._replay_lock:
            # Select a random actor, and from that, a random buffer entry.
            for _ in range(replay_entry_count):
                # Pick an actor and remove it from our options
                actor_index = random_state.choice(actor_indices)

                if not reuse_actor_indices and not self._model_flags.always_reuse_actor_indices:
                    actor_indices.remove(actor_index)

                # From that actor's set of available indices, pick one randomly.
                replay_indices = self._get_replay_buffer_filled_indices(self._replay_buffers, actor_index=actor_index)
                if len(replay_indices) > 0:
                    buffer_index = random_state.choice(replay_indices)
                    shuffled_subset.append((actor_index, buffer_index))

            if len(shuffled_subset) > 0:
                replay_batch = {
                    # Get the actor_index and entry_id from the raw id
                    key: torch.stack([self._replay_buffers[key][actor_id][buffer_id]
                                      for actor_id, buffer_id in shuffled_subset], dim=1)
                    for key in self._replay_buffers
                }

                replay_entries_retrieved = torch.sum(replay_batch["reservoir_val"] > 0)
                assert replay_entries_retrieved <= replay_entry_count, \
                    f"Incorrect replay entries retrieved. Expected at most {replay_entry_count} got {replay_entries_retrieved}"

                replay_batch = {
                    k: t.to(device=self._model_flags.device, non_blocking=True)
                    for k, t in replay_batch.items()
                }

                # Combine the replay in with the recent entries
                if batch is not None:
                    combo_batch = {
                        key: torch.cat((batch[key], replay_batch[key]), dim=1) for key in batch
                    }
                else:
                    combo_batch = replay_batch

                # Store the batch so we can generate some losses with it
                if store_for_loss:
                    self._replay_batches_for_loss.put(replay_batch)

            else:
                combo_batch = batch

        return combo_batch

    def custom_loss(self, task_flags, model, initial_agent_state, batch, vtrace_returns):
        """
        Compute the policy and value cloning losses
        """
        # If the get doesn't happen basically immediately, it's not happening
        cloning_loss = torch.Tensor([0]).to(batch['frame'].device)
        stats = {}

        try:
            replay_batch = self._replay_batches_for_loss.get(timeout=5)
        except queue.Empty:
            replay_batch = None
            print("Skipping CLEAR custom loss due to lack of replay_batch")

        if replay_batch is not None:
            replay_learner_outputs, unused_state = model(replay_batch, task_flags.action_space_id, initial_agent_state)

            replay_batch_policy = replay_batch['policy_logits']
            current_policy = replay_learner_outputs['policy_logits']
            policy_cloning_loss = self._model_flags.policy_cloning_cost * self._compute_policy_cloning_loss(
                replay_batch_policy, current_policy)

            replay_batch_baseline = replay_batch['baseline']
            current_baseline = replay_learner_outputs['baseline']
            value_cloning_loss = self._model_flags.value_cloning_cost * self._compute_value_cloning_loss(
                replay_batch_baseline, current_baseline)

            cloning_loss = policy_cloning_loss + value_cloning_loss
            stats = {
                "policy_cloning_loss": policy_cloning_loss.item(),
                "value_cloning_loss": value_cloning_loss.item(),
            }

        return cloning_loss, stats
