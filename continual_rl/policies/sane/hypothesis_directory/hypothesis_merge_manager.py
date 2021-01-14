import torch
import numpy as np
from continual_rl.policies.sane.hypothesis_directory.utils import Utils


class HypothesisMergeManager(object):
    """
    Manages the consolidation of hypotheses, both long-term (gate) and short-term (leaf primitive).
    """

    def __init__(self, data, lifetime_manager):
        self._data = data
        self._lifetime_manager = lifetime_manager  # Also our access point to hypothesis comms

    @property
    def logger(self):
        logger = Utils.create_logger(f"{self._data._output_dir}/core_process.log")
        return logger

    def _ensure_max_hypotheses_for_directory(self, layer_id, directory, num_samples):
        max_hypotheses = self._data._max_hypotheses_per_layer[layer_id]
        hypotheses_merged = []

        if directory is None:
            directory = self._data._long_term_directory

        if len(directory) > 0:
            while len(directory) > max_hypotheses:
                if directory[0].is_long_term:
                    offset = 0
                    hypothesis_merged = self._create_best_meta(offset=offset, directory=directory, max_layer_hypotheses=max_hypotheses)
                    hypotheses_merged.append(hypothesis_merged.prototype)
                else:
                    hypothesis_merged = self._merge_short_term_hypotheses(directory, max_hypotheses)

                hypotheses_merged.append(hypothesis_merged)

                # TODO: this gives us a *chance* of training the new hypothesis before it gets squished again. TODO: ensure this? do it at all? or just let the replay merging take care of it
                # TODO: directly accessing these hypothesis_comms is suboptimal - "train manager"?
                self._lifetime_manager.get_comms(hypothesis_merged).train(num_samples=num_samples, id_start_frac=0,
                                                                  id_end_frac=1,
                                                                  num_times_to_train=self._data._num_times_to_train_meta)

                # TODO: if we don't train the prototype's critic, after we saturate our memory, it's very hard to update the prototype
                # (the new one has to beat the existing one in number of usages)
                if hypothesis_merged.is_long_term:
                    self.logger.info(f"Training prototype: {hypothesis_merged.prototype.friendly_name}")
                    self._lifetime_manager.get_comms(hypothesis_merged.prototype).train(num_samples=num_samples, id_start_frac=0,
                                                                    id_end_frac=1,
                                                                    num_times_to_train=self._data._num_times_to_train_meta)
                # TODO: wait for it?

            if directory[0].is_long_term:
                for gate_entry in directory:  # TODO: this recursion is not the most friendly for parallelizing
                    hypotheses_merged.extend(self._ensure_max_hypotheses_for_directory(layer_id + 1, gate_entry.short_term_versions,
                                                                  num_samples))

        self.logger.info(f"Layer {layer_id} and down merged {len(hypotheses_merged)} hypotheses")
        return hypotheses_merged

    def _create_best_meta(self, offset, directory, max_layer_hypotheses):
        """
        Create a new hypothesis using two existing hypotheses. The existing hypotheses are stored as short-term versions of the new hypothesis.
        In other words, we consider these two hypotheses to be instances of some abstraction (the new hypothesis).
        """
        #directory_subset = [hypo.prototype for hypo in
        #                    directory[offset:int(max_layer_hypotheses * self._data._merge_ratio) + offset]]
        directory_subset = directory[offset:int(max_layer_hypotheses * self._data._merge_ratio) + offset]
        self.logger.info(f"Getting closest hypotheses from subset: {[d.policy for d in directory_subset]}")
        long_term_hypothesis_ids = self._get_closest_hypothesis_ids(directory_subset)
        long_term_hypothesis_ids = [id + offset for id in long_term_hypothesis_ids]
        long_term_hypotheses = [directory[id] for id in
                                long_term_hypothesis_ids]  # TODO: this may be out of date...what policy/filter to use?

        self.logger.info(
            f"Found min ids: {long_term_hypothesis_ids} mapping to {[h.friendly_name for h in long_term_hypotheses]}")
        assert len(long_term_hypothesis_ids) == 2, "Attempting to merge an unexpected number of hypotheses"

        # TODO: send one of the pattern filters over as a seed? - yeah, doing this now, so clean this whole method up
        new_meta_hypothesis = long_term_hypotheses[0]
        hypothesis_to_delete = long_term_hypotheses[1]

        if long_term_hypotheses[1].non_decayed_usage_count > long_term_hypotheses[0].non_decayed_usage_count:
            new_meta_hypothesis = long_term_hypotheses[1]
            hypothesis_to_delete = long_term_hypotheses[0]

        # Update the parameters of the new meta hypothesis using the PROTOTYPE stored in the long-term gate
        # TODO: since we're re-using one of the gates, we don't have to grab its entries and send them over
        self._merge_replay_buffers(long_term_hypotheses[0], long_term_hypotheses[1],
                                   destination_hypothesis=new_meta_hypothesis)
        self._merge_replay_buffers(long_term_hypotheses[0].prototype, long_term_hypotheses[1].prototype,
                                   destination_hypothesis=new_meta_hypothesis.prototype)

        # TODO: this is necessary for determining which entries to merge?
        #new_meta_hypothesis.prototype._policy.data = self._create_combined_policy([long_term_hypotheses[0].prototype,
        #                                                                          long_term_hypotheses[1].prototype])
        new_meta_hypothesis._policy.data = self._create_combined_policy([long_term_hypotheses[0], long_term_hypotheses[1]])

        new_meta_hypothesis.usage_count = 0  # (long_term_hypotheses[0].usage_count + long_term_hypotheses[1].usage_count) // 2  # TODO: consistent with ST creation
        new_meta_hypothesis.prototype.usage_count = 0  # (long_term_hypotheses[0].usage_count + long_term_hypotheses[1].usage_count) // 2

        # Sum not average because the hypothesis, after merger, represents the total expertise of both... you know, in theory
        new_meta_hypothesis.non_decayed_usage_count = (long_term_hypotheses[0].non_decayed_usage_count +
                                                       long_term_hypotheses[1].non_decayed_usage_count) #// 2  # TODO: consistent with ST creation
        new_meta_hypothesis.prototype.non_decayed_usage_count = (long_term_hypotheses[0].non_decayed_usage_count +
                                                                 long_term_hypotheses[1].non_decayed_usage_count) #// 2

        #self.logger.info(f"Kept hypothesis {new_meta_hypothesis.friendly_name} as meta, with policy {new_meta_hypothesis.prototype.policy}")
        self.logger.info(f"Kept hypothesis {new_meta_hypothesis.friendly_name} as meta, with policy {new_meta_hypothesis.policy}")

        assert long_term_hypothesis_ids[0] != long_term_hypothesis_ids[1], "If the hypotheses have the same id, then we just use sequential hypotheses with no errors"

        # Add the existing long-term hypotheses as short-term versions of the new hypothesis, and then delete them.
        for long_term_gate in [hypothesis_to_delete]:  # Keeping it like this to make it easy to switch back if I want to (TODO)
            self.logger.info(
                f"Creating meta from, and deleting {long_term_gate.friendly_name} (STVs: {[entry.friendly_name for entry in long_term_gate.short_term_versions]})")

            # Set the long-term parameters to reflect the new parenting situation
            # long_term_prototype.num_times_used_independently = 0  # So it doesn't immediately get re-spun out into a new LT
            # long_term_prototype.usage_count = 0  # So it has to collect data in its new home

            # Add the STVs to the meta
            for short_term_version_id in sorted(range(len(long_term_gate.short_term_versions)), reverse=True):
                short_term_version = long_term_gate.short_term_versions[short_term_version_id]
                self.logger.info(
                    f"Deleting STV {short_term_version.friendly_name} from {long_term_gate.friendly_name}")

                self._lifetime_manager.delete_hypothesis(short_term_version,
                                       kill_process=False)  # TODO: "delete" here is really a bit weird, maybe just access parent's remove_short_term directly
                new_meta_hypothesis.add_short_term(short_term_version)

        for short_term_version in new_meta_hypothesis.short_term_versions:
            short_term_version.usage_count = 0

        # We use entry new_meta_hypothesis_index as the new meta, delete the other one
        # What this is doing is moving the hypothesis to delete as a child under the meta
        hypothesis_to_delete_duplicate = self._lifetime_manager._duplicate_hypothesis(hypothesis_to_delete, new_meta_hypothesis, hypothesis_to_delete,
                              random_policy=False, keep_non_decayed=True)  # Keeping non-decayed because it's not a new hypo, really (and for merging considerations) TODO: This is not exactly efficient...
        self._lifetime_manager.delete_hypothesis(hypothesis_to_delete, kill_process=True)  # Killing the gate; the prototype lives on
        self._lifetime_manager.delete_hypothesis(hypothesis_to_delete.prototype,
                               kill_process=True)  # Killing the gate; the prototype lives on - jk, not with the STV version

        return new_meta_hypothesis

    def _merge_short_term_hypotheses(self, short_term_directory, max_layer_hypotheses):
        """
        Combine two short-term hypotheses into one hypothesis: replay buffer, policy, etc.
        """
        hypo_to_delete_id, hypo_to_keep_id = self._get_closest_hypothesis_ids(
            short_term_directory[:int(max_layer_hypotheses * self._data._merge_ratio)])

        assert hypo_to_delete_id != hypo_to_keep_id, "Should never be trying to combine a hypothesis with itself"

        if short_term_directory[hypo_to_delete_id].non_decayed_usage_count > short_term_directory[hypo_to_keep_id].non_decayed_usage_count:
            hypo_to_keep_id, hypo_to_delete_id = hypo_to_delete_id, hypo_to_keep_id

        hypo_to_keep = short_term_directory[hypo_to_keep_id]
        hypo_to_delete = short_term_directory[hypo_to_delete_id]

        self.logger.info(
            f"Merging STs {hypo_to_keep.friendly_name} (keep) and {hypo_to_delete.friendly_name} (delete)")

        # Combine their replay entries
        self._merge_replay_buffers(hypo_to_keep, hypo_to_delete, destination_hypothesis=hypo_to_keep)

        # Combine their policy
        hypo_to_keep._policy.data = self._create_combined_policy([hypo_to_keep, hypo_to_delete])
        hypo_to_keep.usage_count_since_last_update += hypo_to_delete.usage_count_since_last_update
        # hypo_to_keep.usage_count = (hypo_to_keep.usage_count + hypo_to_delete.usage_count) //2  # TODO - maybe just set to 0 so it has to get used?
        hypo_to_keep.usage_count = 0  # (hypo_to_keep.usage_count + hypo_to_delete.usage_count) //2  # 0 TODO - maybe just set to 0 so it has to get used?

        # As with metas, the kept hypothesis now represents the total of the expertise, not the mean
        hypo_to_keep.non_decayed_usage_count = (hypo_to_keep.non_decayed_usage_count + hypo_to_delete.non_decayed_usage_count) #// 2

        self._lifetime_manager.delete_hypothesis(short_term_directory[hypo_to_delete_id], kill_process=True)

        return hypo_to_keep

    def _scale_usage_count(self, usage_count):
        # If we allow the usage counts to scale linearly, then existing hypotheses that have been used extensively will
        # essentially never "lose ground" to new hypotheses
        # TODO: what scalings?
        return np.tanh(usage_count / self._data._config.usage_scale) * 100

    def _get_closest_hypothesis_ids(self, directory):  # TODO unit test, and de-dupe with the version in replay buffer
        """
        Get the two hypotheses that are "closest" by some metric. Currently this metric is the distance between their policies.
        """
        # TODO: to consider: scipy.stats.wasserstein_distance, see: https://github.com/scipy/scipy/issues/9152 for demo
        policies = torch.stack([entry.get_policy_as_categorical().probs for entry in directory])
        difference = policies.unsqueeze(1) - policies  # Taking advantage of the auto-dim matching thing.
        square_distances = (difference ** 2).sum(dim=-1).detach().cpu()

        self.logger.info(f"Square distances: {square_distances}")

        side_length = square_distances.shape[0]
        indices = np.argsort(square_distances, axis=None)
        indices_x = indices // side_length
        indices_y = indices % side_length

        hypo_index = 0

        # Ignore diagonals
        while indices_x[hypo_index] == indices_y[hypo_index]:
            hypo_index += 1

        selected_x = indices_x[hypo_index]
        selected_y = indices_y[hypo_index]

        self.logger.info(f"Selected index ({hypo_index}): {selected_x} {selected_y} from indices: {indices}, indices_x: {indices_x}, indices_y: {indices_y}")

        # Returns two lists of indices. indices_x[0] pairs with indices_y[0]. TODO: make clearer
        return selected_x, selected_y  # Just taking the first for consistency with the old API. Should consider enabling more than one pair

    def _merge_replay_buffers(self, hypothesis_0, hypothesis_1, destination_hypothesis):
        """
        Add new entries to container_replay randomly from the replay buffers of the passed-in hypotheses. The number of each selected
        is proportional to their usage counts.
        """
        entries_to_add = []

        # TODO: use public methods on the replay buffer. Just being lazy
        # TODO: also just being lazy about the replay_length thing - it's kind of double counting the usage...except that currently it's just getting max
        hypothesis_0_replay_length = self._scale_usage_count(hypothesis_0.non_decayed_usage_count)  #self._hypothesis_comms[hypothesis_0].get_replay_buffer_length()  # This must be fetched from the HYPOTHESIS process TODO: sync like I'm doing with Env/Train?
        hypothesis_1_replay_length = self._scale_usage_count(hypothesis_1.non_decayed_usage_count)  #self._hypothesis_comms[hypothesis_1].get_replay_buffer_length()  # This must be fetched from the HYPOTHESIS process
        total_usage = hypothesis_0_replay_length + hypothesis_1_replay_length
        total_replay_size = min(total_usage, hypothesis_0._replay_buffer.maxlen)

        num_hypothesis_0_to_get = int(total_replay_size * (hypothesis_0_replay_length + 1) / (total_usage + 1))  # TODO: naive prevention of div by 0
        num_hypothesis_1_to_get = int(total_replay_size * (hypothesis_1_replay_length + 1) / (total_usage + 1))

        if hypothesis_0 is not destination_hypothesis:
            entries_from_hypo_0 = self._lifetime_manager.get_comms(hypothesis_0).get_random_replay_buffer_entries(
                num_to_get=num_hypothesis_0_to_get)
            entries_to_add.append(entries_from_hypo_0)

        if hypothesis_1 is not destination_hypothesis:
            entries_from_hypo_1 = self._lifetime_manager.get_comms(hypothesis_1).get_random_replay_buffer_entries(
                num_to_get=num_hypothesis_1_to_get)
            entries_to_add.append(entries_from_hypo_1)

        np.random.shuffle(entries_to_add)

        # If the destination hypothesis is one of the source hypotheses, we're not grabbing replay entries and passing them,
        # we're just letting them remain as they are, for speed. Otherwise clear the replay buffer
        if destination_hypothesis is not hypothesis_0 and destination_hypothesis is not hypothesis_1:
            self._lifetime_manager.get_comms(destination_hypothesis).clear_replay()

        self._lifetime_manager.get_comms(destination_hypothesis).add_many_to_replay(entries_to_add)
        self.logger.info(f"Merging replay buffers with usage counts {hypothesis_0.usage_count} (replay length: {hypothesis_0_replay_length}) and {hypothesis_1.usage_count} (replay length: {hypothesis_1_replay_length})")

    def _create_combined_policy(self, hypotheses):
        """
        Combine policies weighted by their usage.
        """
        self.logger.info("Combining policies")
        scaled_policy = 0
        total_usages = 0

        for hypothesis in hypotheses:
            scaled_usage_count = self._scale_usage_count(hypothesis.non_decayed_usage_count + 1)  # +1 to prevent /0
            total_usages += scaled_usage_count
            scaled_policy += scaled_usage_count * hypothesis._policy.data
            self.logger.info(f"Adding {hypothesis._policy} with nd usage counts {hypothesis.non_decayed_usage_count} (scaled: {scaled_usage_count})")

        normalized_policy = scaled_policy / total_usages
        self.logger.info(f"Final combined policy: {normalized_policy}")

        return normalized_policy
