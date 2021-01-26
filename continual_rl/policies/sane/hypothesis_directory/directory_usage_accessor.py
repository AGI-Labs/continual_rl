import torch
import numpy as np
from collections import deque
from continual_rl.policies.sane.hypothesis.usage_accessor import UsageAccessor
from continual_rl.policies.sane.hypothesis_directory.utils import Utils


class DirectoryUsageAccessor(object):
    """
    Methods associated exclusively with the Usage process accessing of the Directory.
    "Accessor" is a bit misleading, because this also contains usage-only state data.
    """

    def __init__(self, directory_data):
        self._data = directory_data

        self._hypothesis_creation_buffer = {}
        self.hypothesis_accessor = UsageAccessor

        self._last_used_hypotheses = deque(maxlen=self._data._config.used_hypotheses_count)  # TODO: this isn't persisted in any way, it's just a local best attempt

    @property
    def logger(self):
        logger = Utils.create_logger(f"{self._data._output_dir}/usage_process.log")
        return logger

    def _add_to_creation_buffer(self, creation_buffer, hypothesis, parent, pattern_filter_source, random_policy, priority):
        hypo_id = hypothesis.unique_id if hypothesis is not None else None
        parent_id = parent.unique_id if parent is not None else None
        pattern_filter_source_id = pattern_filter_source.unique_id if pattern_filter_source is not None else None

        if hypo_id not in creation_buffer:
            creation_buffer[hypo_id] = {}
            count = 0
        elif priority not in creation_buffer[hypo_id]:
            count = 0
        else:
            count = creation_buffer[hypo_id][priority]["count"]

        count += 1

        # TODO: if I keep it this way, rename priority
        entry = {"count": count, "priority": priority, "parent_id": parent_id,
                 "pattern_filter_source_id": pattern_filter_source_id, "random_policy": random_policy}

        creation_buffer[hypo_id][priority] = entry

    def _get_pattern_filter_result(self, hypothesis, x):
        with torch.no_grad():
            result = hypothesis.pattern_filter(x.to(hypothesis._device)).to(self._data._master_device_id)
        return result

    def _apply_filter_adjustments(self, hyp, raw_result, use_refractory, refractory_cache, default_cache_entry):
        scaled_result = raw_result.squeeze(0)

        # (If refractory is on) if the number of steps we still have left to wait is greater than 0, set the filter result to a very negative value, so this hypothesis doesn't get triggered
        if (use_refractory and refractory_cache.get(hyp.unique_id, default_cache_entry.copy())[1] > 0):
            scaled_result[0] = 0 * scaled_result[0] - 1e5
            scaled_result[1] = 0 * scaled_result[1]  # Currently gets abs'd before getting added, so having it large is not useful

        if hyp.unique_id in self._last_used_hypotheses:
            scaled_result *= self._data._config.recently_used_multiplier  # TODO: assumes positive!

        return scaled_result.unsqueeze(0)

    def _get_from_directory(self, x, directory, per_episode_storage, refractory_step_counts):
        """
        Get the best hypothesis from the specified directory. Various metrics for "best" in progress, so not describing it here for now. (TODO)
        """
        selected_hypothesis = None
        selected_filter_result = None

        refractory_cache = per_episode_storage.get("refractory_cache", {})
        default_cache_entry = [0, 0]  # Number of times used, number of steps left to wait
        use_refractory = refractory_step_counts is not None
        default_skip_hypothesis_chance = 0  # Only allow skipping when the top two choices are close

        if len(directory) > 0:
            pattern_filter_results_raw = [self._apply_filter_adjustments(hyp, self._get_pattern_filter_result(hyp, x),
                                                                         use_refractory, refractory_cache, default_cache_entry) for hyp in directory]

            pattern_filter_results = [entry[:, 0] + self._data._allowed_error_scale * self._convert_filter_out_to_error(entry, get_pos=True)
                                      for entry in pattern_filter_results_raw]

            # hypothesis x results
            pattern_filter_results = torch.stack(pattern_filter_results).squeeze(-1)
            sorted_ids = torch.argsort(pattern_filter_results, descending=True, dim=0)

            # If the top two are close (ie we just cloned a hypothesis), give them equal chance
            if len(sorted_ids) >= 2 and torch.abs(pattern_filter_results[sorted_ids[0]] - pattern_filter_results[sorted_ids[1]]) < self._data._closeness_threshold:
                skip_hypothesis_chance = .5
            else:
                skip_hypothesis_chance = default_skip_hypothesis_chance

            # Select our hypothesis, with some chance of skipping down to the next one
            for id in sorted_ids:
                if np.random.uniform(0, 1.0) < 1 - skip_hypothesis_chance:
                    selected_hypothesis = directory[id]
                    selected_filter_result = pattern_filter_results[id]
                    break

                # Only let the bonus increased skip last one cycle (technically this means they're slightly unequal, but at least closer)
                skip_hypothesis_chance = default_skip_hypothesis_chance

        if use_refractory:
            refractory_threshold, refractory_num_wait = refractory_step_counts

            # After the hypothesis has fired n times in a row, make it wait m turns before firing again
            for hyp in directory:
                current_entry = refractory_cache.get(hyp.unique_id, default_cache_entry.copy())
                if hyp == selected_hypothesis:
                    current_entry[0] += 1  # Usage count

                    if current_entry[0] >= refractory_threshold:  # If we've exceeded our threshold, trigger a waiting period. TODO: separate val?
                        current_entry[1] = refractory_num_wait  # Steps left to wait
                else:
                    current_entry[0] = 0
                    current_entry[1] = max(0, current_entry[1] - 1)  # Has to not get used n times in a row before it's usable again

                refractory_cache[hyp.unique_id] = current_entry

            per_episode_storage["refractory_cache"] = refractory_cache

            # TODO: hacky - if the chosen one should be "turned off", make a new hypothesis.... This is not a good check though, just lazy
            if selected_filter_result is None or selected_filter_result < -1e4:
                selected_hypothesis = None

        if selected_hypothesis is not None:
            self._last_used_hypotheses.append(selected_hypothesis.unique_id)

        return selected_hypothesis

    def _convert_filter_out_to_error(self, raw_out, get_pos):
        error = raw_out[:, 1]

        # Abs instead of relu because if it's negative, that should be treated as "needs more data" not "needs no data"
        return torch.abs(error) #F.relu(raw_out)  # Otherwise newly duplicated long-terms can continuously create short-terms

    def _convert_filter_out_to_closeness(self, raw_out):
        closeness = raw_out[:, 2]
        return torch.sigmoid(closeness)

    def _compute_new_creation_buffer_entries(self, step_creation_buffer, long_term_entry, short_term_entry, x):

        if long_term_entry is None:
            self._add_to_creation_buffer(step_creation_buffer, None, parent=None, random_policy=True,
                                         priority=0, pattern_filter_source=None)  # The params are ignored for None anyway
            self.logger.info(f"Creating new LT because none was found.")

        elif short_term_entry is None:
            self._add_to_creation_buffer(step_creation_buffer, long_term_entry, parent=long_term_entry,
                                         random_policy=False, priority=5, pattern_filter_source=long_term_entry)
            self.logger.info(f"Adding LT {long_term_entry.friendly_name} with prototype "
                             f"{long_term_entry.prototype.friendly_name} to creation buffer because no ST was found")

        elif (self._data._min_short_term_total_usage_count < 1 and short_term_entry.usage_count > short_term_entry.non_decayed_usage_count * self._data._min_short_term_total_usage_count and short_term_entry.non_decayed_usage_count > 10000) or \
                (self._data._min_short_term_total_usage_count >= 1 and short_term_entry.usage_count > self._data._min_short_term_total_usage_count):
            # If the min_usage-count is < 1 we assume it's a ratio of the total seen usage, otherwise we assume it's an absolute threshold

            # TODO: do I want to have a usage count that decays, so that it captures recency better than just "what's happened in this episode" - more robust to
            # process counts, etc
            # The problem being encountered is that old hypotheses that start getting triggered will rapidly "take over" during an episode when they start getting triggered, before
            # having gotten trained at all. Getting enough usages, and being wrong enough about their estimates to get created.
            # I'm pretty sure the tuning of the ST usage count I'm having to do is to get the cadence of creation to tuning correct....gah

            # Note that on the USAGE process, the usage_count is the usage since the last training step (the usages while this process has been alive).
            # That means it's somewhat a function of how many environments are on a particular process
            # I initially intended to use the *total* usage_count over all time, but that trained significantly worse
            # TODO: not accurate: and short_term_entry.usage_count > self._data._min_short_term_episode_usage_count:

            # We use the PROTOTYPE here because it acts as an anchor - if the other hypotheses drift too far from it, they get converted to LTs
            long_term_pattern_result_raw = self._get_pattern_filter_result(long_term_entry, x)  # Squeeze out the batch
            long_term_prototype_pattern_result_raw = self._get_pattern_filter_result(long_term_entry.prototype, x)  # Squeeze out the batch. DON'T USE TO COMPUTE LCB/UCB, since its uncertainty doesn't decrease
            short_term_pattern_result_raw = self._get_pattern_filter_result(short_term_entry, x)
            #print(
            #    f"Long raw: {long_term_pattern_result_raw} Proto raw: {long_term_prototype_pattern_result_raw} Short raw: {short_term_pattern_result_raw} with "
            #    f"Proto usage count {long_term_entry.prototype.usage_count} and ST total usage count {short_term_entry.usage_count} and ST usage count {short_term_entry.usage_count}")

            short_term_upper_bound = short_term_pattern_result_raw[:, 0] + self._data._allowed_error_scale * self._convert_filter_out_to_error(short_term_pattern_result_raw, get_pos=True)
            long_term_upper_bound = long_term_pattern_result_raw[:, 0] + self._data._allowed_error_scale * self._convert_filter_out_to_error(long_term_pattern_result_raw, get_pos=True)

            if self._data._allowed_error_scale_strict is not None:
                if isinstance(self._data._allowed_error_scale_strict, list):
                    allowed_lower = self._data._allowed_error_scale_strict[0]
                    allowed_upper = self._data._allowed_error_scale_strict[1]
                else:
                    allowed_lower = allowed_upper = self._data._allowed_error_scale_strict

                short_term_upper_bound_strict = short_term_pattern_result_raw[:, 0] + allowed_upper * self._convert_filter_out_to_error(
                    short_term_pattern_result_raw, get_pos=True)
                short_term_lower_bound_strict = short_term_pattern_result_raw[:, 0] - allowed_lower * self._convert_filter_out_to_error(
                    short_term_pattern_result_raw, get_pos=False)
            else:
                short_term_upper_bound_strict = None
                short_term_lower_bound_strict = None

            if long_term_entry.prototype.usage_count == 0 and self._data._usage_count_min_to_convert_to_long_term > 0 and \
                    short_term_entry.usage_count > self._data._usage_count_min_to_convert_to_long_term:
                # The initial prototype is essentially random, don't rely on it forever
                self._add_to_creation_buffer(step_creation_buffer, short_term_entry, parent=long_term_entry.parent_hypothesis,
                                             random_policy=False, priority=1, pattern_filter_source=short_term_entry)

                self.logger.info(
                    f"Initial prototype outgrown: adding ST {short_term_entry.friendly_name} from LT {long_term_entry.friendly_name} to creation buffer")

            elif short_term_lower_bound_strict is not None and short_term_lower_bound_strict > long_term_prototype_pattern_result_raw[:, 0] \
                    and (self._data._skip_short_term_greater_than_long_term or short_term_upper_bound > long_term_upper_bound):  # The gate itself has drifted above the prototype, create a new gate from the gate
                # Move the prototype out into a new LT hypothesis, replacing it with the best short-term (TODO?)
                # None of the params here are used other than the hypothesis and the replace_with (TODO)
                self._add_to_creation_buffer(step_creation_buffer, short_term_entry, parent=long_term_entry.parent_hypothesis,
                                             random_policy=False, priority=1, pattern_filter_source=short_term_entry)

                self.logger.info(
                    f"STRICT Long LCB > prototype expected value, gate has drifted ABOVE prototype: adding ST {short_term_entry.friendly_name} from LT {long_term_entry.friendly_name} to creation buffer")

            elif short_term_upper_bound_strict is not None and short_term_upper_bound_strict < long_term_prototype_pattern_result_raw[:, 0] \
                    and (self._data._skip_short_term_greater_than_long_term or short_term_upper_bound > long_term_upper_bound):
                # If the prototype is doing better than the gate, if we make a new hypothesis from the prototype, that one will win, even though it's not up-to-date
                # And then that gate will get worse, and a new entry will be made from that prototype, etc. An endless loop of being bad at the task
                # Thus we keep the prototype in the current gate, but create a new LT from our best ST
                self._add_to_creation_buffer(step_creation_buffer, short_term_entry, parent=long_term_entry.parent_hypothesis,
                                             random_policy=False, priority=1, pattern_filter_source=short_term_entry)
                self.logger.info(
                    f"STRICT Long UCB < prototype expected value, gate has drifted BELOW prototype: adding ST {short_term_entry.friendly_name} from LT {long_term_entry.friendly_name} to creation buffer")

    def _get_best_hypotheses(self, x, per_episode_storage):
        best_hypotheses = []  # One entry per layer

        for layer_id in range(len(self._data._max_hypotheses_per_layer)):
            if len(best_hypotheses) == 0:
                directory = self._data._long_term_directory
            else:
                directory = best_hypotheses[-1].short_term_versions

            episode_storage_id = f"layer_{layer_id}"
            layer_episode_storage = per_episode_storage.get(episode_storage_id, {})

            # TODO: get_from_directory still assumes envs, but I've removed it from up here
            refractory_step_counts = self._data._refractory_step_counts_per_layer[layer_id] \
                if self._data._refractory_step_counts_per_layer is not None else None
            best_hypothesis = self._get_from_directory(x, directory,
                                                          per_episode_storage=layer_episode_storage,
                                                          refractory_step_counts=refractory_step_counts)

            per_episode_storage[episode_storage_id] = layer_episode_storage
            best_hypotheses.append(best_hypothesis)

            if best_hypothesis is None:
                break

        return best_hypotheses

    def get(self, x, eval_mode, per_episode_storage):
        """
        Gets the best hypothesis overall. Will create new hypotheses as appropriate. In particular, if a long-term entry
        is beating its best short-term (instance) entry, create a new short-term by duplicating the long-term (added to a buffer here, triggers in the update step).
        If a short-term entry beats its parent long-term entry enough times, it will get spun out into a new long-term entry (in the update step)
        """
        step_creation_buffer = {}
        best_hypotheses = self._get_best_hypotheses(x, per_episode_storage)
        long_term_entry = None  # Holds the *last* gate before a None
        short_term_entry = None

        # Iterate at least once, but otherwise n-1 times
        layer_count = max(1, len(best_hypotheses) - 1)

        # Compare each hypothesis to its parent to determine if new hypotheses need to be created
        for layer_id in range(layer_count):
            long_term_entry = best_hypotheses[layer_id]
            short_term_entry = best_hypotheses[layer_id + 1] if layer_id + 1 < len(best_hypotheses) else None

            # Only create new hypotheses in training mode
            if not eval_mode:
                self._compute_new_creation_buffer_entries(step_creation_buffer, long_term_entry,
                                                          short_term_entry, x)

            # No point in continuing the loop if we've already found a None
            if short_term_entry is None:
                break

        if long_term_entry is None:
            long_term_entry = np.random.choice(self._data._long_term_directory)

        if short_term_entry is None:
            #short_term_entry = long_term_entry.prototype
            short_term_entry = long_term_entry

        return short_term_entry, step_creation_buffer
