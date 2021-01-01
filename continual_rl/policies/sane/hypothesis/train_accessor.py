import torch
import torch.optim as optim
import numpy as np
import gc
from continual_rl.policies.sane.hypothesis_directory.utils import Utils
import copy
from torch.distributions.categorical import Categorical
import traceback


class TrainAccessor(object):
    """
    Due to the quantity and structure of the hypotheses, the hypothesis gets passed into each of these (ie, no state on this object).
    """

    @classmethod
    def train_logger(cls, hypothesis):
        logger = Utils.create_logger(f"{hypothesis._output_dir}/hypothesis_{hypothesis.friendly_name}_train.log")
        return logger

    @classmethod
    def _get_pattern_filter_optimizer(cls, hypothesis, learner):
        """
        Note that the optimizer is supposed to be created after the object is on its gpu, so we're lazy-loading it
        """
        if hypothesis._pattern_filter_optimizer is None:
            params = []
            params.extend(learner.parameters())
            hypothesis._pattern_filter_optimizer = optim.Adam(
                [{'params': params, 'lr': hypothesis._filter_learning_rate}],
                lr=1e-3)

        return hypothesis._pattern_filter_optimizer

    @classmethod
    def _compute_importance_weight(cls, hypothesis, selected_action, orig_log_prob):
        # At the moment this is not super principled, just intuitive. TODO

        if hypothesis.policy is not None:
            policy_dist = Categorical(logits=hypothesis.policy.clone())
            result = torch.exp(policy_dist.log_prob(selected_action)) / torch.exp(orig_log_prob)  # Could be more efficient (re: e^log), but for consistency for now TODO
        else:
            result = torch.Tensor([1])

        return result

    @classmethod
    def load_learner(cls, hypothesis):
        if hypothesis._pattern_filter_learner is not None:
            hypothesis.pattern_filter.load_state_dict(hypothesis._pattern_filter_learner.state_dict())

    @classmethod
    def train_pattern_filter(cls, hypothesis, num_samples, id_start_frac, id_end_frac):
        if hypothesis._pattern_filter_learner is None:
            hypothesis._pattern_filter_learner = copy.deepcopy(hypothesis.pattern_filter)
        pattern_filter_to_train = hypothesis._pattern_filter_learner

        # Fail out quickly if we can't train (TODO: probably better to wrap the rest, but feeling lazy.)
        if len(hypothesis._replay_buffer) == 0:
            return

        if hypothesis._device.type != "cpu":
            torch.cuda.set_device(hypothesis._device)  # Making sure the optimizer creates its stuff in the right place...Otherwise we create loads of tensors on cuda:0

        optimizer = cls._get_pattern_filter_optimizer(hypothesis, hypothesis._pattern_filter_learner)

        max_num_negative_entries = int(
            hypothesis._config.negative_example_scale * min(num_samples, len(hypothesis._replay_buffer)))
        batch_size = hypothesis._config.batch_size

        if max_num_negative_entries > 0:
            negative_replay_entries = hypothesis._negative_examples.get(max_num_negative_entries, 0, 1)
        else:
            negative_replay_entries = []

        if len(hypothesis._replay_buffer) > 0:
            entries = hypothesis._replay_buffer.get(num_samples, id_start_frac, id_end_frac)  # Returns a random subset, shuffled (TODO rename)

            input_states = []
            rewards_received = []
            importance_weights = []
            uncertainty_scale = []  # Negative examples don't contribute to uncertainty, for now (Should it? I'm undecided)

            for entry in entries:
                if entry.reward_received is not None:  # TODO: can remove these checks
                    input_states.append(entry.input_state.squeeze(0))  # Squeeze out the 1 "fake" batch - we'll add in our own batch in the stack below
                    rewards_received.append(entry.reward_received)
                    importance_weights.append(cls._compute_importance_weight(hypothesis, entry.selected_action, entry.action_log_prob))
                    uncertainty_scale.append(torch.Tensor([1]))

            for negative_entry in negative_replay_entries:
                if negative_entry.reward_received is not None:
                    input_states.append(negative_entry.input_state.squeeze(0))
                    # TODO: config -10
                    rewards_received.append(negative_entry.reward_received * 0 - 10)  # Needs to be a value manipulation (I think), to prevent value from just always being high
                    importance_weights.append(cls._compute_importance_weight(hypothesis, negative_entry.selected_action,
                                                                             negative_entry.action_log_prob))
                    uncertainty_scale.append(torch.Tensor([1]))

            if len(input_states) > 0:  # TODO: this shouldn't be happening... This gets called after rewards are assigned
                input_states = torch.stack(input_states)
                rewards_received = torch.stack(rewards_received)
                importance_weights = torch.stack(importance_weights)
                uncertainty_scale = torch.stack(uncertainty_scale)

                indices = np.array(list(range(input_states.shape[0])))
                np.random.shuffle(indices)

                for indices_index in range(0, len(indices), batch_size):
                    indices_subset = indices[indices_index:indices_index + batch_size]

                    # According to the docs here: https://docs.scipy.org/doc/numpy/user/basics.indexing.html indexing by array makes a copy of the data, thus consuming more RAM
                    # More ideal would be slicing, but then folding in the negative_examples is more challenging. TODO: slice, if I remove the negative examples/do them differently
                    input_states_subset = input_states[indices_subset]
                    rewards_received_subset = rewards_received[indices_subset]
                    importance_weights_subset = importance_weights[indices_subset]
                    uncertainty_scale_subset = uncertainty_scale[indices_subset]

                    # Storing the replay buffer not on the gpu because it can get quite big.... Though it's a tradeoff. TODO: better check
                    if hypothesis._device is not None:
                        input_states_subset = input_states_subset.to(hypothesis._device)
                        rewards_received_subset = rewards_received_subset.to(hypothesis._device)
                        importance_weights_subset = importance_weights_subset.to(hypothesis._device)
                        uncertainty_scale_subset = uncertainty_scale_subset.to(hypothesis._device)

                    # =====Train the pattern_filter network=====
                    # Compute the filter values all at once
                    # We make it half to save on space - reinflate to float32 here, otherwise you get "not implemented" exceptions (at least on cpu, not sure atm on cuda)
                    filter_results_raw = pattern_filter_to_train(input_states_subset.float())
                    filter_means = filter_results_raw[:, 0]
                    filter_errors = filter_results_raw[:, 1]

                    target = rewards_received_subset.squeeze(1)
                    target_errors = torch.abs(rewards_received_subset.squeeze(1) - filter_means)  # **2 makes it smaller (if < 1), so harder to model accurately.
                    importance_weights_subset = importance_weights_subset.squeeze(1)
                    uncertainty_scale_subset = uncertainty_scale_subset.squeeze(1)

                    assert target.shape == filter_means.shape, "Mismatched shape (means), which may result in pytorch miscomputing the loss."
                    assert target_errors.shape == filter_errors.shape, "Mismatched shape (errors), which may result in pytorch miscomputing the loss."
                    assert importance_weights_subset.shape == target.shape, "Mismatched shape (importance weights to target), which may result in pytorch miscomputing the loss."
                    assert importance_weights_subset.shape == target_errors.shape, "Mismatched shape (importance weights to target_errors), which may result in pytorch miscomputing the loss."
                    assert uncertainty_scale_subset.shape == target_errors.shape, "Mismatched shape (uncertainty_scale), which may result in pytorch miscomputing the loss."

                    error = (importance_weights_subset.detach() * (target.detach() - filter_means) ** 2).mean(dim=0) + \
                            (uncertainty_scale_subset.detach() * importance_weights_subset.detach() * (
                                        target_errors.detach() - filter_errors) ** 2).mean(dim=0)

                    filt_stats = filter_means.cpu().detach().numpy()
                    reward_stats = rewards_received_subset.cpu().detach().numpy()
                    cls.train_logger(hypothesis).info(
                        f"Train pattern filter for {hypothesis.friendly_name} had loss {error} "
                        f"and filt min {filt_stats.min()} and max {filt_stats.max()} "
                        f"and reward min {reward_stats.min()} and max {reward_stats.max()}."
                        f"Num_neg: {len(negative_replay_entries)}")
                    # Update the nets
                    optimizer.zero_grad()
                    error.backward()
                    optimizer.step()

                    del input_states_subset
                    del rewards_received_subset
                    del filter_results_raw
                    del filter_means
                    del filter_errors

                    gc.collect()
                    # torch.cuda.empty_cache()  # TODO: hangs here sometimes

    @classmethod
    def try_train_pattern_filter(cls, hypothesis, num_samples, id_start_frac, id_end_frac, num_times_to_train):
        try:  # TODO: cleanup
            train_scale = 1  # hypothesis.replay_entries_since_last_train // (len(hypothesis._replay_buffer) + 1) + 1
            num_times_to_train *= train_scale
            hypothesis.replay_entries_since_last_train = 0

            cls.train_logger(hypothesis).info(f"Training {hypothesis.friendly_name} {num_times_to_train} times")

            for _ in range(num_times_to_train):
                cls.train_pattern_filter(hypothesis, num_samples, id_start_frac, id_end_frac)

        except Exception as e:
            cls.train_logger(hypothesis).error(f"{hypothesis.friendly_name} failed with exception {e}")
            traceback.print_exc()  # TODO: get this into the logger?
            raise e