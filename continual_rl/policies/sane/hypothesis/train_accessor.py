import torch
import torch.optim as optim
import gc
from continual_rl.policies.sane.hypothesis_directory.utils import Utils
from continual_rl.policies.sane.hypothesis.replay_buffer import ReplayBufferDataLoaderWrapper
import copy
from torch.distributions.categorical import Categorical
import traceback
from torch.utils.data import DataLoader


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

        if hypothesis.policy is not None:  # TODO
            policy_dist = Categorical(logits=hypothesis.policy.clone())
            result = torch.exp(policy_dist.log_prob(selected_action) - orig_log_prob)  # Could be more efficient (re: e^log), but for consistency for now TODO
        else:
            result = torch.Tensor([1 for _ in range(len(selected_action))]).unsqueeze(1).unsqueeze(1)

        return result

    @classmethod
    def load_learner(cls, hypothesis):
        if hypothesis._pattern_filter_learner is not None:
            hypothesis.pattern_filter.load_state_dict(hypothesis._pattern_filter_learner.state_dict())

    @classmethod
    def train_pattern_filter(cls, hypothesis, num_samples, id_start_frac, id_end_frac):  # TODO: these ignored params
        if hypothesis._pattern_filter_learner is None:
            hypothesis._pattern_filter_learner = copy.deepcopy(hypothesis.pattern_filter)
        pattern_filter_to_train = hypothesis._pattern_filter_learner

        # Fail out quickly if we can't train (TODO: probably better to wrap the rest, but feeling lazy.)
        if len(hypothesis._replay_buffer) == 0:
            return

        if hypothesis._device.type != "cpu":
            torch.cuda.set_device(hypothesis._device)  # Making sure the optimizer creates its stuff in the right place...Otherwise we create loads of tensors on cuda:0

        optimizer = cls._get_pattern_filter_optimizer(hypothesis, hypothesis._pattern_filter_learner)
        batch_size = hypothesis._config.batch_size

        data_loader = DataLoader(ReplayBufferDataLoaderWrapper(hypothesis._replay_buffer), batch_size=batch_size, shuffle=True,
                                 pin_memory=False)
        for batch in data_loader:
            input_states, rewards_received, action_log_probs, selected_actions = batch

            # TODO: these importance weights are just at one timestep...because the full trajectory seems like a lot to keep track of/compute. But still, should consider
            # Maybe keep track of the *next* hypothesis, and grab its policy ratio for importance sampling...
            importance_weights = cls._compute_importance_weight(hypothesis, selected_actions, action_log_probs)

            # TODO: track down where these extra dims are coming from
            input_states = input_states
            rewards_received = rewards_received.squeeze(1)
            importance_weights = importance_weights.squeeze(1)

            # Storing the replay buffer not on the gpu because it can get quite big.... Though it's a tradeoff. TODO: better check
            if hypothesis._device is not None:
                input_states = input_states.to(hypothesis._device)
                rewards_received = rewards_received.to(hypothesis._device)
                importance_weights = importance_weights.to(hypothesis._device)

            # =====Train the pattern_filter network=====
            # Compute the filter values all at once
            # We make it half to save on space - reinflate to float32 here, otherwise you get "not implemented" exceptions (at least on cpu, not sure atm on cuda)
            filter_results_raw = pattern_filter_to_train(input_states.float())
            filter_means = filter_results_raw[:, 0]
            filter_errors = filter_results_raw[:, 1]

            target = rewards_received
            target_errors = torch.abs(rewards_received - filter_means)  # **2 makes it smaller (if < 1), so harder to model accurately.

            assert target.shape == filter_means.shape, "Mismatched shape (means), which may result in pytorch miscomputing the loss."
            assert target_errors.shape == filter_errors.shape, "Mismatched shape (errors), which may result in pytorch miscomputing the loss."
            assert importance_weights.shape == target.shape, "Mismatched shape (importance weights to target), which may result in pytorch miscomputing the loss."
            assert importance_weights.shape == target_errors.shape, "Mismatched shape (importance weights to target_errors), which may result in pytorch miscomputing the loss."

            error = (importance_weights.detach() * (target.detach() - filter_means) ** 2).mean(dim=0) + \
                    (importance_weights.detach() * (target_errors.detach() - filter_errors) ** 2).mean(dim=0)

            filt_stats = filter_means.cpu().detach().numpy()
            reward_stats = rewards_received.cpu().detach().numpy()
            cls.train_logger(hypothesis).info(
                f"Train pattern filter for {hypothesis.friendly_name} had loss {error} "
                f"and filt min {filt_stats.min()} and max {filt_stats.max()} "
                f"and reward min {reward_stats.min()} and max {reward_stats.max()}."
                f"Num_neg: {0}")
            # Update the nets
            optimizer.zero_grad()
            error.backward()
            optimizer.step()

            del input_states
            del rewards_received
            del filter_results_raw
            del filter_means
            del filter_errors

            gc.collect()

    @classmethod
    def try_train_pattern_filter(cls, hypothesis, num_samples, id_start_frac, id_end_frac, num_times_to_train):
        try:
            hypothesis.replay_entries_since_last_train = 0
            cls.train_logger(hypothesis).info(f"Training {hypothesis.friendly_name} {num_times_to_train} times")

            for _ in range(num_times_to_train):
                cls.train_pattern_filter(hypothesis, num_samples, id_start_frac, id_end_frac)

        except Exception as e:
            cls.train_logger(hypothesis).error(f"{hypothesis.friendly_name} failed with exception {e}")
            traceback.print_exc()  # TODO: get this into the logger?
            raise e
