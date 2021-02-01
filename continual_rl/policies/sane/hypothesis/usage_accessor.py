from continual_rl.policies.sane.hypothesis.replay_buffer import ReplayEntry


class UsageAccessor(object):
    """
    Due to the quantity and structure of the hypotheses, the hypothesis gets passed into each of these (ie, no state on this object).
    """

    @classmethod
    def forward(self, hypothesis, x, eval_mode, counter_lock, create_replay=True):
        assert x.shape[0] == 1, "When Hypothesis forward is called, a batch size of 1 is expected."

        x = x.to(hypothesis._device)

        replay_entry = None
        pattern_filter_result, error = hypothesis.pattern_filter(x).squeeze(0).detach()

        if not eval_mode and create_replay:
            replay_entry = ReplayEntry(x.cpu())

        # TODO: keep in sync with training of policy
        policy = hypothesis.get_policy(x)
        return policy, pattern_filter_result, replay_entry
