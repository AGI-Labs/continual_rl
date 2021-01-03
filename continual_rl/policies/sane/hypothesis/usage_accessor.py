from continual_rl.policies.sane.hypothesis.replay_buffer import ReplayEntry


class UsageAccessor(object):
    """
    Due to the quantity and structure of the hypotheses, the hypothesis gets passed into each of these (ie, no state on this object).
    """

    @classmethod
    def forward(self, hypothesis, x, eval_mode, counter_lock, create_replay=True):  # TODO: NOT local to the hypothesis process - if/when I move this into the process, also make sure parent_hypothesis gets communicated around appropriately
        assert x.shape[0] == 1, "When Hypothesis forward is called, a batch size of 1 is expected."

        x = x.to(hypothesis._device)

        replay_entry = None
        pattern_filter_result, error = hypothesis.pattern_filter(x).squeeze(0).detach()

        if not eval_mode and create_replay:
            replay_entry = ReplayEntry(x.cpu())

        # TODO: keep in sync with training of policy
        policy = hypothesis.get_policy_with_entropy(x)  # policy  # TODO: keep in sync with policy train()

        # TODO: this doesn't seem to be doing what I want - moved it into the policy itself, because the policy training
        # was re-grabbing the policy.
        #if (hypothesis.is_prototype or hypothesis.remain_random) and policy is not None: # or hypothesis.parent_hypothesis is None:  # TODO: calling out this parent hypothesis thing - if we're an LT, don't update the policy
        #    policy = policy.detach()

        return policy, pattern_filter_result, replay_entry