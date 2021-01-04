from continual_rl.policies.sane.hypothesis_directory.utils import Utils


class CoreAccessor(object):
    """
    Due to the quantity and structure of the hypotheses, the hypothesis gets passed into each of these
    (ie, no state on this object).
    """

    @classmethod
    def core_logger(cls, hypothesis):
        logger = Utils.create_logger(f"{hypothesis._output_dir}/hypothesis_{hypothesis.friendly_name}_core.log")
        return logger

    @classmethod
    def increment(cls, hypothesis, count):
        # From USAGE -> CORE (Core side)
        hypothesis.usage_count += count
        hypothesis.usage_count_since_last_update += count
        hypothesis.non_decayed_usage_count += count
        hypothesis.usage_count_since_creation += count

    @classmethod
    def load_pattern_filter_from_state_dict(cls, hypothesis, new_pattern_filter_state_dict):
        # TODO remove this method, I hate it
        cls.core_logger(hypothesis).info("Hypothesis: loading state dict")
        hypothesis.pattern_filter.load_state_dict(new_pattern_filter_state_dict)

        cls.core_logger(hypothesis).info(f"Hypothesis: to device {hypothesis._device}")
        hypothesis.pattern_filter.to(hypothesis._device)  # Needs to happen before the ReplayBuffer initialization (if I use the encoder)

        cls.core_logger(hypothesis).info("Hypothesis: pattern filter load complete")
