import copy
from continual_rl.policies.sane.hypothesis.hypothesis import Hypothesis


class LongTermGateHypothesis(Hypothesis):
    """
    The Long Term Gate has the pattern filter that is intended to capture the behavior of its STs.
    It also contains the prototype hypothesis that will be used when we need to create a new ST from this LT

    This is subclassed to Hypothesis so we get all of the pattern_filter functionality
    """

    def __init__(self, source_hypothesis, parent_hypothesis):

        super().__init__(config=source_hypothesis._config,
                         device=source_hypothesis._device,
                         master_device=source_hypothesis._master_device,
                         output_dir=source_hypothesis._output_dir,
                         input_space=source_hypothesis._input_space,
                         output_size=source_hypothesis._output_size,
                         replay_buffer_size=source_hypothesis._replay_buffer_size,
                         filter_learning_rate=source_hypothesis._filter_learning_rate,
                         pattern_filter=copy.deepcopy(source_hypothesis.pattern_filter),
                         policy=copy.deepcopy(source_hypothesis._policy),
                         layer_id=source_hypothesis.layer_id,
                         parent_hypothesis=parent_hypothesis)

        # Long-term only params
        self.prototype = source_hypothesis
        self.prototype.is_prototype = True
        self.prototype._policy = None

        if self.prototype.parent_hypothesis is not None:
            self.prototype.parent_hypothesis.remove_short_term(self.prototype)  # "self" causes recursion problems, TODO

        self.short_term_versions = []
        self.is_long_term = True

    def add_short_term(self, short_term_hypothesis):
        assert short_term_hypothesis.parent_hypothesis is None, f"Attempted to add short term {short_term_hypothesis.friendly_name} to parent {self.friendly_name} when it was still attached to parent {short_term_hypothesis.parent_hypothesis.friendly_name}"
        self.short_term_versions.append(short_term_hypothesis)
        short_term_hypothesis.parent_hypothesis = self

    def remove_short_term(self, short_term_hypothesis):
        self.short_term_versions.remove(short_term_hypothesis)

        # Adding the check so we can do add/remove in either order... though add_short_term currently checks (TODO)
        if short_term_hypothesis.parent_hypothesis == self:
            short_term_hypothesis.parent_hypothesis = None

    @property
    def all_hypotheses(self):
        """
        Note that this does not return the prototype, because doing so was causing me confusion.
        """
        yield self

        for short_term_entry in self.short_term_versions:
            if isinstance(short_term_entry, LongTermGateHypothesis):
                yield from short_term_entry.all_hypotheses
            else:
                yield short_term_entry