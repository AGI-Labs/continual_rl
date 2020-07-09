from continual_rl.policies.random_policy.random_policy import DiscreteRandomPolicy
from continual_rl.policies.random_policy.random_policy_config import RandomPolicyConfig


class PolicyStruct(object):
    def __init__(self, policy, config):
        self.policy = policy
        self.config = config


def get_available_policies():
    """
    We could do this with dynamic loading, but that's more restrictive in terms of patterns we expect people to follow.
    This is a small bit more work, but requires less structure from policy implementers.
    """
    policies = {"discrete_random" : PolicyStruct(DiscreteRandomPolicy, RandomPolicyConfig)}

    return policies
