from continual_rl.policies.ppo_policy.ppo_policy import PPOPolicy
from continual_rl.policies.ppo_policy.ppo_policy_config import PPOPolicyConfig


class PolicyStruct(object):
    def __init__(self, policy, config):
        self.policy = policy
        self.config = config


def get_available_policies():
    """
    We could do this with dynamic loading, but that's more restrictive in terms of patterns we expect people to follow.
    This is a small bit more work, but requires less structure from policy implementers.
    """
    policies = {"PPO": PolicyStruct(PPOPolicy, PPOPolicyConfig)}

    return policies
