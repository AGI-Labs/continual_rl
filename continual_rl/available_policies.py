from continual_rl.policies.ppo_policy.ppo_policy import PPOPolicy
from continual_rl.policies.ppo_policy.ppo_policy_config import PPOPolicyConfig
from continual_rl.policies.discrete_random_policy.discrete_random_policy import DiscreteRandomPolicy
from continual_rl.policies.discrete_random_policy.discrete_random_policy_config import DiscreteRandomPolicyConfig


class PolicyStruct(object):
    def __init__(self, policy, config):
        self.policy = policy
        self.config = config


def get_available_policies():
    """
    The registry of policies that are available for ease of use. To create your own, duplicate prototype_policy's
    folder, populate it (reference policy_base.py as necessary), and add it here.
    """
    policies = {"discrete_random" : PolicyStruct(DiscreteRandomPolicy, DiscreteRandomPolicyConfig),
                "PPO": PolicyStruct(PPOPolicy, PPOPolicyConfig)}

    return policies
