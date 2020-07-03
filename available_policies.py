from policies.ppo_policy.ppo_policy import PPOPolicy


def get_available_policies():
    policies = {"PPO": PPOPolicy}

    return policies
