import random
from continual_rl.policies.policy_base import PolicyBase
from continual_rl.policies.random_policy.random_policy_config import RandomPolicyConfig  # Switch to your config type
from continual_rl.experiments.episode_runners.episode_runner_sync import EpisodeRunnerSync


class RandomPolicy(PolicyBase):
    """
    A simple implementation of policy as a sample of how policies can be created.
    Refer to policy_base itself for more detailed descriptions of the method signatures.
    """
    def __init__(self, config: RandomPolicyConfig):  # Switch to your config type
        super().__init__()
        self._config = config

    def get_episode_runner(self):
        return EpisodeRunnerSync(policy=self, timesteps_per_collection=self._config.timesteps_per_collection)

    def compute_action(self, observation, task_action_count):
        return random.choice(task_action_count), {}

    def train(self, storage_buffer):
        pass

    def save(self, output_path_dir, task_id, task_total_steps):
        pass

    def load(self, model_path):
        pass
