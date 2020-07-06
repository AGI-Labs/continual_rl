from numpy import random
from continual_rl.policies.policy_base import PolicyBase
from continual_rl.policies.random_policy.random_policy_config import RandomPolicyConfig  # Switch to your config type
from continual_rl.experiments.environment_runners.environment_runner_sync import EnvironmentRunnerSync
from continual_rl.experiments.environment_runners.environment_runner_batch import EnvironmentRunnerBatch


class RandomPolicy(PolicyBase):
    """
    A simple implementation of policy as a sample of how policies can be created.
    Refer to policy_base itself for more detailed descriptions of the method signatures.
    """
    def __init__(self, config: RandomPolicyConfig):
        super().__init__()
        self._config = config

    def get_environment_runner(self):
        if self._config.num_parallel_envs is None:
            runner = EnvironmentRunnerSync(policy=self, timesteps_per_collection=self._config.timesteps_per_collection)
        else:
            runner = EnvironmentRunnerBatch(policy=self, num_parallel_envs=self._config.num_parallel_envs,
                                            timesteps_per_collection=self._config.timesteps_per_collection)
        return runner

    def compute_action(self, observation, task_action_count):
        if self._config.num_parallel_envs is None:
            action = random.choice(range(task_action_count)), {}
        else:
            action = random.choice(range(task_action_count), self._config.num_parallel_envs), {}

        return action

    def train(self, storage_buffer):
        pass

    def save(self, output_path_dir, task_id, task_total_steps):
        pass

    def load(self, model_path):
        pass
