from numpy import random
from continual_rl.policies.policy_base import PolicyBase
from continual_rl.policies.discrete_random_policy.discrete_random_policy_config import DiscreteRandomPolicyConfig
from continual_rl.policies.discrete_random_policy.discrete_random_info_to_store import DiscreteRandomInfoToStore
from continual_rl.experiments.environment_runners.environment_runner_sync import EnvironmentRunnerSync
from continual_rl.experiments.environment_runners.environment_runner_batch import EnvironmentRunnerBatch


class DiscreteRandomPolicy(PolicyBase):
    """
    A simple implementation of policy as a sample of how policies can be created.
    Refer to policy_base itself for more detailed descriptions of the method signatures.
    """
    def __init__(self, config: DiscreteRandomPolicyConfig, observation_size, action_spaces):
        super().__init__()
        self._config = config
        self._action_spaces = action_spaces

    def get_environment_runner(self):
        if self._config.num_parallel_envs is None:
            runner = EnvironmentRunnerSync(policy=self, timesteps_per_collection=self._config.timesteps_per_collection)
        else:
            runner = EnvironmentRunnerBatch(policy=self, num_parallel_envs=self._config.num_parallel_envs,
                                            timesteps_per_collection=self._config.timesteps_per_collection)
        return runner

    def compute_action(self, observation, action_space_id):
        task_action_count = self._action_spaces[action_space_id]

        if self._config.num_parallel_envs is None:
            action = random.choice(range(task_action_count))
        else:
            action = random.choice(range(task_action_count), self._config.num_parallel_envs)

        return action, DiscreteRandomInfoToStore()

    def train(self, storage_buffer):
        pass

    def save(self, output_path_dir, task_id, task_total_steps):
        pass

    def load(self, model_path):
        pass
