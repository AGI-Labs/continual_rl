from numpy import random
from continual_rl.policies.policy_base import PolicyBase
from continual_rl.policies.discrete_random.discrete_random_policy_config import DiscreteRandomPolicyConfig
from continual_rl.policies.discrete_random.discrete_random_timestep_data import DiscreteRandomTimestepData
from continual_rl.experiments.environment_runners.environment_runner_sync import EnvironmentRunnerSync
from continual_rl.experiments.environment_runners.environment_runner_batch import EnvironmentRunnerBatch


class DiscreteRandomPolicy(PolicyBase):
    """
    A simple implementation of policy as a sample of how policies can be created.
    Refer to policy_base itself for more detailed descriptions of the method signatures.
    """
    def __init__(self, config: DiscreteRandomPolicyConfig, observation_space, action_spaces):
        super().__init__(config)
        self._config = config
        self._action_spaces = action_spaces

    def get_environment_runner(self, task_spec):
        if self._config.num_parallel_envs is None:
            runner = EnvironmentRunnerSync(policy=self, timesteps_per_collection=self._config.timesteps_per_collection)
        else:
            runner = EnvironmentRunnerBatch(policy=self, num_parallel_envs=self._config.num_parallel_envs,
                                            timesteps_per_collection=self._config.timesteps_per_collection,
                                            output_dir=self._config.output_dir)
        return runner

    def compute_action(self, observation, task_id, action_space_id, last_timestep_data, eval_mode):
        task_action_count = self._action_spaces[action_space_id].n

        if self._config.num_parallel_envs is None:
            action = random.choice(range(task_action_count), 1)  # Even sync expects a list of actions
        else:
            action = random.choice(range(task_action_count), self._config.num_parallel_envs)

        return action, DiscreteRandomTimestepData()

    def train(self, storage_buffer):
        pass

    def save(self, output_path_dir, cycle_id, task_id, task_total_steps):
        pass

    def load(self, output_path_dir):
        pass
