import torch
from continual_rl.policies.policy_base import PolicyBase
from continual_rl.policies.ppo_policy.ppo_policy_config import PPOPolicyConfig
from continual_rl.experiments.environment_runners.environment_runner_batch import EnvironmentRunnerBatch
from continual_rl.policies.ppo_policy.actor_critic_model import ActorCritic


class PPOPolicy(PolicyBase):
    """
    Basically a wrapper around torch-ac's implementation of PPO
    """
    def __init__(self, config: PPOPolicyConfig, observation_size, action_size):
        super().__init__()
        self._config = config
        self._model = ActorCritic(action_space=action_size)

        pass

    def get_environment_runner(self):
        runner = EnvironmentRunnerBatch(policy=self, num_parallel_envs=self._config.num_parallel_envs,
                                        timesteps_per_collection=self._config.timesteps_per_collection)
        return runner

    def compute_action(self, observation, task_action_count):
        # The input observation is [time, batch, C, W, H]
        # We convert to [batch, time * C, W, H]
        rearranged_observation = observation.permute(1, 0, 2, 3, 4)
        compacted_observation = observation.view(rearranged_observation.shape[0], -1, *rearranged_observation.shape[3:])

        action_distribution, value = self._model(compacted_observation, task_action_count)
        actions = action_distribution.sample()

        return actions, {}

    def train(self, storage_buffer):
        pass

    def save(self, output_path_dir, task_id, task_total_steps):
        pass

    def load(self, model_path):
        pass
