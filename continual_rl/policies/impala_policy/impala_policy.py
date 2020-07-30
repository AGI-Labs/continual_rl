import numpy as np
import torch
from continual_rl.policies.policy_base import PolicyBase
from continual_rl.policies.impala_policy.impala_policy_config import ImpalaPolicyConfig
from continual_rl.experiments.environment_runners.full_parallel.environment_runner_full_parallel import EnvironmentRunnerFullParallel
from torchbeast.monobeast import AtariNet


class ImpalaPolicy(PolicyBase):
    """
    A simple implementation of policy as a sample of how policies can be created.
    Refer to policy_base itself for more detailed descriptions of the method signatures.
    """
    def __init__(self, config: ImpalaPolicyConfig, observation_size, action_spaces):  # Switch to your config type
        super().__init__()
        # Naively for now just taking the maximum, rather than having multiple heads
        common_action_size = np.array(list(action_spaces.values())).max()
        self._actor = AtariNet(observation_size, common_action_size, config.use_lstm)
        self._learner_model = AtariNet(observation_size, common_action_size, config.use_lstm)

        self._optimizer = torch.optim.RMSprop(
            self._learner_model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            eps=config.epsilon,
            alpha=config.alpha,
        )

    def get_environment_runner(self):
        runner = EnvironmentRunnerFullParallel(self, num_parallel_processes=3, timesteps_per_collection=1000,
                                               render_collection_freq=50)
        return runner

    def compute_action(self, observation, task_id, last_info_to_store):
        pass

    def train(self, storage_buffer):
        pass

    def save(self, output_path_dir, task_id, task_total_steps):
        pass

    def load(self, model_path):
        pass
