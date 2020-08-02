import numpy as np
import torch
from continual_rl.policies.policy_base import PolicyBase
from continual_rl.policies.impala.impala_policy_config import ImpalaPolicyConfig
from continual_rl.policies.impala.impala_timestep_data import ImpalaTimestepData
from continual_rl.experiments.environment_runners.full_parallel.environment_runner_full_parallel import EnvironmentRunnerFullParallel
from torchbeast.monobeast import AtariNet


class ImpalaPolicy(PolicyBase):
    """
    A simple implementation of policy as a sample of how policies can be created.
    Refer to policy_base itself for more detailed descriptions of the method signatures.
    """
    def __init__(self, config: ImpalaPolicyConfig, observation_size, action_spaces):  # Switch to your config type
        super().__init__()
        self._config = config

        # Naively for now just taking the maximum, rather than having multiple heads
        common_action_size = int(np.array(list(action_spaces.values())).max())
        self._actor = AtariNet(observation_size, common_action_size, config.use_lstm)
        self._learner_model = AtariNet(observation_size, common_action_size, config.use_lstm)

        # Learner gets trained, actor gets updated with the results periodically
        self._actor.share_memory()

        self._optimizer = torch.optim.RMSprop(
            self._learner_model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            eps=config.epsilon,
            alpha=config.alpha,
        )

    def get_environment_runner(self):
        runner = EnvironmentRunnerFullParallel(self, num_parallel_processes=self._config.num_actors,
                                               timesteps_per_collection=self._config.unroll_length,
                                               render_collection_freq=10000)
        return runner

    def compute_action(self, observation, task_id, last_timestep_data):
        # Input is (B, T, C, W, H), AtariNet says it expects (T, B, C, W, H), but it looks like it expects (B, 1, T*C, W, H)?
        # If our handling of frames is wildly inefficient, look at torchbeast's LazyFrames
        observation = observation.view((observation.shape[0], 1, -1, *observation.shape[3:]))

        if last_timestep_data is None:
            agent_state = self._actor.initial_state(batch_size=1)
            last_action = torch.Tensor([[0]]).to(torch.int64)
            reward = torch.Tensor([[0]])
        else:
            # I don't think whether an episode has finished (done=True) has any bearing on this
            agent_state = last_timestep_data.agent_state
            last_action = last_timestep_data.action
            reward = torch.Tensor(last_timestep_data.reward)  # Env gives it to us as numpy, so convert it

        model_input = {"frame": observation,
                       "last_action": last_action,
                       "reward": reward}

        with torch.no_grad():
            agent_output, agent_state = self._actor(model_input, agent_state)

        action = agent_output["action"]
        timestep_data = ImpalaTimestepData(agent_state, action)

        return action, timestep_data

    def train(self, storage_buffer):
        pass

    def save(self, output_path_dir, task_id, task_total_steps):
        pass

    def load(self, model_path):
        pass
