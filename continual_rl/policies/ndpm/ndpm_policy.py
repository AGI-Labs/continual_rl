import torch
from torch.utils.tensorboard.writer import SummaryWriter
from torch.distributions.categorical import Categorical
import numpy as np
from numpy import random
from continual_rl.experiments.environment_runners.environment_runner_sync import EnvironmentRunnerSync
from continual_rl.policies.policy_base import PolicyBase
from continual_rl.policies.ndpm.ndpm_policy_config import NdpmPolicyConfig
from continual_rl.policies.ndpm.ndpm_timestep_data import NdpmTimestepData
from models.ndpm_model import NdpmModel


class NdpmPolicy(PolicyBase):
    """
    NDPM is not a reinforcement learning policy, but rather a classification one. Putting it in here to do
    classification comparisons.
    compute_action will do inference, but isn't involved, really, in the training step...which again, is doing
    classification.
    """
    def __init__(self, config: NdpmPolicyConfig, observation_space, action_spaces):  # Switch to your config type
        super().__init__()
        self._config = config
        self._model = self._create_ndpm_model(config)
        self._action_spaces = action_spaces
        self._observation_space = observation_space
        self._step = 0

    def _create_ndpm_model(self, config):
        ndpm_config = config.__dict__
        summary_writer = SummaryWriter(log_dir=self._config.output_dir)
        model = NdpmModel(ndpm_config, summary_writer)
        return model

    def get_environment_runner(self, task_spec):
        runner = EnvironmentRunnerSync(policy=self, timesteps_per_collection=self._config.batch_size,
                                       render_collection_freq=self._config.render_collection_freq,
                                       output_dir=self._config.output_dir)
        return runner

    def compute_action(self, observation, task_id, action_space_id, last_timestep_data, eval_mode):
        try:
            action_logits = self._model(observation)
            if eval_mode:
                action = action_logits.argmax(dim=1).unsqueeze(0).cpu()
            else:
                action = Categorical(logits=action_logits).sample().unsqueeze(0).cpu()
        except RuntimeError as e:
            # Learn needs to be called sufficiently to create a second expert, I think, before we can do inference.
            assert "no expert to run on the input" in str(e)

            task_action_count = self._action_spaces[action_space_id].n
            action = random.choice(range(task_action_count), 1)

        return action, NdpmTimestepData(observation)

    def train(self, storage_buffer):
        xs = []
        ys = []
        for entry in storage_buffer[0]:
            normalized_xs = torch.Tensor(entry.observation.squeeze(0).squeeze(0).float()) / self._observation_space.high.max()
            xs.append(normalized_xs)
            ys.append(torch.Tensor(np.expand_dims(entry.info[0]["correct_action"], 0)).long().squeeze(0))  # Hacky because Tensor() is weird with 0-dim np arrays

        xs = torch.stack(xs)
        ys = torch.stack(ys)
        self._model.learn(xs, ys, t=None, step=self._step)  # t doesn't seem used?
        self._step += 1

    def save(self, output_path_dir, task_id, task_total_steps):
        pass

    def load(self, model_path):
        pass
