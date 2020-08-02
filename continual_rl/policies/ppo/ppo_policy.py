import torch
from torch_ac.algos.ppo import PPOAlgo
from torch_ac.utils.dictlist import DictList
import numpy as np
from continual_rl.policies.policy_base import PolicyBase
from continual_rl.policies.ppo.ppo_policy_config import PPOPolicyConfig
from continual_rl.policies.ppo.ppo_timestep_data import PPOTimestepDataBatch
from continual_rl.experiments.environment_runners.environment_runner_batch import EnvironmentRunnerBatch
from continual_rl.policies.ppo.actor_critic_model import ActorCritic


class PPOParent(PPOAlgo):
    """
    Our goal in this file is to use torch_ac's implementation of PPO.
    Unfortunately PPOAlgo does a number of things we do not want (e.g. spins up environments).
    More specifically, we only want the function update_parameters on PPOAlgo, not the abilities of its base class, so
    subclass it but intentionally don't call super(). Instead do only the parts of super we do care about manually
    (specifically set variables).
    """
    def __init__(self, config, model):
        # Intentionally not calling super() because I do not want the normal initialization to be executed
        # Specifically, PPOAlgo's init calls BaseAlgo's init which initializes the environments. Since Policy is
        # intended to be isolated from the envs, I override like this.

        self.clip_eps = config.clip_eps
        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.recurrence = 1  # Number of timesteps over which the gradient is propagated. Recurrency not currently supported.
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=config.adam_eps)
        self.acmodel = model
        self.entropy_coef = config.entropy_coef
        self.value_loss_coef = config.value_loss_coef
        self.max_grad_norm = config.max_grad_norm
        self.num_frames_per_proc = config.timesteps_per_collection
        self.num_frames = self.num_frames_per_proc * config.num_parallel_envs

        # Internal counter
        self.batch_num = 0


class PPOPolicy(PolicyBase):
    """
    Basically a wrapper around torch-ac's implementation of PPO
    """
    def __init__(self, config: PPOPolicyConfig, observation_size, action_spaces):
        super().__init__()
        self._config = config
        self._action_spaces = action_spaces

        # For this current simple implementation we just use the maximum action for our network, and extract the
        # subset necessary for a given task. The natural alternative is to have several different heads, one per
        # task.
        common_action_size = np.array(list(action_spaces.values())).max()

        # Due to the manipulation we do in compute_action, the observation_size is not exactly as input
        # Note that observation size does not include batch size
        observation_size = [observation_size[0] * observation_size[1], *observation_size[2:]]
        self._model = ActorCritic(observation_space=observation_size, action_space=common_action_size)

        if config.use_cuda:
            self._model.cuda()

        self._model.share_memory()  # Necessary for FullParallel
        self._ppo_trainer = PPOParent(config, self._model)

    def get_environment_runner(self):
        runner = EnvironmentRunnerBatch(policy=self, num_parallel_envs=self._config.num_parallel_envs,
                                        timesteps_per_collection=self._config.timesteps_per_collection,
                                        render_collection_freq=self._config.render_collection_freq)

        return runner

    def compute_action(self, observation, action_space_id, last_timestep_data):
        if self._config.use_cuda:
            observation = observation.cuda()

        task_action_count = self._action_spaces[action_space_id]

        # The input observation is [batch, time, C, W, H]
        # We convert to [batch, time * C, W, H]
        compacted_observation = observation.view(observation.shape[0], -1, *observation.shape[3:])

        # Collect the data and generate the action
        action_distribution, values = self._model(compacted_observation, task_action_count)
        actions = action_distribution.sample()
        log_probs = action_distribution.log_prob(actions)

        timestep_data = PPOTimestepDataBatch(compacted_observation, actions.detach(), values.detach(),
                                            log_probs.detach(), task_action_count)

        return actions.cpu(), timestep_data

    def train(self, storage_buffer):
        # PPOAlgo assumes the model forward only accepts observation, so doing this for now
        task_action_count = storage_buffer[0][0].task_action_count
        self._model.set_task_action_count(task_action_count)

        experiences = self._convert_to_ppo_experiences(storage_buffer)
        logs = self._ppo_trainer.update_parameters(experiences)

        # Would rather fail fast if something bad happens than to use the wrong action_count somehow
        self._model.set_task_action_count(None)

        print(logs)

    def save(self, output_path_dir, task_id, task_total_steps):
        pass

    def load(self, model_path):
        pass

    def _compute_advantages(self, timestep_datas):
        """
        Input should be a list of timestep_datas in order by time (0..T) all for the same environment.
        """
        # Compute the predicted value of the last entry
        with torch.no_grad():
            _, next_value = self._model(timestep_datas[-1].observation.unsqueeze(0), timestep_datas[-1].task_action_count)
            next_value = next_value.squeeze(0)  # Remove the batch

        next_advantage = 0

        # The final output container for the computed advantages, in the same order as timestep_datas
        advantages = [None for _ in range(len(timestep_datas))]

        for entry_id, info_entry in reversed(list(enumerate(timestep_datas))):
            if info_entry.done:
                next_value = 0
                next_advantage = 0

            delta = info_entry.reward + self._config.discount * next_value - info_entry.value
            advantages[entry_id] = delta + self._config.discount * self._config.gae_lambda * next_advantage
            next_value = info_entry.value
            next_advantage = advantages[entry_id]

        return advantages

    def _convert_to_ppo_experiences(self, storage_buffers):
        """
        Format the experiences collected in the form expected by torch_ac
        """
        # storage_buffer contains #processes x timesteps_collected_per_env entries of PPOTimestepDataBatch
        # Each batch stores multiple environments' worth of data.
        # Group the data instead by environment, which is more meaningful.
        all_env_sorted_timestep_datas = []
        for storage_buffer in storage_buffers:
            env_sorted_timestep_datas = [timestep_data.regroup_by_env() for timestep_data in storage_buffer]
            condensed_env_sorted = list(zip(*env_sorted_timestep_datas))
            all_env_sorted_timestep_datas.extend(condensed_env_sorted)

        all_observations = []
        all_actions = []
        all_values = []
        all_rewards = []
        all_advantages = []
        all_log_probs = []

        for env_data in all_env_sorted_timestep_datas:
            env_data = [data.to_tensors(self._config.use_cuda) for data in env_data]

            all_observations.append([entry.observation for entry in env_data])
            all_actions.append(torch.stack([entry.action for entry in env_data]))
            all_values.append(torch.stack([entry.value for entry in env_data]))
            all_rewards.append(torch.stack([entry.reward for entry in env_data]))
            all_advantages.append(torch.stack(self._compute_advantages(env_data)))
            all_log_probs.append(torch.stack([entry.log_prob for entry in env_data]))

        # torch_ac's experiences expect [num_envs, timesteps_per_collection] -> [num_envs * timesteps_per_collection]
        # Thanks to torch_ac for this PPO implementation - LICENSE available as a sibling to this file
        experiences = DictList()
        experiences.obs = torch.stack([all_observations[j][i]
                                       for j in range(len(all_observations))
                                       for i in range(len(all_observations[0]))]).detach()
        experiences.action = torch.stack(all_actions).reshape(-1).detach()
        experiences.value = torch.stack(all_values).reshape(-1).detach()
        experiences.reward = torch.stack(all_rewards).reshape(-1).detach()
        experiences.advantage = torch.stack(all_advantages).reshape(-1).detach()
        experiences.log_prob = torch.stack(all_log_probs).reshape(-1).detach()

        experiences.returnn = experiences.value + experiences.advantage

        return experiences
