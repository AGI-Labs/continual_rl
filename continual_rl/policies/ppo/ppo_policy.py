import torch
from continual_rl.policies.policy_base import PolicyBase
from continual_rl.policies.ppo.ppo_policy_config import PPOPolicyConfig
from continual_rl.policies.ppo.ppo_timestep_data import PPOTimestepData
from continual_rl.policies.ppo.a2c_ppo_acktr_gail.ppo import PPO
from continual_rl.policies.ppo.a2c_ppo_acktr_gail.model import Policy
from continual_rl.policies.ppo.a2c_ppo_acktr_gail.storage import RolloutStorage
from continual_rl.experiments.environment_runners.environment_runner_batch import EnvironmentRunnerBatch


class PPOPolicy(PolicyBase):
    """
    A simple implementation of policy as a sample of how policies can be created.
    Refer to policy_base itself for more detailed descriptions of the method signatures.

    Some of the code in this file is adapted from:
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/84a7582477fb0d5c82ad6d850fe476829dddd2e1/main.py

    """
    def __init__(self, config: PPOPolicyConfig, observation_size, action_spaces):  # Switch to your config type
        super().__init__()
        common_action_space = self._get_common_action_space(action_spaces)

        # Original observation_size is [time, channels, width, height]
        # Compact it into [time * channels, width, height]
        observation_size = observation_size.shape
        compressed_observation_size = [observation_size[0] * observation_size[1], observation_size[2], observation_size[3]]
        self._config = config
        self._actor_critic = Policy(obs_shape=compressed_observation_size,
                                    action_space=common_action_space)
        self._rollout_storage = RolloutStorage(num_steps=config.num_steps,
                                               num_processes=config.num_processes,
                                               obs_shape=compressed_observation_size,
                                               action_space=common_action_space,
                                               recurrent_hidden_state_size=self._actor_critic.recurrent_hidden_state_size)
        self._ppo_trainer = PPO(
            self._actor_critic,
            self._config.clip_param,
            self._config.ppo_epoch,
            self._config.num_mini_batch,
            self._config.value_loss_coef,
            self._config.entropy_coef,
            lr=self._config.learning_rate,
            eps=self._config.eps,
            max_grad_norm=self._config.max_grad_norm)
        self._step_id = 0

    def _get_common_action_space(self, action_spaces):
        common_action_space = None
        for action_space in action_spaces.values():
            if common_action_space is None:
                common_action_space = action_space
            assert common_action_space == action_space, \
                "PPO currently only supports environments with the same action spaces."
        return common_action_space

    def get_environment_runner(self):
        # Since this method is using a shared memory storage (self._rollout_storage), FullParallel cannot be supported.
        # To support it, move to using only what is returned in TimestepData from compute_action
        runner = EnvironmentRunnerBatch(policy=self, num_parallel_envs=self._config.num_processes,
                                        timesteps_per_collection=self._config.num_steps,
                                        render_collection_freq=self._config.render_collection_freq)
        return runner

    def _update_rollout_storage(self, observation, last_timestep_data):
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in last_timestep_data.done])

        # The original a2c_ppo_acktr_gail uses a TimeLimit gym wrapper, and that sets bad_transition
        # This is analogous to utils/env_wrappers/TimeLimit, which uses TimeLimit.truncated
        # This is not currently fully tested
        bad_masks = torch.FloatTensor(
            [[0.0] if 'TimeLimit.truncated' in info.keys() else [1.0]
             for info in last_timestep_data.info])
        rewards = torch.FloatTensor(last_timestep_data.reward).unsqueeze(1)

        # The codebase being used expects the resultant observation, not the producer observation.
        self._rollout_storage.insert(observation, last_timestep_data.recurrent_hidden_states,
                                     last_timestep_data.actions, last_timestep_data.action_log_probs,
                                     last_timestep_data.values, rewards, masks, bad_masks)

    def compute_action(self, observation, action_space_id, last_timestep_data):
        # The observation now includes the batch
        observation = observation.view((observation.shape[0], -1, observation.shape[3], observation.shape[4]))
        observation = observation * 255.0  # [0, 1] given, [0, 255] expected

        # Insert the previous step's data, now that it has been populated with reward and done
        if last_timestep_data is not None:
            self._update_rollout_storage(observation, last_timestep_data)

        # We could get this from the timestep data itself, but doing it this way for consistency with the original
        # codebase (a2c_ppo_acktr_gail)
        recurrent_hidden_state = self._rollout_storage.recurrent_hidden_states[self._step_id]
        masks = self._rollout_storage.masks[self._step_id]

        with torch.no_grad():
            value, action, action_log_prob, recurrent_hidden_states = \
                self._actor_critic.act(observation, recurrent_hidden_state, masks)

        timestep_data = PPOTimestepData(observation=observation, recurrent_hidden_states=recurrent_hidden_states,
                                        actions=action,action_log_probs=action_log_prob, values=value)

        self._step_id = (self._step_id + 1) % self._config.num_steps

        return action, timestep_data

    def train(self, storage_buffer):
        with torch.no_grad():
            next_value = self._actor_critic.get_value(
                self._rollout_storage.obs[-1], self._rollout_storage.recurrent_hidden_states[-1],
                self._rollout_storage.masks[-1]).detach()

        self._rollout_storage.compute_returns(next_value, self._config.use_gae, self._config.gamma,
                                 self._config.gae_lambda, self._config.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = self._ppo_trainer.update(self._rollout_storage)
        self._rollout_storage.after_update()

        # TODO: return logs and enable summary-writering-them
        print(f"value_loss: {value_loss}, action_loss: {action_loss}, dist_entropy: {dist_entropy}")

    def save(self, output_path_dir, task_id, task_total_steps):
        pass

    def load(self, model_path):
        pass
