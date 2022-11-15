import torch
import os
from continual_rl.policies.policy_base import PolicyBase
from continual_rl.policies.ppo.ppo_policy_config import PPOPolicyConfig
from continual_rl.policies.ppo.ppo_timestep_data import PPOTimestepData
from continual_rl.policies.ppo.a2c_ppo_acktr_gail.ppo import PPO
from continual_rl.policies.ppo.a2c_ppo_acktr_gail.model import Policy
from continual_rl.policies.ppo.a2c_ppo_acktr_gail.storage import RolloutStorage
from continual_rl.experiments.environment_runners.environment_runner_batch import EnvironmentRunnerBatch
from continual_rl.utils.utils import Utils
import continual_rl.policies.ppo.a2c_ppo_acktr_gail.utils as utils


class PPOPolicy(PolicyBase):
    """
    A simple implementation of policy as a sample of how policies can be created.
    Refer to policy_base itself for more detailed descriptions of the method signatures.

    Some of the code in this file is adapted from:
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/84a7582477fb0d5c82ad6d850fe476829dddd2e1/main.py

    This method is NOT multi-headed. I.e. if the tasks have mismatched action spaces, the biggest one is used,
    and the rest are subsets.
    """
    def __init__(self, config: PPOPolicyConfig, observation_space, action_spaces):  # Switch to your config type
        super().__init__(config)
        max_action_space = Utils.get_max_discrete_action_space(action_spaces)
        self._action_spaces = action_spaces

        # Original observation_space is [time, channels, width, height]
        # Compact it into [time * channels, width, height]
        observation_size = observation_space.shape
        compressed_observation_size = [observation_size[0] * observation_size[1], observation_size[2], observation_size[3]]
        self._config = config
        self._device = torch.device("cuda:0" if self._config.cuda else "cpu")

        self._actor_critic = Policy(obs_shape=compressed_observation_size,
                                    action_space=max_action_space)
        self._actor_critic.to(self._device)

        self._rollout_storage = RolloutStorage(num_steps=config.num_steps,
                                               num_processes=config.num_processes,
                                               obs_shape=compressed_observation_size,
                                               action_space=max_action_space,
                                               recurrent_hidden_state_size=self._actor_critic.recurrent_hidden_state_size)
        self._rollout_storage.to(self._device)

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
        self._step_id = 0  # What collection step we're at, in the current num_steps size collection
        self._train_step_id = 0  # How many times we've trained

    def get_environment_runner(self, task_spec):
        # See note in policy_base.get_environment_runner
        num_parallel_envs = 1 if task_spec.eval_mode else self._config.num_processes

        # Since this method is using a shared memory storage (self._rollout_storage), FullParallel cannot be supported.
        # To support it, move to using only what is returned in TimestepData from compute_action
        runner = EnvironmentRunnerBatch(policy=self, num_parallel_envs=num_parallel_envs,
                                        timesteps_per_collection=self._config.num_steps,
                                        render_collection_freq=self._config.render_collection_freq,
                                        output_dir=self._config.output_dir)
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

        if self._config.clip_reward:
            rewards = torch.sign(rewards)

        # The codebase being used expects the resultant observation, not the producer observation.
        self._rollout_storage.insert(observation, last_timestep_data.recurrent_hidden_states,
                                     last_timestep_data.actions, last_timestep_data.action_log_probs,
                                     last_timestep_data.values, rewards, masks, bad_masks)

    def _update_learning_rate(self):
        if self._config.use_linear_lr_decay:
            num_updates = self._config.decay_over_steps // (self._config.num_steps * self._config.num_processes)

            # decrease learning rate linearly
            utils.update_linear_schedule(
                self._ppo_trainer.optimizer, self._train_step_id, num_updates, self._config.learning_rate)

    def compute_action(self, observation, task_id, action_space_id, last_timestep_data, eval_mode):
        action_space = self._action_spaces[action_space_id]

        # The observation now includes the batch
        observation = observation.view((observation.shape[0], -1, observation.shape[3], observation.shape[4]))

        # Insert the previous step's data, now that it has been populated with reward and done
        if last_timestep_data is not None:
            self._update_rollout_storage(observation, last_timestep_data)

        # We could get this from the timestep data itself, but doing it this way for consistency with the original
        # codebase (a2c_ppo_acktr_gail)
        observation = self._rollout_storage.obs[self._step_id]
        recurrent_hidden_state = self._rollout_storage.recurrent_hidden_states[self._step_id]
        masks = self._rollout_storage.masks[self._step_id]

        with torch.no_grad():
            value, action, action_log_prob, recurrent_hidden_states = \
                self._actor_critic.act(observation, recurrent_hidden_state, masks, action_space=action_space)

        timestep_data = PPOTimestepData(observation=observation, recurrent_hidden_states=recurrent_hidden_states,
                                        actions=action, action_log_probs=action_log_prob, values=value,
                                        action_space=action_space)

        self._step_id = (self._step_id + 1) % self._config.num_steps

        return action, timestep_data

    def train(self, storage_buffer):
        self._update_learning_rate()

        with torch.no_grad():
            next_value = self._actor_critic.get_value(
                self._rollout_storage.obs[-1], self._rollout_storage.recurrent_hidden_states[-1],
                self._rollout_storage.masks[-1]).detach()

        self._rollout_storage.compute_returns(next_value, self._config.use_gae, self._config.gamma,
                                 self._config.gae_lambda, self._config.use_proper_time_limits)

        # Initial experiments seem to indicate that truncating the evaluate_action using the action_space
        # makes learning worse. So disabling it by setting action_space to None.
        value_loss, action_loss, dist_entropy = self._ppo_trainer.update(self._rollout_storage,
                                                                         action_space=None)
        self._rollout_storage.after_update()
        self._train_step_id += 1

        logs = [{"type": "scalar", "tag": "value_loss", "value": value_loss},
                {"type": "scalar", "tag": "action_loss", "value": action_loss},
                {"type": "scalar", "tag": "dist_entropy", "value": dist_entropy}]
        return logs

    def save(self, output_path_dir, cycle_id, task_id, task_total_steps):
        checkpoint_data = {
                "model_state_dict": self._actor_critic.state_dict(),
                "optimizer_state_dict": self._ppo_trainer.optimizer.state_dict(),
            }
        model_path = os.path.join(output_path_dir, "actor_critic.pt")
        torch.save(checkpoint_data, model_path)

    def load(self, output_path_dir):
        model_path = os.path.join(output_path_dir, "actor_critic.pt")
        if os.path.exists(model_path):
            checkpoint_data = torch.load(model_path)
            self._actor_critic.load_state_dict(checkpoint_data["model_state_dict"])
            self._ppo_trainer.optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
