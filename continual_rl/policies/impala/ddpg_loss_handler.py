import torch
import torch.nn as nn
from torch.nn import functional as F
from continual_rl.policies.impala.torchbeast.core import vtrace


class DdpgLossHandler(object):
    """
    The default loss handler for IMPALA
    """
    def __init__(self, model_flags, learner_model):
        self._critic_optimizer = self._create_optimizer(model_flags, learner_model.critic_parameters(), model_flags.learning_rate)
        self._actor_optimizer = self._create_optimizer(model_flags, learner_model.actor_parameters(), model_flags.actor_learning_rate)
        self.optimizer = self._create_optimizer(model_flags, learner_model.parameters(), model_flags.learning_rate)  # Used for custom losses (TODO?)

        self._scheduler_state_dict = None  # Filled if we load()
        self._scheduler = None  # Task-specific, so created there

        self._learner_model = learner_model
        self._model_flags = model_flags

        # Exponential moving average
        avg_fn = lambda averaged_model_parameter, model_parameter, num_averaged: \
            (1 - model_flags.target_learner_tau) * averaged_model_parameter + model_flags.target_learner_tau * model_parameter

        self._target_learner_model = torch.optim.swa_utils.AveragedModel(learner_model, avg_fn=avg_fn)
        self._demonstration_mode = False

    def _create_optimizer(self, model_flags, parameters, learning_rate):
        if model_flags.optimizer == "rmsprop":
            optimizer = torch.optim.RMSprop(
                parameters,
                lr=learning_rate,
                momentum=model_flags.momentum,
                eps=model_flags.epsilon,
                alpha=model_flags.alpha,
            )
        elif model_flags.optimizer == "adam":
            optimizer = torch.optim.Adam(
                parameters,
                lr=learning_rate,
            )
        else:
            raise ValueError(f"Unsupported optimizer type {model_flags.optimizer}.")

        return optimizer

    def get_save_data(self):
        checkpoint_data = {
                "optimizer_state_dict": self.optimizer.state_dict(),
            }

        if self._scheduler is not None:
            checkpoint_data["scheduler_state_dict"] = self._scheduler.state_dict()

        return checkpoint_data

    def load_save_data(self, checkpoint):
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self._model_flags.use_scheduler:
            self._scheduler_state_dict = checkpoint.get("scheduler_state_dict", None)

            if self._scheduler_state_dict is None:
                # Tracked by issue #109
                self.logger.warn("No scheduler state dict found to load when one was expected.")

    def initialize_for_task(self, task_flags):
        T = self._model_flags.unroll_length
        B = self._model_flags.batch_size

        def lr_lambda(epoch):
            return 1 - min(epoch * T * B, task_flags.total_steps) / task_flags.total_steps

        if self._model_flags.use_scheduler:
            self._scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        else:
            self._scheduler = None

        if self._scheduler is not None and self._scheduler_state_dict is not None:
            self._scheduler.load_state_dict(self._scheduler_state_dict)
            self._scheduler_state_dict = None

        self._demonstration_mode = task_flags.demonstration_task

    def compute_loss_ddpg_demo(self, model_flags, task_flags, batch, initial_agent_state):
        """
        Train the ddpg actor-critic using demonstrations
        """
        current_time_batch = {key: tensor[:-1] for key, tensor in batch.items()}
        q_batch, unused_state = self._learner_model(current_time_batch, task_flags.action_space_id, initial_agent_state, action=None)
        current_time_batch["action"] = current_time_batch["action"].squeeze(1)
        #q_batch['action'] = q_batch['action'].view(current_time_batch['action'].shape)  # TODO: this shouldn't be necessary...?

        #print(f"Current vector: {current_time_batch['state_vector']}")
        print(f"Q batch action: {q_batch['action']}")

        assert q_batch["action"].shape == current_time_batch["action"].shape, f"Learned ({q_batch['action'].shape}) and stored actions ({current_time_batch['action'].shape}) should have the same shape"
        actor_loss = nn.MSELoss(reduction="sum")(q_batch["action"], current_time_batch["action"])
        stats = {"demo_actor_loss": actor_loss.item()}

        return stats, actor_loss

    def compute_loss_ddpg(self, model_flags, task_flags, batch, initial_agent_state, custom_loss_fn, compute_action):
        # Note the action_space_id isn't really used - it's used to generate an action, but we use the action that
        # was already computed and executed
        current_time_batch = {key: tensor[:-1] for key, tensor in batch.items()}
        next_time_batch = {key: tensor[1:] for key, tensor in batch.items()}

        action_for_model = current_time_batch['action'] if not compute_action else None
        q_batch, unused_state = self._learner_model(current_time_batch, task_flags.action_space_id, initial_agent_state, action=action_for_model)
        next_q_values, unused_state = self._target_learner_model(next_time_batch, task_flags.action_space_id, initial_agent_state, action=None)  # Target recomputes, to emulate "max"

        rewards = current_time_batch["reward"]  # TODO: current and next right

        # from https://github.com/MiniHackPlanet/MiniHack/blob/e124ae4c98936d0c0b3135bf5f202039d9074508/minihack/agent/polybeast/polybeast_learner.py#L243
        if model_flags.normalize_reward:
            self._learner_model.update_running_moments(rewards)
            rewards /= self._learner_model.get_running_std()

        if model_flags.reward_clipping == "abs_one":
            clipped_rewards = torch.clamp(rewards, -1, 1)
        elif model_flags.reward_clipping == "none":
            clipped_rewards = rewards

        discounts = (~current_time_batch["done"]).float() * model_flags.discounting

        # Compute baseline loss
        target_q_batch = clipped_rewards + discounts * next_q_values["baseline"].detach()
        baseline_loss = model_flags.baseline_cost * nn.MSELoss()(target_q_batch, q_batch["baseline"])

        # Compute actor loss
        actor_loss = -q_batch["baseline"].mean()

        stats = {"actor_loss": actor_loss.item(),
                 "baseline_loss": baseline_loss.item()}

        if custom_loss_fn is not None: # auxilary terms for continual learning
            custom_loss, custom_stats = custom_loss_fn(task_flags, self._learner_model, initial_agent_state)
            stats.update(custom_stats)
        else:
            custom_loss = torch.tensor([0]).squeeze()

        stats["custom_loss"] = custom_loss.item()

        return stats, actor_loss, baseline_loss, custom_loss

    def _step_optimizer(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        norm = nn.utils.clip_grad_norm_(self._learner_model.parameters(), self._model_flags.grad_norm_clipping)
        optimizer.step()
        return norm

    def compute_loss(self, task_flags, batch, initial_agent_state, custom_loss_fn):
        stats = {}

        # Update the critic
        critic_stats, _, baseline_loss, _ = self.compute_loss_ddpg(self._model_flags, task_flags, batch,
                                                         initial_agent_state, custom_loss_fn=None, compute_action=False)
        critic_norm = self._step_optimizer(baseline_loss, self._critic_optimizer)
        critic_stats["critic_norm"] = critic_norm.item()
        stats.update(critic_stats)

        # Update the actor
        if task_flags.demonstration_task:
            actor_stats, actor_loss = self.compute_loss_ddpg_demo(self._model_flags, task_flags, batch,
                                                                  initial_agent_state)
            pass
        else:
            actor_stats, actor_loss, _, _ = self.compute_loss_ddpg(self._model_flags, task_flags, batch,
                                                                   initial_agent_state, custom_loss_fn=None,
                                                                   compute_action=True)
        actor_norm = self._step_optimizer(actor_loss, self._actor_optimizer)
        actor_stats["actor_norm"] = actor_norm.item()
        stats.update(actor_stats)

        # Update using the custom loss
        """if custom_loss_fn is not None:
            custom_stats, _, _, custom_loss = self.compute_loss_ddpg(self._model_flags, task_flags, batch,
                                                             initial_agent_state, custom_loss_fn=custom_loss_fn)
            custom_loss_norm = self._step_optimizer(custom_loss, self.optimizer)
            custom_stats["custom_loss_norm"] = custom_loss_norm.item()
            stats.update(custom_stats)"""

        if self._scheduler is not None:
            self._scheduler.step()

        self._target_learner_model.update_parameters(self._learner_model)

        return stats
