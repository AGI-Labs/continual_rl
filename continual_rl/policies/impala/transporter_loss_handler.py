import torch
import torch.nn as nn
from torch.nn import functional as F
from continual_rl.policies.impala.torchbeast.core import vtrace
from ravens_torch.dataset import Dataset
from continual_rl.envs.ravens_demonstration_env import RavensDemonstrationEnv
import random
import os


class TransporterLossHandler(object):
    """
    The default loss handler for IMPALA
    """
    def __init__(self, model_flags, learner_model):
        self._model_flags = model_flags
        self._learner_model = learner_model

        # TODO: the transporter has optimizers already. Use them
        self.optimizer = self._create_optimizer(model_flags, learner_model.parameters(), model_flags.learning_rate)  # Used for custom losses (TODO?)

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

        return checkpoint_data

    def load_save_data(self, checkpoint):
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def initialize_for_task(self, task_flags):
        pass

    def compute_loss_transporter_demo(self, model_flags, task_flags, batch, initial_agent_state):
        """
        Train the ddpg actor-critic using demonstrations
        """
        current_time_batch = {key: tensor[:-1] for key, tensor in batch.items()}

        # Construct a dummy Dataset to train with, to make using the existing ravens_torch API easier... TODO
        seed = random.randint(0, 1e8)  # TODO: this is hacky
        dataset = Dataset(path=os.path.join(model_flags.output_dir, str(seed)))

        for episode_id in range(current_time_batch["action"].shape[1]):
            episode_data = []  # "Trajectory" would be more accurate...

            for timestep_id in range(current_time_batch["action"].shape[0]):
                all_color_data, all_depth_data = self._learner_model._convert_aggregated_images_to_per_camera_data(
                    current_time_batch["image"][timestep_id][episode_id].squeeze(0))

                # If we don't dump the info (and why would we?) we don't need to specify the seed
                image = {"color": all_color_data, "depth": all_depth_data}
                timestep_data = (image,
                                 RavensDemonstrationEnv.convert_unified_action_to_dict(
                                     current_time_batch['action'][timestep_id][episode_id]),
                                 current_time_batch['reward'][timestep_id][episode_id],
                                 None)
                episode_data.append(timestep_data)

            dataset.add(seed=seed + episode_id, episode=episode_data)

        attention_loss, transport_loss = self._learner_model.agent.train(dataset)
        actor_loss = attention_loss + transport_loss
        stats = {"attention_loss": attention_loss.item(), "transport_loss": transport_loss.item()}

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

        # Update the actor
        if task_flags.demonstration_task:
            actor_stats, actor_loss = self.compute_loss_transporter_demo(self._model_flags, task_flags, batch,
                                                                  initial_agent_state)
            pass
        else:
            # TODO: removed the target network here...
            actor_stats, actor_loss, _, _ = self.compute_loss_ddpg(self._model_flags, task_flags, batch,
                                                                   initial_agent_state, custom_loss_fn=None,
                                                                   compute_action=True)
        #actor_norm = self._step_optimizer(actor_loss, self.optimizer)
        #actor_stats["actor_norm"] = actor_norm.item()
        stats.update(actor_stats)

        # Update using the custom loss
        """if custom_loss_fn is not None:
            custom_stats, _, _, custom_loss = self.compute_loss_ddpg(self._model_flags, task_flags, batch,
                                                             initial_agent_state, custom_loss_fn=custom_loss_fn)
            custom_loss_norm = self._step_optimizer(custom_loss, self.optimizer)
            custom_stats["custom_loss_norm"] = custom_loss_norm.item()
            stats.update(custom_stats)"""

        return stats
