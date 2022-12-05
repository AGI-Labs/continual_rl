import torch
import torch.nn as nn
from torch.nn import functional as F
from continual_rl.policies.impala.torchbeast.core import vtrace


class VtraceLossHandler(object):
    """
    The default loss handler for IMPALA
    """
    def __init__(self, model_flags, learner_model):
        self.optimizer = self._create_optimizer(model_flags, learner_model)
        self._scheduler_state_dict = None  # Filled if we load()
        self._scheduler = None  # Task-specific, so created there

        self._learner_model = learner_model
        self._model_flags = model_flags

    def _create_optimizer(self, model_flags, learner_model):
        if model_flags.optimizer == "rmsprop":
            optimizer = torch.optim.RMSprop(
                learner_model.parameters(),
                lr=model_flags.learning_rate,
                momentum=model_flags.momentum,
                eps=model_flags.epsilon,
                alpha=model_flags.alpha,
            )
        elif model_flags.optimizer == "adam":
            optimizer = torch.optim.Adam(
                learner_model.parameters(),
                lr=model_flags.learning_rate,
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
        assert not task_flags.demonstration_task, "Demonstration tasks not supported by IMPALA vtrace loss handler"
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

    def compute_baseline_loss(self, advantages):
        return 0.5 * torch.sum(advantages ** 2)

    def compute_entropy_loss(self, logits):
        """Return the entropy loss, i.e., the negative entropy of the policy."""
        policy = F.softmax(logits, dim=-1)
        log_policy = F.log_softmax(logits, dim=-1)
        return torch.sum(policy * log_policy)

    def compute_policy_gradient_loss(self, logits, actions, advantages):
        cross_entropy = F.nll_loss(
            F.log_softmax(torch.flatten(logits, 0, 1), dim=-1),
            target=torch.flatten(actions, 0, 1),
            reduction="none",
        )
        cross_entropy = cross_entropy.view_as(advantages)
        return torch.sum(cross_entropy * advantages.detach())

    def compute_loss_vtrace(self, model_flags, task_flags, learner_model, batch, initial_agent_state, custom_loss_fn):
        # Note the action_space_id isn't really used - it's used to generate an action, but we use the action that
        # was already computed and executed
        learner_outputs, unused_state = learner_model(batch, task_flags.action_space_id, initial_agent_state)

        # Take final value function slice for bootstrapping.
        bootstrap_value = learner_outputs["baseline"][-1]

        # Move from obs[t] -> action[t] to action[t] -> obs[t].
        batch = {key: tensor[1:] for key, tensor in batch.items()}
        learner_outputs = {key: tensor[:-1] for key, tensor in learner_outputs.items()}

        rewards = batch["reward"]

        # from https://github.com/MiniHackPlanet/MiniHack/blob/e124ae4c98936d0c0b3135bf5f202039d9074508/minihack/agent/polybeast/polybeast_learner.py#L243
        if model_flags.normalize_reward:
            learner_model.update_running_moments(rewards)
            rewards /= learner_model.get_running_std()

        if model_flags.reward_clipping == "abs_one":
            clipped_rewards = torch.clamp(rewards, -1, 1)
        elif model_flags.reward_clipping == "none":
            clipped_rewards = rewards

        discounts = (~batch["done"]).float() * model_flags.discounting

        vtrace_returns = vtrace.from_logits(
            behavior_policy_logits=batch["policy_logits"],
            target_policy_logits=learner_outputs["policy_logits"],
            actions=batch["action"],
            discounts=discounts,
            rewards=clipped_rewards,
            values=learner_outputs["baseline"],
            bootstrap_value=bootstrap_value,
        )

        pg_loss = self.compute_policy_gradient_loss(
            learner_outputs["policy_logits"],
            batch["action"],
            vtrace_returns.pg_advantages,
        )
        baseline_loss = model_flags.baseline_cost * self.compute_baseline_loss(
            vtrace_returns.vs - learner_outputs["baseline"]
        )
        entropy_loss = model_flags.entropy_cost * self.compute_entropy_loss(
            learner_outputs["policy_logits"]
        )

        total_loss = pg_loss + baseline_loss + entropy_loss
        stats = {
            "pg_loss": pg_loss.item(),
            "baseline_loss": baseline_loss.item(),
            "entropy_loss": entropy_loss.item(),
        }

        if custom_loss_fn is not None: # auxilary terms for continual learning
            custom_loss, custom_stats = custom_loss_fn(task_flags, learner_model, initial_agent_state, batch, vtrace_returns.vs)
            total_loss += custom_loss
            stats.update(custom_stats)

        stats["total_loss"] = total_loss.item()

        return total_loss, stats, pg_loss, baseline_loss

    def compute_loss(self, task_flags, batch, initial_agent_state, custom_loss_fn):

        total_loss, stats, _, _ = self.compute_loss_vtrace(self._model_flags, task_flags, self._learner_model, batch,
                                                           initial_agent_state, custom_loss_fn=custom_loss_fn)

        self.optimizer.zero_grad()
        total_loss.backward()

        norm = nn.utils.clip_grad_norm_(self._learner_model.parameters(), self._model_flags.grad_norm_clipping)
        stats["total_norm"] = norm.item()

        self.optimizer.step()
        if self._scheduler is not None:
            self._scheduler.step()

        return stats
