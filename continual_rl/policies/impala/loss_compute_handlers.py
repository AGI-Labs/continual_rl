import numpy as np
import torch
from torch.nn import functional as F

from continual_rl.policies.impala.torchbeast.core import vtrace


class MonobeastLossComputeHandler():

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

    def compute_loss(self, model_flags, task_flags, learner_model, batch, initial_agent_state, custom_loss_fn=None):
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

        if custom_loss_fn is not None: # auxiliary terms for continual learning
            custom_loss, custom_stats = custom_loss_fn(task_flags, learner_model, initial_agent_state)
            total_loss += custom_loss
            stats.update(custom_stats)

        return total_loss, stats, pg_loss, baseline_loss


class APPOMonobeastComputeLossHandler():
    """
    Adapted from hackrl/experiment.py
    """
    def compute_baseline_loss(
        self, actor_baseline, learner_baseline, target, clip_delta_value=None, stats=None
    ):
        baseline_loss = (target - learner_baseline) ** 2

        if clip_delta_value:
            # Common PPO trick - clip a change in baseline fn
            # (cf PPO2 github.com/Stable-Baselines-Team/stable-baselines)
            delta_baseline = learner_baseline - actor_baseline
            clipped_baseline = actor_baseline + torch.clamp(
                delta_baseline, -clip_delta_value, clip_delta_value
            )

            clipped_baseline_loss = (target - clipped_baseline) ** 2

            if stats:
                clipped = (clipped_baseline_loss > baseline_loss).float().mean().item()
                stats["clipped_baseline_fraction"] += clipped

            baseline_loss = torch.max(baseline_loss, clipped_baseline_loss)

        if stats:
            stats["max_baseline_value"] += torch.max(learner_baseline).item()
            stats["min_baseline_value"] += torch.min(learner_baseline).item()
            stats["mean_baseline_value"] += torch.mean(learner_baseline).item()
        return 0.5 * torch.mean(baseline_loss)

    def compute_entropy_loss(self, logits, stats=None):
        policy = F.softmax(logits, dim=-1)
        log_policy = F.log_softmax(logits, dim=-1)
        entropy_per_timestep = torch.sum(-policy * log_policy, dim=-1)
        if stats:
            stats["max_entropy_value"] += torch.max(entropy_per_timestep).item()
            stats["min_entropy_value"] += torch.min(entropy_per_timestep).item()
            stats["mean_entropy_value"] += torch.mean(entropy_per_timestep).item()
        return -torch.mean(entropy_per_timestep)

    def compute_policy_gradient_loss(
        self,
        actor_log_prob,
        learner_log_prob,
        advantages,
        normalize_advantages=False,
        clip_delta_policy=None,
        stats=None,
    ):
        advantages = advantages.detach()

        if normalize_advantages:
            # Common PPO trick (cf PPO2 github.com/Stable-Baselines-Team/stable-baselines)
            adv = advantages
            advantages = (adv - adv.mean()) / max(1e-3, adv.std())

        if clip_delta_policy:
            # APPO policy loss - clip a change in policy fn
            ratio = torch.exp(learner_log_prob - actor_log_prob)
            policy_loss = ratio * advantages

            clip_high = 1 + clip_delta_policy
            clip_low = 1.0 / clip_high

            clipped_ratio = torch.clamp(ratio, clip_low, clip_high)
            clipped_policy_loss = clipped_ratio * advantages

            if stats:
                clipped_fraction = (clipped_policy_loss < policy_loss).float().mean().item()
                stats["clipped_policy_fraction"] += clipped_fraction
            policy_loss = torch.min(policy_loss, clipped_policy_loss)
        else:
            # IMPALA policy loss
            policy_loss = learner_log_prob * advantages

        return -torch.mean(policy_loss)

    def compute_loss(self, model_flags, task_flags, learner_model, batch, initial_agent_state, custom_loss_fn=None):
        stats = None  # TODO

        #env_outputs = batch["env_outputs"]
        #actor_outputs = batch["actor_outputs"]
        #initial_core_state = batch["initial_core_state"]

        learner_outputs, unused_state = learner_model(batch, task_flags.action_space_id, initial_agent_state)

        # Use last baseline value (from the value function) to bootstrap.
        bootstrap_value = learner_outputs["baseline"][-1]

        rewards = batch["reward"] * model_flags.reward_scale
        if model_flags.reward_clip:
            rewards = torch.clip(rewards, -model_flags.reward_clip, model_flags.reward_clip)

        if model_flags.normalize_reward:
            # Only NetHackNet models
            learner_model.update_running_moments(rewards)
            rewards /= learner_model.get_running_std()

        discounts = (~batch["done"]).float() * model_flags.discounting

        vtrace_returns = vtrace.from_logits(
            behavior_policy_logits=batch["policy_logits"],
            target_policy_logits=learner_outputs["policy_logits"],
            actions=batch["action"],
            discounts=discounts,
            rewards=rewards,
            values=learner_outputs["baseline"],
            bootstrap_value=bootstrap_value,
        )

        pg_loss = self.compute_policy_gradient_loss(
            vtrace_returns.behavior_action_log_probs,
            vtrace_returns.target_action_log_probs,
            vtrace_returns.pg_advantages,
            model_flags.normalize_advantages,
            model_flags.appo_clip_policy,
            stats,
        )

        baseline_loss = model_flags.baseline_cost * self.compute_baseline_loss(
            batch["baseline"],
            learner_outputs["baseline"],
            vtrace_returns.vs,
            model_flags.appo_clip_baseline,
            stats,
        )

        entropy_loss = model_flags.entropy_cost * self.compute_entropy_loss(
            learner_outputs["policy_logits"], stats
        )

        total_loss = pg_loss + baseline_loss + entropy_loss
        stats = {
            "pg_loss": pg_loss.item(),
            "baseline_loss": baseline_loss.item(),
            "entropy_loss": entropy_loss.item(),
        }

        if custom_loss_fn is not None: # auxiliary terms for continual learning
            custom_loss, custom_stats = custom_loss_fn(task_flags, learner_model, initial_agent_state)
            total_loss += custom_loss
            stats.update(custom_stats)

        return total_loss, stats, pg_loss, baseline_loss
