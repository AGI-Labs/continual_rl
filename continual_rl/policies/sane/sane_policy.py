import os
import copy
import json
import torch
import torch.optim as optim
import numpy as np
import threading
from continual_rl.policies.policy_base import PolicyBase
from continual_rl.policies.impala.impala_environment_runner import ImpalaEnvironmentRunner
from continual_rl.policies.impala.impala_policy import ImpalaPolicy
from continual_rl.policies.clear.clear_monobeast import ClearMonobeast
from continual_rl.utils.utils import Utils
from continual_rl.policies.impala.torchbeast.core.environment import Environment
from continual_rl.policies.sane.node_viz_singleton import NodeVizSingleton


class SaneEnvironmentRunner(ImpalaEnvironmentRunner):
    def __init__(self, config, policy):
        super().__init__(config, policy)
        self._policy = policy
        self._cached_environment_runners = {}
        self._total_timesteps = 0
        self._timesteps_since_update = 0

    def collect_data(self, task_spec):
        # Compute who should run the next data collection
        active_node, max_predicted_value, selected_uncertainty = self._policy.get_active_node(task_spec)
        self._logger.info(f"Collecting data with active node {active_node.unique_id}")
        if active_node.unique_id not in self._cached_environment_runners:
            self._cached_environment_runners[active_node.unique_id] = active_node.get_environment_runner(task_spec)

        timesteps, all_env_data, rewards_to_report, logs_to_report = self._cached_environment_runners[active_node.unique_id].collect_data(task_spec)
        self._total_timesteps += timesteps
        self._timesteps_since_update += timesteps
        active_node.usage_count += timesteps

        if self._config.train_all:
            self._policy.train_all(task_spec)

        # Check if anything needs updating
        if not task_spec.eval_mode and not self._config.static_ensemble:
            self._policy.update_available_nodes(task_spec, self._total_timesteps, active_node)
            self._policy.ensure_max_nodes(task_spec)

        if self._config.use_slow_critic and self._timesteps_since_update > self._config.slow_critic_update_cadence:
            self._timesteps_since_update = 0
            active_node.slow_critic.update_parameters(active_node.impala_trainer.actor_model)

        suffix = f"_eval_{task_spec.task_id}" if task_spec.eval_mode else ""
        logs_to_report.append({'type': 'scalar', 'tag': f'num_sane_nodes', 'value': len(self._policy._nodes)})
        logs_to_report.append({'type': 'scalar', 'tag': f'active_node_id{suffix}', 'value': active_node.unique_id})
        logs_to_report.append({'type': 'scalar', 'tag': f'predicted_value{suffix}', 'value': max_predicted_value})
        logs_to_report.append({'type': 'scalar', 'tag': f'uncertainty{suffix}', 'value': selected_uncertainty})

        return timesteps, all_env_data, rewards_to_report, logs_to_report

    def cleanup(self, task_spec):
        if not task_spec.eval_mode:
            for node in self._policy._nodes:
                node.impala_trainer.cleanup()
        del self._result_generators


class SanePolicy(PolicyBase):
    def __init__(self, config, observation_space, action_spaces):
        super().__init__(config)
        self._config = config
        self._observation_space = observation_space
        self._action_spaces = action_spaces
        self._nodes = []
        self._task_id_to_node_map = {}

    @property
    def _logger(self):
        logger = Utils.create_logger(f"{self._config.output_dir}/sane.log")
        return logger

    def _get_canonical_obs(self, task_spec):
        dummy_env = Environment(Utils.make_env(task_spec.env_spec)[0])
        obs = dummy_env.initial()
        return obs

    def _add_replay_buffer(self, source_node, target_node):
        num_actors = len(source_node.impala_trainer._replay_buffers['frame'])
        num_buffers = len(source_node.impala_trainer._replay_buffers['frame'][0])
        for actor_index in range(num_actors):
            for buffer_id in range(num_buffers):
                new_buffers = source_node.impala_trainer._replay_buffers
                if new_buffers['reservoir_val'][actor_index][buffer_id] > 0:
                    actor_buffers = {key: new_buffers[key][actor_index][buffer_id] for key in new_buffers.keys()}
                    target_node.impala_trainer.on_act_unroll_complete(task_flags=None, actor_index=actor_index, agent_output=None,
                                                                      env_output=None, new_buffers=actor_buffers)

    def _duplicate_node(self, source_node):
        new_node = SaneNode(self._config, self._observation_space, self._action_spaces, self)
        new_node.impala_trainer.actor_model.load_state_dict(source_node.impala_trainer.actor_model.state_dict())
        new_node.impala_trainer.learner_model.load_state_dict(source_node.impala_trainer.actor_model.state_dict())

        if self._config.duplicate_optimizer:
            new_node.impala_trainer.optimizer.load_state_dict(source_node.impala_trainer.optimizer.state_dict())

        if self._config.use_slow_critic:
            new_node.slow_critic.load_state_dict(source_node.slow_critic.state_dict())
            new_node.prototype.load_state_dict(source_node.slow_critic.state_dict())
        else:
            new_node.prototype.load_state_dict(source_node.impala_trainer.actor_model.state_dict())

        if self._config.create_adds_replay:
            self._add_replay_buffer(source_node, new_node)

        return new_node

    def _create_node_from_source(self, source_node, node_fell_below_anchor):
        new_node = None
        if self._config.creation_pattern == "keep_anchor":
            # Create new node
            new_node = self._duplicate_node(source_node)

            # Set old node back to its prototype
            source_node.impala_trainer.actor_model.load_state_dict(source_node.prototype.state_dict())
            source_node.impala_trainer.learner_model.load_state_dict(source_node.prototype.state_dict())

        elif self._config.creation_pattern == "asymmetric_reset_anchor":
            # If we are in a new task, and the previous behavior is doing poorly, create a new node to minimally
            # disrupt the previous node
            if node_fell_below_anchor:
                new_node = self._duplicate_node(source_node)

            # Replace the anchor - we're learning, and want to avoid churn
            if self._config.use_slow_critic:
                source_node.prototype.load_state_dict(source_node.slow_critic.state_dict())
            else:
                source_node.prototype.load_state_dict(source_node.impala_trainer.actor_model.state_dict())

        else:
            raise Exception(f"Unexpected creation pattern {self._config.creation_pattern}")

        if self._config.visualize_nodes and new_node is not None:
            NodeVizSingleton.instance().create_node(self._config.output_dir, new_node.unique_id)
            NodeVizSingleton.instance().register_created_from(self._config.output_dir, new_node.unique_id,
                                                              source_node.unique_id)

        return new_node

    def train_all(self, task_flags):
        for node in self._nodes:
            self._train(node, task_flags)

    def _train(self, node, task_flags):
        batch = node.impala_trainer.get_batch_for_training(None, store_for_loss=False)
        if batch is not None:
            initial_agent_state = None
            node.impala_trainer.learn(model_flags=self._config,
                                      task_flags=task_flags,
                                      actor_model=node.impala_trainer.actor_model,
                                      learner_model=node.impala_trainer.learner_model,
                                      batch=batch,
                                      initial_agent_state=initial_agent_state,
                                      optimizer=node.impala_trainer.optimizer,
                                      scheduler=node.impala_trainer._scheduler,
                                      lock=threading.Lock())

    def get_active_node(self, task_spec):
        initial_obs = self._get_canonical_obs(task_spec)

        max_predicted_value = None
        selected_node = None
        selected_uncertainty = None

        if len(self._nodes) == 0:
            num_new_nodes = self._config.max_nodes if self._config.static_ensemble else 1
            for _ in range(num_new_nodes):
                new_node = SaneNode(self._config, self._observation_space, self._action_spaces, self)
                self._nodes.append(new_node)
                NodeVizSingleton.instance().create_node(self._config.output_dir, new_node.unique_id)

        if self._config.map_task_id_to_module:
            if task_spec.task_id not in self._task_id_to_node_map:
                self._task_id_to_node_map[task_spec.task_id] = self._nodes[len(self._task_id_to_node_map)]

            selected_node = self._task_id_to_node_map[task_spec.task_id]
            max_predicted_value = -1
            selected_uncertainty = -1
        else:

            for node in self._nodes:
                if self._config.use_slow_critic:
                    actor_result = node.slow_critic_forward(initial_obs, task_spec.action_space_id)[0]
                else:
                    actor_result = node.policy_forward(initial_obs, task_spec.action_space_id)[0]

                predicted_value = actor_result['baseline'] + self._config.uncertainty_scale_in_get_active * torch.abs(actor_result['uncertainty'])
                if max_predicted_value is None or predicted_value > max_predicted_value:
                    selected_node = node
                    max_predicted_value = predicted_value
                    selected_uncertainty = torch.abs(actor_result['uncertainty'])

        return selected_node, max_predicted_value, selected_uncertainty

    def update_available_nodes(self, task_spec, total_timesteps, active_node):
        canonical_obs = self._get_canonical_obs(task_spec)
        new_nodes = []

        if self._config.only_create_from_active:
            nodes = [active_node]
        else:
            nodes = self._nodes

        for node in nodes:
            if self._config.use_slow_critic:
                policy_result = node.slow_critic_forward(canonical_obs, task_spec.action_space_id)[0]
            else:
                policy_result = node.policy_forward(canonical_obs, task_spec.action_space_id)[0]

            prototype_result = node.prototype_forward(canonical_obs, task_spec.action_space_id)[0]['baseline']
            lower_bound = policy_result['baseline'] - self._config.allowed_uncertainty_scale_for_creation[0] * torch.abs(policy_result['uncertainty'])
            upper_bound = policy_result['baseline'] + self._config.allowed_uncertainty_scale_for_creation[1] * torch.abs(policy_result['uncertainty'])
            node_beat_anchor = torch.any(prototype_result < lower_bound)
            node_fell_below_anchor = torch.any(prototype_result > upper_bound)

            self._logger.info(f"[{node.unique_id}] LB: {lower_bound}, UB: {upper_bound}, proto: {prototype_result}")

            if node_beat_anchor or node_fell_below_anchor or \
                    (len(self._nodes) == 1 and total_timesteps > self._config.min_steps_before_force_create):
                new_node = self._create_node_from_source(node, node_fell_below_anchor=node_fell_below_anchor)
                if new_node is not None:
                    self._nodes.append(new_node)

        self._nodes.extend(new_nodes)

    def _get_closest_nodes(self, mergeable_nodes):
        policies = torch.stack([node.get_merge_metric() for node in mergeable_nodes])

        difference = policies.unsqueeze(1) - policies  # Taking advantage of the auto-dim matching thing.
        square_distances = (difference ** 2).sum(dim=-1).detach().cpu()

        side_length = square_distances.shape[0]
        indices = np.argsort(square_distances, axis=None)
        indices_x = indices // side_length
        indices_y = indices % side_length

        hypo_index = 0
        hypo_indices = []
        num_indices = 1  # Vestigial, only allowing 1 right now

        # Gather the num_indices entries with the smallest values that aren't on the diagonal
        for _ in range(num_indices):
            # Ignore diagonals
            while indices_x[hypo_index] == indices_y[hypo_index] and hypo_index < len(indices_x):
                hypo_index += 1

            if hypo_index < len(indices_x):
                hypo_indices.append(hypo_index)

            hypo_index += 1

        final_hypo_index = hypo_indices[0]
        assert final_hypo_index in hypo_indices, "Somehow picked a bad index"

        selected_x = indices_x[final_hypo_index]
        selected_y = indices_y[final_hypo_index]

        return mergeable_nodes[selected_x], mergeable_nodes[selected_y]

    def ensure_max_nodes(self, task_flags):
        while len(self._nodes) > self._config.max_nodes:
            num_mergeable = int(self._config.fraction_of_nodes_mergeable * self._config.max_nodes)
            node_to_keep, node_to_remove = self._get_closest_nodes(self._nodes[:num_mergeable])

            # Using min (non-zero) reservoir value as a proxy for usage count, so we keep the node with the higher value
            if (self._config.keep_larger_reservoir_val_in_merge and
                    node_to_remove.impala_trainer.get_min_reservoir_val_greater_than_zero() > node_to_keep.impala_trainer.get_min_reservoir_val_greater_than_zero()) or \
                    (self._config.usage_count_based_merge and node_to_remove.usage_count > node_to_keep.usage_count):
                node_to_keep, node_to_remove = node_to_remove, node_to_keep

            node_to_keep.usage_count += node_to_remove.usage_count
            self._add_replay_buffer(node_to_remove, node_to_keep)
            self._train(node_to_keep, task_flags)
            self._nodes.remove(node_to_remove)

            self._logger.info(f"Deleting resources for node {node_to_remove.unique_id}")
            node_to_remove.impala_trainer.permanent_delete()
            self._logger.info("Deletion complete")

            if self._config.visualize_nodes:
                NodeVizSingleton.instance().merge_node(self._config.output_dir, node_to_remove.unique_id,
                                                       node_to_keep.unique_id)

    def get_environment_runner(self, task_spec):
        return SaneEnvironmentRunner(self._config, self)

    def compute_action(self, observation, task_id, action_space_id, last_timestep_data, eval_mode):
        pass

    def load(self, output_path_dir):
        node_metadata = os.path.join(output_path_dir, "metadata.json")

        if os.path.exists(node_metadata):
            self._nodes = []
            with open(node_metadata, "r") as metadata_file:
                all_node_data = json.load(metadata_file)

            for unique_id, node_data in all_node_data.items():
                loaded_node = SaneNode(self._config, self._observation_space, self._action_spaces, self, int(unique_id))
                loaded_node.load(node_data["path"])
                loaded_node.usage_count = node_data["usage_count"]
                self._nodes.append(loaded_node)

    def save(self, output_path_dir, cycle_id, task_id, task_total_steps):
        node_data = {}
        for node in self._nodes:
            node_path = os.path.join(output_path_dir, "node_save_data", f"node_{node.unique_id}")
            os.makedirs(node_path, exist_ok=True)
            node.save(node_path, cycle_id, task_id, task_total_steps)
            node_data[node.unique_id] = {"path": node_path, "usage_count": node.usage_count}

        node_metadata_path = os.path.join(output_path_dir, "metadata.json")
        with open(node_metadata_path, "w+") as metadata_file:
            json.dump(node_data, metadata_file)

    def train(self, storage_buffer):
        pass


class SaneMonobeast(ClearMonobeast):
    def custom_loss(self, task_flags, model, initial_agent_state, batch, vtrace_returns):
        clear_loss, stats = super().custom_loss(task_flags, model, initial_agent_state, batch, vtrace_returns)

        model_outputs, unused_state = model(batch, task_flags.action_space_id, initial_agent_state)
        uncertainties = torch.abs(model_outputs['baseline'] - vtrace_returns.vs)
        uncertainty_loss = ((model_outputs['uncertainty'] - uncertainties.detach())**2).mean()
        total_loss = self._model_flags.clear_loss_coeff * clear_loss

        total_loss = total_loss + self._model_flags.uncertainty_scale * uncertainty_loss
        stats["uncertainty_loss"] = uncertainty_loss.item()
        stats["uncertainty_mean"] = uncertainties.mean().item()

        return total_loss, stats


class SaneNode(ImpalaPolicy):
    """
    Each node uses its own CLEAR-based Monobeast so that each has its own separate replay buffer
    """
    UNIQUE_ID_COUNTER = 0

    def __init__(self, config, observation_space, action_spaces, ensemble, unique_id=None):
        self.unique_id = self._get_unique_id(unique_id)
        node_config = copy.deepcopy(config)
        node_config.policy_unique_id = f"{config.policy_unique_id}node_{self.unique_id}"
        super().__init__(node_config, observation_space, action_spaces, impala_class=SaneMonobeast)

        self._ensemble = ensemble
        self.usage_count = 0

        if config.use_slow_critic:
            if config.slow_critic_ema_new_weight > 0:
                # Exponential moving average
                avg_fn = lambda averaged_model_parameter, model_parameter, num_averaged: \
                    (1-config.slow_critic_ema_new_weight) * averaged_model_parameter + config.slow_critic_ema_new_weight * model_parameter
            else:
                # None means equally weighted average
                avg_fn = None

            self.slow_critic = optim.swa_utils.AveragedModel(self.impala_trainer.actor_model, avg_fn=avg_fn)  # Actor is on cpu, easier
            self.prototype = copy.deepcopy(self.slow_critic)
        else:
            self.prototype = copy.deepcopy(self.impala_trainer.actor_model)

    @classmethod
    def _get_unique_id(cls, unique_id=None):
        if unique_id is not None:
            cls.UNIQUE_ID_COUNTER = max(cls.UNIQUE_ID_COUNTER, unique_id + 1)  # The nodes might not be in numerical order, so max() it
        else:
            unique_id = cls.UNIQUE_ID_COUNTER
            cls.UNIQUE_ID_COUNTER += 1

        return unique_id

    def slow_critic_forward(self, obs, action_space_id):
        return self.slow_critic(obs, action_space_id)

    def policy_forward(self, obs, action_space_id):
        return self.impala_trainer.actor_model(obs, action_space_id)

    def prototype_forward(self, obs, action_space_id):
        return self.prototype(obs, action_space_id)

    def get_merge_metric(self):
        buffers = None
        if self._config.merge_by_batch:
            buffers = self.impala_trainer.get_batch_for_training(batch=None, store_for_loss=False,
                                                                 reuse_actor_indices=True,
                                                                 replay_entry_scale=self._config.merge_batch_scale)

        if buffers is None:
            buffers = self.impala_trainer._replay_buffers  # Will possibly include unfilled entries

        if self._config.merge_by_frame:
            metric = buffers['frame']
            if not isinstance(metric, torch.Tensor):  # The batch returns it pre-stacked, don't re-stack in that case
                metric = torch.stack(metric).float().mean(dim=0)
            metric = metric.float().mean(dim=0).mean(dim=0).mean(dim=0).view(-1)
        else:
            policies = buffers['policy_logits']
            if not isinstance(policies, torch.Tensor):
                policies = torch.stack(policies).mean(dim=0)
            metric = policies.mean(dim=0).mean(dim=0)

        return metric.cpu()
