import torch
import threading
import os
import json
from torch.nn import functional as F
from continual_rl.policies.ewc.ewc_monobeast import EWCMonobeast


class ProgressAndCompressMonobeast(EWCMonobeast):
    """
    Progress and Compress leverages Online EWC (implemented in EWCMonobeast). We just modify it such that
    the knowledge base is what is updated using the EWC loss.
    """
    def __init__(self, model_flags, observation_space, action_spaces, policy_class):
        super().__init__(model_flags, observation_space, action_spaces, policy_class)
        self._train_steps_since_boundary = 0
        self._previous_pnc_task_id = None  # Distinct from ewc's _prev_task_id
        self._step_count_lock = threading.Lock()

    def save(self, output_path):
        super().save(output_path)

        pnc_metadata_path = os.path.join(output_path, "pnc_metadata.json")
        metadata = {"prev_pnc_task_id": self._previous_pnc_task_id,
                    "train_steps_since_boundary": self._train_steps_since_boundary}
        with open(pnc_metadata_path, "w+") as pnc_file:
            json.dump(metadata, pnc_file)

    def load(self, output_path):
        super().load(output_path)
        pnc_metadata_path = os.path.join(output_path, "pnc_metadata.json")

        if os.path.exists(pnc_metadata_path):
            self.logger.info(f"Loading pnc metdata from {pnc_metadata_path}")
            with open(pnc_metadata_path, "r") as pnc_file:
                metadata = json.load(pnc_file)

            self._previous_pnc_task_id = metadata["prev_pnc_task_id"]
            self._train_steps_since_boundary = metadata["train_steps_since_boundary"]

    def _compute_kl_div_loss(self, input, target):
        # KLDiv requires inputs to be log-probs, and targets to be probs
        old_policy = F.log_softmax(input, dim=-1)
        curr_log_policy = F.softmax(target, dim=-1)
        kl_loss = torch.nn.KLDivLoss(reduction='sum')(old_policy, curr_log_policy.detach())
        return kl_loss

    def knowledge_base_loss(self, task_flags, model, initial_agent_state):
        # EWC not using batch, vtrace_returns, so not bothering to pass them through
        ewc_loss, ewc_stats = super().custom_loss(task_flags, model.knowledge_base, initial_agent_state, None, None)

        # Additionally, minimize KL divergence between KB and active column (only updating KB)
        replay_buffer_subset = self._sample_from_task_replay_buffer(task_flags.task_id, self._model_flags.batch_size)
        with torch.no_grad():
            targets, _ = model(replay_buffer_subset, task_flags.action_space_id)

        knowledge_base_outputs, _ = model.knowledge_base(replay_buffer_subset, task_flags.action_space_id)
        kl_div_loss = self._compute_kl_div_loss(input=knowledge_base_outputs['policy_logits'],
                                                target=targets['policy_logits'].detach())

        total_loss = ewc_loss + self._model_flags.kl_div_scale * kl_div_loss
        ewc_stats.update({"kl_div_loss": kl_div_loss.item()})

        return total_loss, ewc_stats

    def compute_loss(self, model_flags, task_flags, learner_model, batch, initial_agent_state, with_custom_loss=True):
        """
        Sometimes we want to turn off normal loss computation entirely, so controlling that here.
        During the "wake" part of the cycle, we use the normal compute_loss to update the active column (AC).
        During "sleep" we use EWC+KL to update the knowledge base (KB).
        The P&C paper does not report on cadence/length of the phases, but after discussion with an author,
        we're assuming sleep starts after num_train_steps_of_progress number of training steps, and lasts the rest of
        the task. (In the paper num_train_steps_of_progress is apparently half of the total steps of the task, and
        only the compress datapoints are plotted in Fig 4.)
        We are assuming it is happening alongside continued data collection because the paper references "rewards
        collected during the compress phase".
        """
        # Because we're not going through the normal EWC path
        # self._prev_task_id doesn't get initialized early enough, so force it here
        if self._prev_task_id is None:
            super().custom_loss(task_flags, learner_model.knowledge_base, initial_agent_state, None, None)

        # Only kick off KB training after we switch to a new task, not including the first one. This is
        # being used as boundary detection.
        with self._step_count_lock:
            current_task_id = task_flags.task_id
            self.logger.info(f"Prev id: {self._previous_pnc_task_id}, current id: {current_task_id}, steps since boundary: {self._train_steps_since_boundary}")
            if self._previous_pnc_task_id is not None and current_task_id != self._previous_pnc_task_id:
                self.logger.info("Boundary detected, resetting active column and starting Progress.")
                self._train_steps_since_boundary = 0

                # We have entered a new task. Since the model passed in is the learner model, just reset it.
                # The active column will be updated after this.
                learner_model.reset_active_column()

            self._previous_pnc_task_id = current_task_id

        if self._train_steps_since_boundary <= self._model_flags.num_train_steps_of_progress:
            if self._model_flags.use_collection_pause:
                super().set_pause_collection_state(False)  # Active column training should result in EWC data collection

            # This is the "active column" training setting. The custom loss here would be EWC, so don't include it
            loss, stats, pg_loss, baseline_loss = super().compute_loss(model_flags, task_flags, learner_model, batch, initial_agent_state, with_custom_loss=False)
        else:
            self.logger.info("Compressing...")
            if self._model_flags.use_collection_pause:
                super().set_pause_collection_state(True)  # Don't collect data while compressing

            # This is the "knowledge base" training setting
            loss, stats = self.knowledge_base_loss(task_flags, learner_model, initial_agent_state)
            pg_loss = 0  # No policy gradient when updating the knowledge base
            baseline_loss = 0

            # Monobeast expects these keys. Since we're bypassing the normal loss, add them into stats just as fakes (0)
            extra_keys = ["pg_loss", "baseline_loss", "entropy_loss"]
            for key in extra_keys:
                assert key not in stats
                stats[key] = 0

        with self._step_count_lock:
            self._train_steps_since_boundary += 1

        return loss, stats, pg_loss, baseline_loss
