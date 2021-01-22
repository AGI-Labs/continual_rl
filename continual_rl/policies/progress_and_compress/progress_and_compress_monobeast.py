import torch
from torch.nn import functional as F
from continual_rl.policies.ewc.ewc_monobeast import EWCMonobeast


class ProgressAndCompressMonobeast(EWCMonobeast):
    """
    Progress and Compress leverages Online EWC (implemented in EWCMonobeast). We just modify it such that
    the knowledge base is what is updated using the EWC loss.
    """
    def __init__(self, model_flags, observation_space, action_spaces, policy_class):
        super().__init__(model_flags, observation_space, action_spaces, policy_class)
        self._kb_train_steps_since_boundary = None
        self._current_task_id = None

    def _compute_kl_div_loss(self, input, target):
        # KLDiv requires inputs to be log-probs, and targets to be probs
        old_policy = F.log_softmax(input, dim=-1)
        curr_log_policy = F.softmax(target, dim=-1)
        kl_loss = torch.nn.KLDivLoss(reduction='sum')(old_policy, curr_log_policy.detach())
        return kl_loss

    def knowledge_base_loss(self, model, initial_agent_state):
        ewc_loss, ewc_stats = super().custom_loss(model.knowledge_base, initial_agent_state)

        # Additionally, minimize KL divergence between KB and active column (only updating KB)
        replay_buffer_subset = self._sample_from_task_replay_buffer(self._cur_task_id, self._model_flags.batch_size)
        with torch.no_grad():
            targets, _ = model(replay_buffer_subset)

        knowledge_base_outputs, _ = model.knowledge_base(replay_buffer_subset)
        kl_div_loss = self._compute_kl_div_loss(input=knowledge_base_outputs['policy_logits'],
                                                target=targets['policy_logits'])

        total_loss = ewc_loss + kl_div_loss
        ewc_stats.update({"kl_div_loss": kl_div_loss})

        return total_loss, ewc_stats

    def compute_loss(self, flags, model, batch, initial_agent_state, with_custom_loss=True):
        """
        Sometimes we want to turn off normal loss computation entirely, so controlling that here.
        During the "wake" part of the cycle, we use the normal compute_loss to update the active column (AC).
        During "sleep" we use EWC+KL to update the knowledge base (KB).
        The P&C paper is not very clear on cadence/length of the phases, so we're assuming sleep starts at
        the end of a task, and lasts a hyperparameter number of learning steps (num_train_steps_of_compress).
        We are assuming it is happening alongside continued data collection because the paper references "rewards
        collected during the compress phase".
        """
        if self._kb_train_steps_since_boundary is None or \
                self._kb_train_steps_since_boundary > self._model_flags.num_train_steps_of_compress:
            # This is the "active column" training setting. The custom loss here would be EWC, so don't include it
            results = super().compute_loss(flags, model, batch, initial_agent_state, with_custom_loss=False)
        else:
            # This is the "knowledge base" training setting
            results = self.knowledge_base_loss(model, initial_agent_state)
            self._kb_train_steps_since_boundary += 1

        return results

    def set_current_task(self, task_id):
        super().set_current_task(task_id)

        # Only kick off KB training after we switch to a new task, not including the first one
        if self._current_task_id is not None and self._current_task_id != task_id:
            self._kb_train_steps_since_boundary = 0
            self._current_task_id = task_id
