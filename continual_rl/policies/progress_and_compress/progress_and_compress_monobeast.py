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

    def _compute_kl_div_loss(self, input, target):
        # KLDiv requires inputs to be log-probs, and targets to be probs
        old_policy = F.softmax(input, dim=-1)
        curr_log_policy = F.log_softmax(target, dim=-1)
        kl_loss = torch.nn.KLDivLoss(reduction='sum')(curr_log_policy, old_policy.detach())
        return kl_loss

    def custom_loss(self, model, initial_agent_state):
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
