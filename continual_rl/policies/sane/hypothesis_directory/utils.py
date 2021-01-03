import numpy as np
from torch.distributions.categorical import Categorical
from continual_rl.utils.utils import Utils as ContinualRLUtils


class Utils(object):
    """
    Hypothesis directory-specific utilities.
    """

    @classmethod
    def get_log_probs(self, hypothesis, policy, action_size, random_action_rate, selected_action=None, verbose=False):
        action_logits = policy[:action_size]  # Take the first action_size actions (to enable environments with different action sizes)

        dist_computed = hypothesis.get_policy_as_categorical(policy)

        if selected_action is None:
            if np.random.uniform(0, 1.0) < 1 - random_action_rate:
                selected_action = dist_computed.sample()
            else:
                dist_uniform = Categorical(logits=action_logits * 0 + 1)
                selected_action = dist_uniform.sample()

        log_probs = dist_computed.log_prob(selected_action)
        entropy = dist_computed.entropy()

        if verbose:
            print(f"Action logits: {action_logits.detach().cpu().numpy()}, {selected_action}")

        return log_probs, selected_action, entropy

    @classmethod
    def create_logger(cls, file_path):
        """
        The name must be unique to the logger you're creating, otherwise you're grabbing an existing logger.
        Just delegating this to an existing implementation.
        """
        logger = ContinualRLUtils.create_logger(file_path)
        return logger
