from continual_rl.policies.policy_base import PolicyBase
from continual_rl.policies.progress_and_compress.progress_and_compress_policy_config import ProgressAndCompressPolicyConfig


class ProgressAndCompressPolicy(PolicyBase):
    """
    Based on Progress & Compress, as described here: https://arxiv.org/pdf/1805.06370.pdf

    """
    def __init__(self, config: ProgressAndCompressPolicyConfig, observation_space, action_spaces):  # Switch to your config type
        super().__init__()
        pass

    def get_environment_runner(self, task_spec):
        pass

    def compute_action(self, observation, task_id, action_space_id, last_timestep_data, eval_mode):
        pass

    def train(self, storage_buffer):
        pass

    def save(self, output_path_dir, task_id, task_total_steps):
        pass

    def load(self, model_path):
        pass
