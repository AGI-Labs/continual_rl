from policies.policy_base import PolicyBase


class PPOPolicy(PolicyBase):
    """
    Basically a wrapper around torch-ac's implementation of PPO
    """
    def __init__(self):
        super().__init__()
        pass

    def get_episode_runner(self):
        pass

    def compute_action(self, observation, task_action_count):
        pass

    def train(self, storage_buffer):
        pass

    def save(self, output_path_dir, task_id, task_total_steps):
        pass

    def load(self, model_path):
        pass
