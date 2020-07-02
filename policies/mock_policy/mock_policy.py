from policies.policy_base import PolicyBase


class MockPolicy(PolicyBase):
    """
    A simple implementation of policy as a sample of how policies can be created, and as a mock for unit tests.
    Refer to policy_base itself for more detailed descriptions of the method signatures.
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
