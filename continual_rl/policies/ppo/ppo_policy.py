from continual_rl.policies.policy_base import PolicyBase
from continual_rl.policies.ppo.ppo_policy_config import PPOPolicyConfig  # Switch to your config type
from continual_rl.policies.ppo.a2c_ppo_acktr_gail.ppo import PPO
from continual_rl.policies.ppo.a2c_ppo_acktr_gail.model import Policy
from continual_rl.policies.ppo.a2c_ppo_acktr_gail.storage import RolloutStorage


class PPOPolicy(PolicyBase):
    """
    A simple implementation of policy as a sample of how policies can be created.
    Refer to policy_base itself for more detailed descriptions of the method signatures.
    """
    def __init__(self, config: PPOPolicyConfig, observation_size, action_spaces):  # Switch to your config type
        super().__init__()
        common_action_space = self._get_common_action_space(action_spaces)
        self._actor_critic = Policy(obs_shape=observation_size,
                                    action_space=common_action_space)
        self._rollout_storage = RolloutStorage(num_steps=1,
                                               num_processes=1,
                                               obs_shape=observation_size,
                                               action_space=common_action_space,
                                               recurrent_hidden_state_size=1)

    def _get_common_action_space(self, action_spaces):
        common_action_space = None
        for action_space in action_spaces:
            if common_action_space is None:
                common_action_space = action_space
            assert common_action_space == action_space, \
                "PPO currently only supports environments with the same action spaces."

    def get_environment_runner(self):
        pass

    def compute_action(self, observation, action_space_id, last_timestep_data):
        pass

    def train(self, storage_buffer):
        pass

    def save(self, output_path_dir, task_id, task_total_steps):
        pass

    def load(self, model_path):
        pass
