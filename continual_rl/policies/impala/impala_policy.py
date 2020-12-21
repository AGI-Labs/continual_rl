from continual_rl.policies.policy_base import PolicyBase
from continual_rl.policies.impala.impala_policy_config import ImpalaPolicyConfig
from continual_rl.policies.impala.impala_environment_runner import ImpalaEnvironmentRunner
from continual_rl.policies.impala.nets import ImpalaNet
import numpy as np
import functools


class ImpalaPolicy(PolicyBase):
    """
    With IMPALA, the parallelism is the point, so rather than splitting it up into compute_action and train like normal,
    just let the existing IMPALA implementation handle it all.
    This policy is now basically a container for the network that gets run to compute the action.
    """
    def __init__(self, config: ImpalaPolicyConfig, observation_space, action_spaces):  # Switch to your config type
        super().__init__()
        self._config = config
        self._common_action_size = int(np.array([action.n for action in action_spaces.values()]).max())
        self.policy_class = self._get_policy_class(self._common_action_size)

        # A place to persist the policy info between tasks
        self.replay_buffers = None
        self.model = None
        self.learner_model = None
        self.optimizer = None

    def _create_max_action_class(self, cls, max_actions):
        """
        The policy needs to have access to both the max number of actions and the current number,
        but the IMPALA signature only admits the second. Rather than piping the max all the way through, just patch it
        in here.
        """
        class MaxActionNetWrapper(cls):
            __init__ = functools.partialmethod(cls.__init__, max_actions=max_actions,
                                               net_flavor=self._config.net_flavor)

        return MaxActionNetWrapper

    def _get_policy_class(self, common_action_size):
        policy_net = self._create_max_action_class(ImpalaNet, common_action_size)
        return policy_net

    def get_environment_runner(self):
        return ImpalaEnvironmentRunner(self._config, self)

    def compute_action(self, observation, action_space_id, last_timestep_data, eval_mode):
        pass

    def train(self, storage_buffer):
        pass

    def save(self, output_path_dir, task_id, task_total_steps):
        pass

    def load(self, model_path):
        pass
