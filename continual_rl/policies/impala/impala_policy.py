import os
import copy
import functools
from continual_rl.policies.policy_base import PolicyBase
from continual_rl.policies.impala.impala_policy_config import ImpalaPolicyConfig
from continual_rl.policies.impala.impala_environment_runner import ImpalaEnvironmentRunner
from continual_rl.policies.impala.nets import ImpalaNet
from continual_rl.policies.impala.torchbeast.monobeast import Monobeast


class ImpalaPolicy(PolicyBase):
    """
    With IMPALA, the parallelism is the point, so rather than splitting it up into compute_action and train like normal,
    just let the existing IMPALA implementation handle it all.
    This policy is now basically a container for the Monobeast object itself, which holds persistent information
    (e.g. the model and the replay buffers).
    """
    def __init__(self, config: ImpalaPolicyConfig, observation_space, action_spaces):  # Switch to your config type
        super().__init__()
        self._config = config
        self._common_action_space = self._get_max_action_space(action_spaces)

        os.environ["OMP_NUM_THREADS"] = "1"  # Necessary for multithreading.

        model_flags = self._create_model_flags()
        policy_class = self._get_policy_class(self._common_action_space)
        self.impala_trainer = Monobeast(model_flags, observation_space, action_spaces, policy_class)

    def _create_model_flags(self):
        """
        Finishes populating the config to contain the rest of the flags used by IMPALA in the creation of the model.
        """
        # torchbeast will change flags, so copy it so config remains unchanged for other tasks.
        flags = copy.deepcopy(self._config)
        flags.savedir = str(self._config.output_dir)

        # Arbitrary - the output_dir is already unique and consistent
        flags.xpid = "impala"

        # Currently always initialized, but only used if use_clear==True
        # We have one replay entry per unroll, split between actors
        flags.replay_buffer_size = max(flags.num_actors,
                                       self._config.replay_buffer_frames // flags.unroll_length) if flags.use_clear else 0

        # CLEAR specifies 1
        flags.num_learner_threads = 1 if flags.use_clear else self._config.num_learner_threads

        return flags

    def _get_max_action_space(self, action_spaces):
        max_action_space = None
        for action_space in action_spaces.values():
            if max_action_space is None or action_space.n > max_action_space.n:
                max_action_space = action_space
        return max_action_space

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
