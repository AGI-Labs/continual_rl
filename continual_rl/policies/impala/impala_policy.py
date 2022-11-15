import os
import copy
from continual_rl.policies.policy_base import PolicyBase
from continual_rl.policies.impala.impala_policy_config import ImpalaPolicyConfig
from continual_rl.policies.impala.impala_environment_runner import ImpalaEnvironmentRunner
from continual_rl.policies.impala.nets import ImpalaNet
from continual_rl.policies.impala.torchbeast.monobeast import Monobeast
from continual_rl.utils.utils import Utils


class ImpalaPolicy(PolicyBase):
    """
    With IMPALA, the parallelism is the point, so rather than splitting it up into compute_action and train like normal,
    just let the existing IMPALA implementation handle it all.
    This policy is now basically a container for the Monobeast object itself, which holds persistent information
    (e.g. the model and the replay buffers).
    """
    def __init__(self, config: ImpalaPolicyConfig, observation_space, action_spaces, impala_class: Monobeast = None,
                 policy_net_class: ImpalaNet = None):
        super().__init__(config)
        self._config = config
        self._action_spaces = action_spaces

        model_flags = self._create_model_flags()

        if impala_class is None:
            impala_class = Monobeast

        if policy_net_class is None:
            policy_net_class = ImpalaNet

        self.impala_trainer = impala_class(model_flags, observation_space, action_spaces, policy_net_class)

    def _create_model_flags(self):
        """
        Finishes populating the config to contain the rest of the flags used by IMPALA in the creation of the model.
        """
        # torchbeast will change flags, so copy it so config remains unchanged for other tasks.
        flags = copy.deepcopy(self._config)
        flags.savedir = str(self._config.output_dir)
        return flags

    def get_environment_runner(self, task_spec):
        return ImpalaEnvironmentRunner(self._config, self)

    def compute_action(self, observation, task_id, action_space_id, last_timestep_data, eval_mode):
        pass

    def train(self, storage_buffer):
        pass

    def save(self, output_path_dir, cycle_id, task_id, task_total_steps):
        self.impala_trainer.save(output_path_dir)

    def load(self, output_path_dir):
        self.impala_trainer.load(output_path_dir)
