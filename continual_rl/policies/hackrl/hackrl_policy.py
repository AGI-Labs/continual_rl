from continual_rl.policies.policy_base import PolicyBase
from continual_rl.policies.hackrl.hackrl_environment_runner import HackRLEnvironmentRunner
from continual_rl.utils.utils import Utils
from hackrl.experiment import HackRLLearner

from .hackrl_policy_config import HackRLPolicyConfig


class HackRLPolicy(PolicyBase):
    """
    Holds the permanent information needed by HackRL.
    # TODO: can probably move more into here, out of environment_runner. Leaving the changes minimal for now.
    """
    def __init__(self, config: HackRLPolicyConfig, observation_space, action_spaces):  # Switch to your config type
        super().__init__()
        self._config = config
        self._observation_space = observation_space
        self._action_spaces = action_spaces

        # HackRL uses "action_space" to mean the list of available actions, instead of the OpenAI definition. Doing a 
        # quick-and-dirty conversion from OpenAI to HackRL (TODO: probably should fix in HackRL to be consistent)
        common_action_space = Utils.get_max_discrete_action_space(action_spaces)
        action_list = list(range(common_action_space.n))
        self.learner = HackRLLearner(self._config.omega_conf, observation_space.shape, action_list)

    def get_environment_runner(self, task_spec):
        return HackRLEnvironmentRunner(self._config, self)

    def compute_action(self, observation, task_id, action_space_id, last_timestep_data, eval_mode):
        raise NotImplementedError

    def train(self, storage_buffer):
        # Handled by hackrl
        pass

    def save(self, output_path_dir, cycle_id, task_id, task_total_steps):
        pass  # TODO

    def load(self, output_path_dir):
        pass  # TODO
