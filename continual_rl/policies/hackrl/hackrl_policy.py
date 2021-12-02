import os
import torch
from continual_rl.policies.policy_base import PolicyBase
from continual_rl.policies.hackrl.hackrl_environment_runner import HackRLEnvironmentRunner
from continual_rl.utils.utils import Utils
from hackrl.experiment import HackRLLearner
from continual_rl.policies.clear.clear_monobeast import ClearReplayHandler


from .hackrl_policy_config import HackRLPolicyConfig


class HackRLPolicy(PolicyBase):
    """
    Holds the permanent information needed by HackRL.
    # TODO: can probably move more into here, out of environment_runner. Leaving the changes minimal for now.
    """
    def __init__(self, config: HackRLPolicyConfig, observation_space, action_spaces):  # Switch to your config type
        super().__init__()

        # Convenience remappings, to use hackrl's namings
        config.omega_conf.localdir = config.output_dir
        config.omega_conf.group = config.output_dir

        # Populate the savedir such that the model is saved/loaded from the correct place
        project_name = os.path.normpath(config.output_dir).replace(os.path.sep, '-')
        config.omega_conf.savedir = os.path.join(config.omega_conf.savedir_prefix, project_name)

        self._config = config
        self._observation_space = observation_space
        self._action_spaces = action_spaces

        # HackRL uses "action_space" to mean the list of available actions, instead of the OpenAI definition. Doing a 
        # quick-and-dirty conversion from OpenAI to HackRL (TODO: probably should fix in HackRL to be consistent)
        common_action_space = Utils.get_max_discrete_action_space(action_spaces)
        action_list = list(range(common_action_space.n))

        # TODO: this num_actors/batch_size thing needs...a lot of clarity. And it would be confusing if a user set it but it gets overridden
        plugin = None
        if config.use_clear_plugin:
            assert self._config.clear_config is not None, "If clear plugin is configured, config expected."
            self._config.clear_config.set_output_dir(self._config.output_dir)
            plugin = ClearReplayHandler(self, self._config.clear_config, observation_space, action_spaces)

        self.learner = HackRLLearner(self._config.omega_conf, observation_space.shape, action_list, learner_plugin=plugin)

    def initial_state(self, batch_size):
        # TODO: doesn't exactly fit here but...going with it for now
        return self.learner._model.initial_state(batch_size)

    def create_buffer_specs(self, unroll_length, obs_space, num_actions):
        """
        Required by ClearReplayHandler - defines what this policy needs to store in a replay buffer
        """
        T = unroll_length
        specs = dict(
            reward=dict(size=(T + 1,), dtype=torch.float32),
            done=dict(size=(T + 1,), dtype=torch.bool),
            episode_return=dict(size=(T + 1,), dtype=torch.float32),
            episode_step=dict(size=(T + 1,), dtype=torch.int32),
            policy_logits=dict(size=(T + 1, num_actions), dtype=torch.float32),
            baseline=dict(size=(T + 1,), dtype=torch.float32),
            last_action=dict(size=(T + 1,), dtype=torch.int64),
            action=dict(size=(T + 1,), dtype=torch.int64),
        )

        # TODO: enforce that any observation exists in the observation space (otherwise it won't be cached in the replay buffer)? (unless it's just for logging....)
        if hasattr(obs_space, "spaces"):
            for obs_name, obs_info in obs_space.spaces.items():
                specs[obs_name] = dict(size=(T+1, *obs_info.shape), dtype=Utils.convert_numpy_dtype_to_torch(obs_info.dtype))
        else:
            specs["frame"] = dict(size=(T + 1, *obs_space.shape), dtype=torch.uint8)

        return specs

    def cleanup(self, task_spec):
        self.learner.cleanup(task_spec)

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
