import gym
from continual_rl.policies.policy_base import PolicyBase
from continual_rl.policies.play.play_policy_config import PlayPolicyConfig
from continual_rl.policies.play.play_timestep_data import PlayTimestepData
from continual_rl.policies.play.play_environment_runner import PlayEnvironmentRunner


class PlayPolicy(PolicyBase):
    """
    A "policy" that allows a human to play in place of an agent.
    Based on: https://github.com/openai/gym/blob/master/examples/agents/keyboard_agent.py
    """
    def __init__(self, config: PlayPolicyConfig, observation_space, action_spaces):
        super().__init__()

        if not isinstance(action_spaces[0], gym.spaces.Discrete):
            raise Exception('Keyboard agent only supports discrete action spaces')

        self._key_pressed = None

    def _on_key_press(self, key, mod):
        self._key_pressed = int(key - ord('0'))

    def _on_key_release(self, key, mod):
        self._key_pressed = None

    def get_environment_runner(self):
        runner = PlayEnvironmentRunner(policy=self, timesteps_per_collection=1000, on_key_press=self._on_key_press,
                                       on_key_release=self._on_key_release)  # Timesteps arbitrary
        return runner

    def compute_action(self, observation, action_space_id, last_timestep_data, eval_mode):
        while self._key_pressed is None:
            pass
        return [0], PlayTimestepData()

    def train(self, storage_buffer):
        pass

    def save(self, output_path_dir, task_id, task_total_steps):
        pass

    def load(self, model_path):
        pass
