import gym
import time
from continual_rl.policies.policy_base import PolicyBase
from continual_rl.policies.play.play_policy_config import PlayPolicyConfig
from continual_rl.policies.play.play_timestep_data import PlayTimestepData
from continual_rl.policies.play.play_environment_runner import PlayEnvironmentRunner


class PlayPolicy(PolicyBase):
    """
    A "policy" that allows a human to play in place of an agent.
    Based on: https://github.com/openai/gym/blob/master/examples/agents/keyboard_agent.py
    """
    # Unfortunately these are most convenient for QWERTY. Couldn't find a good way to access the virtual key codes
    # to determine which physical key was pressed, instead of using the character.
    PAUSE = "pause"
    ATARI_KEY_BINDINGS = {ord('0'): 0,  # No-op
                          ord(' '): 1,  # Fire
                          ord('w'): 2,  # Up
                          ord('d'): 3,  # Right
                          ord('a'): 4,  # Left
                          ord('s'): 5,  # Down

                          ord('p'): PAUSE  # Pause  (Non-official)
                          }
    # This set is just the easiest subset. The rest of the bindings are available at the bottom of
    # https://github.com/openai/gym/blob/master/gym/envs/atari/atari_env.py

    def __init__(self, config: PlayPolicyConfig, observation_space, action_spaces):
        super().__init__(config)

        if not isinstance(action_spaces[0], gym.spaces.Discrete):
            raise Exception('Keyboard agent only supports discrete action spaces')

        self._config = config
        self._bindings = {"atari": self.ATARI_KEY_BINDINGS}
        self._action = None
        self._env = None
        self._paused = False

    def _on_key_press(self, key, mod):
        bindings = self._bindings[self._config.key_bindings]
        if key in bindings:
            self._action = bindings[key]

            if self._action == self.PAUSE:
                self._paused = ~self._paused
                self._action = None

    def _on_key_release(self, key, mod):
        if self._action != self.PAUSE:  # Pause is turned off by pressing the pause button again
            self._action = None

    def on_env_ready(self, env):
        """
        The env runner will call this when it is setup and ready to receive input.
        """
        self._env = env

    def get_environment_runner(self, task_spec):
        # Timesteps are how often we update the event handlers and render
        runner = PlayEnvironmentRunner(policy=self, timesteps_per_collection=1, on_key_press=self._on_key_press,
                                       on_key_release=self._on_key_release)
        return runner

    def compute_action(self, observation, task_id, action_space_id, last_timestep_data, eval_mode):
        action = None

        if self._env is not None:
            # If we're in the paused state, just wait
            while self._paused:
                time.sleep(0.01)
                self._env.render()

            action = self._action

        if action is None:
            action = 0

        time.sleep(0.05)  # Slow the game down to more human-speed

        return [action], PlayTimestepData()

    def train(self, storage_buffer):
        pass

    def save(self, output_path_dir, cycle_id, task_id, task_total_steps):
        pass

    def load(self, output_path_dir):
        pass
