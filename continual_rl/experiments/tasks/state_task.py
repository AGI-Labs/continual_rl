import torch
import gym
from continual_rl.experiments.tasks.task_base import TaskBase
from continual_rl.experiments.tasks.preprocessor_base import PreprocessorBase
from continual_rl.utils.utils import Utils
from continual_rl.utils.env_wrappers import FrameStack


class StateToPyTorch(gym.ObservationWrapper):
    # TODO (Issue 50): If used after LazyFrames, seems to negate the point of LazyFrames
    # As in, LazyFrames provides no benefit?
    # For now switching this to return a Tensor and calling it *before* FrameStack...

    def __init__(self, env, dict_space_key=None):
        super().__init__(env)
        self._key = dict_space_key

    def observation(self, observation):
        state_observation = observation if self._key is None else observation[self._key]
        processed_observation = torch.as_tensor(state_observation)

        if self._key is not None:
            observation[self._key] = processed_observation
        else:
            observation = processed_observation

        return observation


class StatePreprocessor(PreprocessorBase):
    def __init__(self, time_batch_size, env_spec):
        self.env_spec = self._wrap_env(env_spec, time_batch_size)
        dummy_env, _ = Utils.make_env(self.env_spec)
        super().__init__(dummy_env.observation_space)

    def _wrap_env(self, env_spec, time_batch_size):
        # Leverage the existing env wrappers for simplicity
        frame_stacked_env_spec = lambda: FrameStack(StateToPyTorch(Utils.make_env(env_spec)[0]), time_batch_size)
        return frame_stacked_env_spec

    def preprocess(self, batched_env_states):
        """
        The preprocessed image will have values in range [0, 255] and shape [batch, time, channels, width, height].
        Handled as a batch for speed.
        """
        processed_state = torch.stack([state.to_tensor() for state in batched_env_states])
        return processed_state

    def render_episode(self, episode_observations):
        """
        Turn a list of observations gathered from the episode into a video that can be saved off to view behavior.
        """
        raise NotImplementedError("No image to render for state-based experiments")


class StateTask(TaskBase):
    def __init__(self, task_id, action_space_id, env_spec, num_timesteps, time_batch_size, eval_mode,
                 continual_eval=True):
        preprocessor = StatePreprocessor(time_batch_size, env_spec)
        dummy_env, _ = Utils.make_env(preprocessor.env_spec)

        super().__init__(task_id, action_space_id, preprocessor, preprocessor.env_spec, preprocessor.observation_space,
                         dummy_env.action_space, num_timesteps, eval_mode, continual_eval=continual_eval)
