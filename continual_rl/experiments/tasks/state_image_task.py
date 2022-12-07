import torch
from continual_rl.experiments.tasks.task_base import TaskBase
from continual_rl.experiments.tasks.preprocessor_base import PreprocessorBase
from continual_rl.utils.utils import Utils
from continual_rl.experiments.tasks.state_task import StateToPyTorch
from continual_rl.utils.env_wrappers import FrameStack, WarpFrame, ImageToPyTorch


class StateImagePreprocessor(PreprocessorBase):
    def __init__(self, time_batch_size, image_size, grayscale, env_spec, resize_interp_method):
        self.env_spec = self._wrap_env(env_spec, time_batch_size, image_size, grayscale, resize_interp_method)

        # Clean up the dummy env immediately because multiple creation causes issues with some envs
        dummy_env, _ = Utils.make_env(self.env_spec)
        observation_space = dummy_env.observation_space
        dummy_env.close()
        del dummy_env

        super().__init__(observation_space)

    def _wrap_env(self, env_spec, time_batch_size, image_size, grayscale, resize_interp_method):
        def env_wrapper(env):
            # Each handler operates only on the dict_space_key given, and leaves the rest unchanged
            # Thus we can pass the result from one into the next, and operate sequentially on all keys
            state_handler = StateToPyTorch(env, dict_space_key="state_vector")
            image_handler = ImageToPyTorch(WarpFrame(state_handler, image_size[1], image_size[0], grayscale=grayscale,
                            resize_interp_method=resize_interp_method, dict_space_key="image"), dict_space_key="image")  # TODO spowers: handle better...just testing
            frame_stack = FrameStack(image_handler, time_batch_size)  # Will stack all keys, in the dict case
            return frame_stack

        return lambda: env_wrapper(Utils.make_env(env_spec)[0])

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
        return torch.stack(episode_observations).unsqueeze(0)


class StateImageTask(TaskBase):
    """
    To handle environments that have both a state and an image.
    Currently assumes state uses the key "state_vector" and image uses the key "image"
    """
    def __init__(self, task_id, action_space_id, env_spec, num_timesteps, time_batch_size, eval_mode,
                 image_size, grayscale, continual_eval=True, resize_interp_method="INTER_AREA",
                 demonstration_task=False, continual_eval_num_returns=10):
        preprocessor = StateImagePreprocessor(time_batch_size, image_size, grayscale, env_spec, resize_interp_method)

        # Clean up the dummy env immediately because multiple creation causes issues with some envs
        dummy_env, _ = Utils.make_env(preprocessor.env_spec)
        action_space = dummy_env.action_space
        dummy_env.close()
        del dummy_env

        super().__init__(task_id, action_space_id, preprocessor, preprocessor.env_spec, preprocessor.observation_space,
                         action_space, num_timesteps, eval_mode, continual_eval=continual_eval,
                         demonstration_task=demonstration_task, continual_eval_num_returns=continual_eval_num_returns)
