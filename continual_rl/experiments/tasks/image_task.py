import torch
import numpy as np
from continual_rl.experiments.tasks.task_base import TaskBase
from continual_rl.experiments.tasks.preprocessor_base import PreprocessorBase
from continual_rl.utils.utils import Utils
from continual_rl.utils.env_wrappers import FrameStack, WarpFrame, ImageToPyTorch


class ImagePreprocessor(PreprocessorBase):
    def __init__(self, time_batch_size, image_size, grayscale, env_spec, resize_interp_method, dict_space_key=None):
        self._resize_interp_method = resize_interp_method
        self._dict_space_key = dict_space_key
        self.env_spec = self._wrap_env(env_spec, time_batch_size, image_size, grayscale)

        dummy_env, _ = Utils.make_env(self.env_spec)
        observation_space = dummy_env.observation_space
        dummy_env.close()
        del dummy_env

        super().__init__(observation_space)

    def _wrap_env(self, env_spec, time_batch_size, image_size, grayscale):
        # TODO: np.float32 is to handle depth (which is a float) - hacky
        # Leverage the existing env wrappers for simplicity
        frame_stacked_env_spec = lambda: FrameStack(ImageToPyTorch(
            WarpFrame(Utils.make_env(env_spec)[0], image_size[1], image_size[0], grayscale=grayscale,
                      resize_interp_method=self._resize_interp_method, dict_space_key=self._dict_space_key),
            dict_space_key=self._dict_space_key, dtype=np.float32), time_batch_size)
        return frame_stacked_env_spec

    def preprocess(self, batched_env_image):
        """
        The preprocessed image will have values in range [0, 255] and shape [batch, time, channels, width, height].
        Handled as a batch for speed.
        """
        processed_image = torch.stack([image.to_tensor() for image in batched_env_image])
        return processed_image

    def render_episode(self, episode_observations):
        """
        Turn a list of observations gathered from the episode into a video that can be saved off to view behavior.
        """
        return torch.stack(episode_observations).unsqueeze(0)


class ImageTask(TaskBase):
    def __init__(self, task_id, action_space_id, env_spec, num_timesteps, time_batch_size, eval_mode,
                 image_size, grayscale, continual_eval=True, resize_interp_method="INTER_AREA",
                 demonstration_task=False, dict_space_key=None):
        preprocessor = ImagePreprocessor(time_batch_size, image_size, grayscale, env_spec, resize_interp_method,
                                         dict_space_key=dict_space_key)

        dummy_env, _ = Utils.make_env(preprocessor.env_spec)
        action_space = dummy_env.action_space
        dummy_env.close()
        del dummy_env

        super().__init__(task_id, action_space_id, preprocessor, preprocessor.env_spec, preprocessor.observation_space,
                         action_space, num_timesteps, eval_mode, continual_eval=continual_eval,
                         demonstration_task=demonstration_task)
