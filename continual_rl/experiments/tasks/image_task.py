import torch
import torchvision
from continual_rl.experiments.tasks.task_base import TaskBase
from continual_rl.experiments.tasks.preprocessor_base import PreprocessorBase
from continual_rl.utils.utils import Utils
from gym.spaces.box import Box


class ImagePreprocessor(PreprocessorBase):
    def __init__(self, image_size, grayscale):
        channels = 1 if grayscale else 3

        # We transform the input into this size (does not include batch)
        obs_space = Box(low=0, high=1.0, shape=[channels, *image_size])

        super().__init__(obs_space)

        transforms = [torchvision.transforms.ToPILImage(),
                      torchvision.transforms.Resize(obs_space.shape[1:]),
                      torchvision.transforms.ToTensor()]

        if grayscale:
            self._transform = transforms.insert(1, torchvision.transforms.Grayscale())

        self._transform = torchvision.transforms.Compose(transforms)
        self._grayscale = grayscale

    def preprocess(self, single_env_image):
        """
        The preprocessed image will have values in range [0, 1]
        """
        single_env_image = torch.Tensor(single_env_image)

        if single_env_image.shape[0] == 1 and not self._grayscale:
            # Assume we're in the [1, w, h] case. Fake 3 dims for the non-grayscale case
            permuted_image = single_env_image.repeat(3, 1, 1)
        elif single_env_image.shape[0] == 1 or single_env_image.shape[0] == 3:
            # Assume we're in the [3, w, h] case or [1, w, h] + grayscale, so just keep things as they are
            permuted_image = single_env_image
        else:
            # Assume we're in [w, h, c] case, rearrange then verify
            permuted_image = single_env_image.permute(2, 0, 1)
            assert permuted_image.shape[0] == 3 or (permuted_image.shape[0] == 1 and self._grayscale), \
                f"Unexpected image input shape: {single_env_image}"

        transformed_image = self._transform(permuted_image)

        return transformed_image

    def render_episode(self, episode_observations):
        """
        Turn a list of observations gathered from the episode into a video that can be saved off to view behavior.
        """
        return torch.stack(episode_observations).unsqueeze(0)


class ImageTask(TaskBase):
    def __init__(self, action_space_id, env_spec, num_timesteps, time_batch_size, eval_mode, image_size, grayscale):
        dummy_env, _ = Utils.make_env(env_spec)
        action_space = dummy_env.action_space
        preprocessor = ImagePreprocessor(image_size, grayscale)
        super().__init__(action_space_id, preprocessor, env_spec, preprocessor.observation_space, action_space,
                         time_batch_size, num_timesteps, eval_mode)
