import torch
import torchvision
from continual_rl.experiments.tasks.task_base import TaskBase
from continual_rl.utils.utils import Utils


class ImageTask(TaskBase):
    def __init__(self, env_spec, num_timesteps, time_batch_size, eval_mode, output_dir, image_size, grayscale):
        dummy_env = Utils.make_env(env_spec)
        obs_size = [time_batch_size, *image_size]  # We transform the input into this size (does not include batch)
        action_size = dummy_env.action_space.n

        super().__init__(env_spec, obs_size, action_size, time_batch_size, num_timesteps, eval_mode, output_dir)

        transforms = [torchvision.transforms.ToPILImage(),
                      torchvision.transforms.Resize(obs_size[2:]),
                      torchvision.transforms.ToTensor()]

        if grayscale:  # TODO: make consistent with image_size input
            self._transform = transforms.insert(1, torchvision.transforms.Grayscale())
            obs_size[1] = 1

        self._transform = torchvision.transforms.Compose(transforms)

    def preprocess(self, single_env_image):
        single_env_image = torch.Tensor(single_env_image)

        if single_env_image.shape[0] == 1:
            # Assume we're in the [1, w, h] case
            permuted_image = single_env_image.repeat(3, 1, 1)
        elif single_env_image.shape[0] == 3:
            # Assume we're in the [3, w, h] case
            permuted_image = single_env_image
        else:
            # Assume we're in [w, h, c] case (currently only supports c = 3)
            permuted_image = single_env_image.permute(2, 0, 1)

        # To make sure the transformation works properly, ensure we have a [C, H, W] Tensor
        # [H, W, C] numpy array should also work, but is untested.
        assert permuted_image.shape[0] == 3 and isinstance(permuted_image, torch.Tensor)
        transformed_image = self._transform(permuted_image)

        return transformed_image
