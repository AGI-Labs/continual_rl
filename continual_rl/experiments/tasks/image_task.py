import torch
from continual_rl.experiments.tasks.task_base import TaskBase


class ImageTask(TaskBase):
    def __init__(self, env_spec, num_timesteps, time_batch_size, eval_mode, output_dir, image_size):
        import torchvision
        dummy_env = self._make_env(env_spec)
        obs_size = image_size  # We transform the input into this size
        action_size = dummy_env.action_space.n

        super().__init__(env_spec, obs_size, action_size, num_timesteps, time_batch_size, eval_mode, output_dir)

        # TODO: standard normalization?
        self._transform = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                          torchvision.transforms.Resize(obs_size[1:]),
                                                          torchvision.transforms.ToTensor()])

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
