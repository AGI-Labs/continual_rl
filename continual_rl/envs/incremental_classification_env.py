import gym
from continual_rl.utils.utils import Utils
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from gym.spaces import Box, Discrete
import numpy as np
from enum import Enum
import torch
import uuid

import cv2

cv2.setNumThreads(0)  # TODO: does this fix anything? (Does anything use cv2 downstream?)


class DatasetIds(Enum):
    MNIST_TRAIN = 1
    MNIST_TEST = 1


class IncrementalClassificationEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, data_dir, num_steps_per_episode, dataset_id=DatasetIds.MNIST_TRAIN, allowed_class_ids=None):
        self._data_dir = data_dir
        self._current_step = 0
        self._num_steps_per_episode = num_steps_per_episode
        self._last_observation_pair = None  # Tuple of (observation, target)
        self.unique_id = uuid.uuid4()

        self.observation_space = Box(low=-1, high=1, shape=(28, 28), dtype=np.float32)  # TODO check these (and check normalization above)
        self.action_space = Discrete(10)

        self._initialize_loader(dataset_id, allowed_class_ids)  # None means "use all"
        self._data_iter = None  # Lazy load because pickling iters doesn't work (for multiprocessing)
        #self._set_next_iter()

    def _initialize_loader(self, dataset_id, allowed_class_ids):
        #kwargs = {'pin_memory': False} if self._use_cuda else {}  # TODO: dive into this
        kwargs = {'pin_memory': False}  # TODO: dive into this

        if dataset_id == DatasetIds.MNIST_TRAIN:
            dataset = datasets.MNIST(self._data_dir, train=True, download=True,  #TODO specify location
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))
        elif dataset_id == DatasetIds.MNIST_TEST:
            dataset =  datasets.MNIST(self._data_dir, train=False, download=True,
              transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]))
        else:
            raise LookupError(f"Dataset ID {dataset_id} not recognized")

        if allowed_class_ids is None:
            subset_dataset = dataset
        else:
            # Get the desired subset according to allowed_class_ids
            # Courtesy https://discuss.pytorch.org/t/how-to-use-one-class-of-number-in-mnist/26276/12
            index_mask = None  # Will have 1 in allowed entries, 0 elsewhere
            for id in allowed_class_ids:
                new_indices = dataset.targets == id

                if index_mask is None:
                    index_mask = new_indices
                else:
                    index_mask += new_indices

            subset_dataset = torch.utils.data.dataset.Subset(dataset, np.where(index_mask==1)[0])

        # num_workers=0 disables multiprocessing, which is necessary to thread further up the stack (daemonic processes are not allowed to have children)
        # Also I think the multiprocessing causes issues with SubProcVecEnv - debugging in progress (TODO)
        # TODO: if this fixes my seeding issue...then maybe I have an issue in my continual_rl collection process?q
        self._data_loader = DataLoader(subset_dataset, batch_size=1, shuffle=True, num_workers=0, **kwargs)

    def seed(self, seed=None):
        Utils.seed(seed=seed)

    def _set_next_iter(self):
        self._data_iter = iter(self._data_loader)

    def _get_current_observation_pair(self):
        self._current_step += 1

        if self._data_iter is None:
            self._set_next_iter()

        done = self._current_step >= self._num_steps_per_episode + 1  # Off by one because reset() will trigger the first

        try:
            result = next(self._data_iter)
        except StopIteration:
            self._set_next_iter()
            result = next(self._data_iter)

        return result, done

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.numpy()
        if isinstance(action, int) or isinstance(action, np.int64):
            action = np.array([action]).squeeze()

        assert isinstance(action, np.ndarray), "Action is expected to be a numpy object"
        reward = 0

        correct_action = self._last_observation_pair[1].squeeze().cpu().numpy()
        if correct_action == action:  # The "action" is the selected class
            reward = 1

        self._last_observation_pair, done = self._get_current_observation_pair()

        next_observation = self._last_observation_pair[0].squeeze(0)  # First 0 is from the (obs, target) tuple, the second is to remove the batch

        return next_observation, reward, done, {"correct_action": correct_action}

    def reset(self):
        self._current_step = 0
        self._last_observation_pair, _ = self._get_current_observation_pair()
        return self._last_observation_pair[0].squeeze(0)  # Same reason as in step

    def render(self, mode='human', close=False):
        pass
