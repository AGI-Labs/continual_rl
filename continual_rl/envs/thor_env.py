import gym
from gym.spaces import Box, Discrete
import numpy as np
from ai2thor.controller import Controller
from ai2thor.interact import DefaultActions
from continual_rl.utils.utils import Utils


class ThorFindAndPickEnv(gym.Env):
    OBJECT_TO_FIND_IDS = {}

    def __init__(self, scene_name, object_to_find):
        # TODO: what obs size?
        width = 84
        height = 84
        self._grid_size = 0.25
        self._scene_name = scene_name
        self._max_episode_steps = 200

        self.observation_space = Box(low=0, high=255, shape=(width, height, 3), dtype=np.uint8)
        self.action_space = Discrete(5)

        self._controller = Controller(scene=self._scene_name, gridSize=self._grid_size, width=width, height=height)
        self._object_to_find = object_to_find
        self._object_to_find_representation = self._get_representation_for_object(object_to_find)
        self._last_event = None
        self._reward = 1
        self._episode_steps = 0

    def __del__(self):
        self.close()

    def seed(self, seed=None):
        Utils.seed(seed=seed)  # TODO: untested so far

    @classmethod
    def _get_representation_for_object(cls, object_to_find):
        if object_to_find not in cls.OBJECT_TO_FIND_IDS:
            ids = cls.OBJECT_TO_FIND_IDS.values()
            if len(ids) == 0:
                max_id = 0
            else:
                max_id = max(ids)
            cls.OBJECT_TO_FIND_IDS[object_to_find] = max_id + 1

        return cls.OBJECT_TO_FIND_IDS[object_to_find]

    def _get_observation(self, frame):
        # We copy the last frame to re-allocate the memory, because otherwise the conversion to torch Tensor
        # is having negative stride issues: https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663
        observation = frame.copy()

        # Just overwriting the first pixel, as a hacky way of avoiding needing a separate input vector
        observation[0][0][...] = self._object_to_find_representation
        return observation

    def step(self, action):
        thor_action_data = None
        object_found = False

        if action == 0:
            thor_action_data = {"action": DefaultActions.MoveAhead.name, "moveMagnitude": self._grid_size}
        elif action == 1:
            thor_action_data = {"action": DefaultActions.MoveBack.name, "moveMagnitude": self._grid_size}
        elif action == 2:
            thor_action_data = {"action": DefaultActions.RotateLeft.name, "rotation": 90}
        elif action == 3:
            thor_action_data = {"action": DefaultActions.RotateRight.name, "rotation": 90}
        elif action == 4:
            if self._last_event is None or self._object_to_find not in [o['objectType'] for o in self._last_event.metadata['objects']]:
                # Intended to be a no-op
                thor_action_data = {"action": DefaultActions.MoveAhead.name, "moveMagnitude": 0}
            else:
                # We'll find all instances of the object we're trying to pick up, and try pick up the first
                # TODO: is there a way to only find the objects in view or in reach?
                objectIds = [o for o in self._last_event.metadata['objects'] if o['objectType'] == self._object_to_find]
                thor_action_data = {"action": "PickupObject", 'objectId': objectIds[0]['objectId']}
                object_found = True

        self._last_event = self._controller.step(**thor_action_data)
        object_picked_up = object_found and self._last_event.metadata['lastActionSuccess']

        next_observation = self._get_observation(self._last_event.frame)
        reward = self._reward if object_picked_up else 0
        done = object_picked_up or self._episode_steps > self._max_episode_steps

        # We start the reward at 1 and decay for every step taken. (TODO: this is arbitrary...)
        self._reward *= .99
        self._episode_steps += 1

        return next_observation, reward, done, {}

    def reset(self):
        self._episode_steps = 0
        self._reward = 1
        self._last_event = self._controller.reset(self._scene_name)
        self._last_event = self._controller.step("InitialRandomSpawn")
        return self._get_observation(self._last_event.frame)

    def render(self, mode='human', close=False):
        pass

    def close(self):
        try:
            self._controller.stop()  # TODO: this gives bad file descriptor error...not sure why
        except OSError:
            print("OSError (likely bad file descriptor). Just letting it go....")
            pass
