import gym
from gym.spaces import Box, Discrete
import numpy as np
import os
from ai2thor.controller import Controller
from ai2thor.interact import DefaultActions
from continual_rl.utils.utils import Utils


class ThorFindPickPlaceEnv(gym.Env):

    # https://ai2thor.allenai.org/ithor/documentation/overview/examples/#moveahead
    # Kitchens: FloorPlan1 - FloorPlan30
    # Living rooms: FloorPlan201 - FloorPlan230
    # Bedrooms: FloorPlan301 - FloorPlan330
    # Bathrooms: FloorPLan401 - FloorPlan430

    def __init__(self, scene_name,
                 objects_to_find,
                 receptacle_object,
                 goal_conditioned=False,
                 clear_receptacle_object=False,
                 use_cached_goal=False,
                 preopen_receptacle_object=True,
                 ):
        # TODO: what obs size?
        self.width = 84
        self.height = 84
        self._grid_size = 0.25
        self._scene_name = scene_name
        self._max_episode_steps = 200

        self._cache_dir = 'tmp/goal_cache/'
        # TODO: specify some absolute path? depending on where python is run
        # may need to link this dir b/c just using relative path for now
        if not os.path.exists(self._cache_dir):
            os.makedirs(self._cache_dir)

        self._controller = Controller(scene=self._scene_name, gridSize=self._grid_size, width=self.width,
                                      height=self.height)
        self._objects_to_find = set(objects_to_find)
        self._goal_conditioned = goal_conditioned
        self._clear_receptacle_object = clear_receptacle_object
        self._receptacle_object = receptacle_object
        self._preopen_receptacle_object = preopen_receptacle_object
        self._last_event = None  # TODO: use controller.last_event
        self._episode_steps = 0

        if self._goal_conditioned:
            self.observation_space = Box(low=0, high=255, shape=(2 * self.width, self.height, 3),
                                         dtype=np.uint8)  # TODO check
        else:
            self.observation_space = Box(low=0, high=255, shape=(self.width, self.height, 3),
                                         dtype=np.uint8)  # TODO check
        self.action_space = Discrete(6)

    def __del__(self):
        self.close()

    def seed(self, seed=None):
        Utils.seed(seed=seed)  # TODO: untested so far

    def step(self, action):
        thor_action_data = None
        object_pickup = False
        object_put = False
        interactObject = None

        if action == 0:
            thor_action_data = {"action": DefaultActions.MoveAhead.name, "moveMagnitude": self._grid_size}
        elif action == 1:
            thor_action_data = {"action": DefaultActions.MoveBack.name, "moveMagnitude": self._grid_size}
        elif action == 2:
            thor_action_data = {"action": DefaultActions.RotateLeft.name, "rotation": 90}
        elif action == 3:
            thor_action_data = {"action": DefaultActions.RotateRight.name, "rotation": 90}
        elif action == 4:  # pickup
            if self._last_event is None and len(
                    self._last_event.metadata['inventoryObjects']) > 0:  # can't pickup if holding something already
                # Intended to be a no-op
                thor_action_data = {"action": DefaultActions.MoveAhead.name, "moveMagnitude": 0}
            else:
                # We'll find all instances of the object we're trying to pick up, and try pick up the first
                visible_objects = [o for o in self._last_event.metadata['objects'] if
                                   o['visible'] and o['objectType'] in self._objects_to_find]

                if len(visible_objects) > 0:
                    interactObject = visible_objects[0]
                    thor_action_data = {"action": "PickupObject", 'objectId': interactObject['objectId']}
                    object_pickup = True
                else:
                    # Intended to be a no-op
                    thor_action_data = {"action": DefaultActions.MoveAhead.name, "moveMagnitude": 0}
        elif action == 5:  # place in receptacle
            if self._last_event is None or len(self._last_event.metadata['inventoryObjects']) == 0:
                # Intended to be a no-op
                thor_action_data = {"action": DefaultActions.MoveAhead.name, "moveMagnitude": 0}
            else:
                interactObject = self._last_event.metadata['inventoryObjects'][
                    0]  # only has 'objectId', 'objectType' properties
                thor_action_data = {"action": "PutObject", 'objectId': self._receptacle_objectId}
                object_put = True

        self._last_event = self._controller.step(**thor_action_data)

        # We copy the last frame to re-allocate the memory, because otherwise the conversion to torch Tensor
        # is having negative stride issues: https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663
        next_observation = self._last_event.frame.copy()
        if self._last_event.metadata['lastActionSuccess']:
            if object_pickup:
                if interactObject['objectType'] not in self._pickup_history:
                    self._pickup_history.add(interactObject['objectType'])
                    reward = 1
                else:
                    reward = 0
            elif object_put:
                if interactObject['objectType'] not in self._put_history:
                    self._put_history.add(interactObject['objectType'])
                    reward = 10
                else:
                    reward = 0
            else:
                reward = 0
        else:
            reward = 0

        if self._goal_conditioned:
            next_observation = np.concatenate([next_observation, self._goal], axis=0)

        success = self.check_if_put_all()
        done = self._episode_steps > self._max_episode_steps or success

        self._episode_steps += 1

        return next_observation, reward, done, {}

    def check_if_put_all(self):
        found = {}
        receptacle_object = self._last_event.get_object(self._receptacle_objectId)
        for o in receptacle_object['receptacleObjectIds']:
            o = self._last_event.get_object(o)
            if o['objectType'] in self._objects_to_find:
                found[o['objectType']] = 1
        return sum(found.values()) == len(self._objects_to_find)

    def gen_goal_tag(self):
        return f'{self._scene_name}-{self.width}-{self.height}-{self._receptacle_object}-{self._objects_to_find}' + '.png'

    def generate_goal(self):
        goal_tag = self.gen_goal_tag()
        gp = os.path.join(self._cache_dir, goal_tag)
        if os.path.exists(gp):
            from PIL import Image
            goal = np.array(Image.open(gp))
            return goal

        receptacle_object = self._last_event.get_object(self._receptacle_objectId)

        if self._clear_receptacle_object:
            for o in receptacle_object['receptacleObjectIds']:
                self._last_event = self._controller.step('RemoveFromScene', objectId=o)
                assert self._last_event.metadata['lastActionSuccess']

        objs = []
        object_poses = []
        for object_type in self._objects_to_find:
            o = self._last_event.objects_by_type(object_type)
            assert len(o) == 1
            o = o[0]
            objs.append(o)
            p = dict(objectName=o['name'], position=o['position'], rotation=o['rotation'])
            object_poses.append(p)

        # see https://github.com/allenai/ai2thor-rearrangement/blob/86f6c61db86d0235dc22e0d17c878e3e56f96f09/rearrange_config.py#L977
        # and https://arxiv.org/pdf/2011.01975.pdf
        self._last_event = self._controller.step(action='PositionsFromWhichItemIsInteractable',
                                                 objectId=self._receptacle_objectId)
        poses = self._last_event.metadata['actionReturn']

        for o in objs:
            self._last_event = self._controller.step('PickupObject', objectId=o['objectId'], forceAction=True)
            assert self._last_event.metadata['lastActionSuccess']

            placed = False
            i = 0
            while not placed:  # https://github.com/allenai/ai2thor/issues/339
                self._last_event = self._controller.step(action='PutObject', objectId=self._receptacle_objectId,
                                                         forceAction=True)
                # print(o, i, self._last_event.metadata['errorMessage'])
                placed = self._last_event.metadata['lastActionSuccess']

                if not placed:
                    p = {k: poses[k][i] for k in ['x', 'y', 'z', 'rotation', 'horizon']}
                    self._last_event = self._controller.step('TeleportFull', **p)
                    self._last_event = self._controller.step('Stand' if poses['standing'][i] else 'Crouch')
                    i += 1

        # could also use AddThirdPartyCamera?
        in_view = False
        i = 0
        while not in_view:
            p = {k: poses[k][i] for k in ['x', 'y', 'z', 'rotation', 'horizon']}
            if p['horizon'] <= 10:  # heuristic
                self._last_event = self._controller.step('TeleportFull', **p)
                self._last_event = self._controller.step('Stand' if poses['standing'][i] else 'Crouch')
                ro = self._last_event.get_object(self._receptacle_objectId)
                in_view = ro['visible'] and not ro['obstructed']
                in_view = in_view and (1.75 <= ro['distance'] and ro['distance'] <= 2.25)  # heuristic
                for oid in ro['receptacleObjectIds']:
                    o = self._last_event.get_object(oid)
                    in_view = in_view and (o['visible'] and not o['obstructed'])

            i += 1

        goal = self._last_event.frame.copy()
        from PIL import Image
        Image.fromarray(goal).save(gp)
        # save the goal image in a cache b/c it takes awhile to generate

        self._last_event = self._controller.step(action='SetObjectPoses', objectPoses=object_poses)
        self._last_event = self._controller.step("InitialRandomSpawn")

        return goal

    def reset(self):
        self._episode_steps = 0
        self._reward = 1
        self._last_event = self._controller.reset(self._scene_name)
        self._last_event = self._controller.step("InitialRandomSpawn")

        ro = self._last_event.objects_by_type(self._receptacle_object)
        assert len(ro) == 1
        self._receptacle_objectId = ro[0]['objectId']

        self._pickup_history = set()
        self._put_history = set()

        if self._preopen_receptacle_object:
            self._last_event = self._controller.step(action='OpenObject', objectId=self._receptacle_objectId,
                                                     forceAction=True, moveMagnitude=1.0)
            assert self._last_event.metadata['lastActionSuccess']

        if self._goal_conditioned:
            self._goal = self.generate_goal()
            obs = self._last_event.frame.copy()
            return np.concatenate([obs, self._goal], axis=0)
        else:
            return self._last_event.frame.copy()

    def render(self, mode='human', close=False):
        pass

    def close(self):
        try:
            self._controller.stop()  # TODO: this gives bad file descriptor error...not sure why
        except OSError:
            print("OSError (likely bad file descriptor). Just letting it go....")
            pass
