import gym
import numpy as np
from ravens.environments.environment import Environment as RavensVisualForesightEnvironment
from ravens.tasks.put_block_base_mcts import PutBlockBaseMCTS
import pickle
import os
import random
import torch


class RavensSimEnvironment(gym.Env):
    def __init__(self, assets_root):
        super().__init__()
        self._env = RavensVisualForesightEnvironment(assets_root=assets_root, task=PutBlockBaseMCTS(), disp=False) #  TODO: requires installation, maybe? Hanging, currently: use_egl=True)
        self._max_steps = 50  # TODO...
        self._current_step = 0

        observation_space = self._env.observation_space
        self.observation_space = gym.spaces.Dict({"image": observation_space["color"][0]})  # TODO: include depth. [0] means "front camera" (vs "left" and "right)

        action_space_shape = 0
        action_space_low = []
        action_space_high = []

        for pose_id in ("pose0", "pose1"):
            for space in self._env.action_space[pose_id].spaces:
                action_space_shape = action_space_shape + space.shape[0]
                action_space_low = [*action_space_low, *space.low]
                action_space_high = [*action_space_high, *space.high]

        self.action_space = gym.spaces.Box(shape=np.array([action_space_shape]), low=np.array(action_space_low), high=np.array(action_space_high), dtype=np.float32)

    def _convert_observation(self, observation):
        converted_observation = {"image": observation["color"][0]}
        return converted_observation

    def reset(self):
        observation = self._env.reset()
        self._current_step = 0
        return self._convert_observation(observation)

    def step(self, action):
        converted_action = {}
        current_action_start = 0
        for pose_id in ("pose0", "pose1"):
            pose_actions = []
            for space in self._env.action_space[pose_id].spaces:
                space_size = space.shape[0]
                pose_actions.append(action[current_action_start:current_action_start+space_size])
                current_action_start += space_size

            converted_action[pose_id] = tuple(pose_actions)

        observation, reward, done, info = self._env.step(converted_action)

        done = done or self._current_step >= self._max_steps
        self._current_step += 1

        return self._convert_observation(observation), reward, done, info


class RavensDemonstrationEnv(RavensSimEnvironment):
    # TODO: inheriting from the SimEnv just to grab the observation space and action space, lazily. It's probably
    # more heavy than desired
    def __init__(self, assets_root, data_dir, valid_dataset_indices):
        super().__init__(assets_root)
        self._data_dir = data_dir

        self._action_dir = os.path.join(self._data_dir, "action")
        self._color_dir = os.path.join(self._data_dir, "color")
        self._depth_dir = os.path.join(self._data_dir, "depth")  # TODO: use this
        self._info_dir = os.path.join(self._data_dir, "info")
        self._reward_dir = os.path.join(self._data_dir, "reward")

        self._trajectory_ids = os.listdir(self._action_dir)[valid_dataset_indices[0]:valid_dataset_indices[1]]
        self._current_trajectory_id = None
        self._current_actions = None
        self._current_colors = None
        self._current_depths = None
        self._current_infos = None
        self._current_rewards = None

        self._current_timestep = 0

    def _convert_observation(self, color):
        return {"image": color[0]}  # TODO: check, but I think it should be (center, left, right)? So we're grabbing center

    def _convert_dict_to_unified_action(self, dict_action):
        unified_action = []
        for pose_id in ("pose0", "pose1"):
            for space in dict_action[pose_id]:
                unified_action.append(space)

        return torch.tensor(np.concatenate(unified_action))  # TODO: this really shouldn't convert to torch here, but it is very convenient

    def reset(self):
        self._current_trajectory_id = random.randint(0, len(self._trajectory_ids)-1)
        trajectory_filename = self._trajectory_ids[self._current_trajectory_id]
        self._current_timestep = 0

        with open(os.path.join(self._action_dir, trajectory_filename), 'rb') as file:
            self._current_actions = pickle.load(file)

        with open(os.path.join(self._color_dir, trajectory_filename), 'rb') as file:
            self._current_colors = pickle.load(file)

        with open(os.path.join(self._depth_dir, trajectory_filename), 'rb') as file:
            self._current_depths = pickle.load(file)

        with open(os.path.join(self._info_dir, trajectory_filename), 'rb') as file:
            self._current_infos = pickle.load(file)

        with open(os.path.join(self._reward_dir, trajectory_filename), 'rb') as file:
            self._current_rewards = pickle.load(file)

        return self._convert_observation(self._current_colors[self._current_timestep])

    def step(self, action):
        demo_action = self._convert_dict_to_unified_action(self._current_actions[self._current_timestep])
        observation = self._convert_observation(self._current_colors[self._current_timestep])
        info = {"demo_action": demo_action}  # Just has pose0 and pose1, which I already have from action? TODO: self._current_infos[self._current_timestep]
        reward = self._current_rewards[self._current_timestep]
        done = self._current_timestep + 1 == len(self._current_actions) or self._current_actions[self._current_timestep + 1] is None

        self._current_timestep += 1

        return observation, reward, done, info
