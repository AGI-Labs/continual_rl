import gym
import numpy as np
from ravens.environments.environment import Environment as RavensVisualForesightEnvironment
from ravens.tasks.put_block_base_mcts import PutBlockBaseMCTS
import pickle
import os
import random
import torch
from ravens_torch.dataset import Dataset


class RavensSimEnvironment(gym.Env):
    def __init__(self, assets_root):
        super().__init__()
        self._env = RavensVisualForesightEnvironment(assets_root=assets_root, task=PutBlockBaseMCTS(), disp=False) #  TODO: requires installation, maybe? Hanging, currently: use_egl=True)
        self._max_steps = 50  # TODO...
        self._current_step = 0

        observation_space = self._env.observation_space
        color_spaces = observation_space["color"]
        depth_spaces = observation_space["depth"]
        aggregated_dim = 0
        lows = []
        highs = []

        for camera_id, color_space in enumerate(color_spaces):
            depth_space = depth_spaces[camera_id]
            aggregated_dim += color_space.shape[-1]
            aggregated_dim += 1  # Depth

            lows.append(color_space.low)
            lows.append(np.expand_dims(depth_space.low, -1))

            highs.append(color_space.high)
            highs.append(np.expand_dims(depth_space.high, -1))

        combined_low = np.concatenate(lows, axis=-1)
        combined_high = np.concatenate(highs, axis=-1)

        combined_shape = [*color_spaces[0].shape[:-1], aggregated_dim]  # TODO: assumes consistent dims
        combined_color_depth_space = gym.spaces.Box(low=combined_low, high=combined_high, shape=combined_shape, dtype=np.uint8)

        self.observation_space = gym.spaces.Dict({"image": combined_color_depth_space})

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
        all_camera_data = []

        for camera_id in range(len(observation["color"])):
            camera_data = np.concatenate((observation["color"][camera_id], np.expand_dims(observation["depth"][camera_id], -1)), axis=-1)
            all_camera_data.append(camera_data)

        converted_observation = {"image": np.concatenate(all_camera_data, axis=-1)}
        return converted_observation

    def reset(self):
        observation = self._env.reset()
        self._current_step = 0
        return self._convert_observation(observation)

    def step(self, action):
        converted_action = {}
        current_action_start = 0
        for pose_id in ("pose0", "pose1"):  # TODO: de-dupe with converter in demonstration env
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
        self._dataset = Dataset(data_dir)
        self._max_steps = 10  # Episodes don't have a done in demonstration-mode. TODO?
        self._current_step = 0

        self._action_dir = os.path.join(self._data_dir, "action")
        self._color_dir = os.path.join(self._data_dir, "color")
        self._depth_dir = os.path.join(self._data_dir, "depth")
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

    def _convert_dict_to_unified_action(self, dict_action):
        unified_action = []
        for pose_id in ("pose0", "pose1"):
            for space in dict_action[pose_id]:
                unified_action.append(space)

        return torch.tensor(np.concatenate(unified_action))  # TODO: this really shouldn't convert to torch here, but it is very convenient

    @staticmethod
    def convert_unified_action_to_dict(unified_action):
        action_dict = {}
        action_index = 0

        for pose_id in ("pose0", "pose1"):
            action_dict[pose_id] = []
            for space_dim in (3, 4):  # xyz, xyzw - TODO: not hard-coded
                action_dict[pose_id].append(unified_action[action_index : action_index + space_dim])
                action_index += space_dim

        return action_dict

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

        observation = {"color": self._current_colors[self._current_timestep],
                       "depth": self._current_depths[self._current_timestep]}

        return self._convert_observation(observation)

    def step(self, action):
        (obs, act, reward, _), (goal_obs, _, _, _)  = self._dataset.sample()
        demo_action = self._convert_dict_to_unified_action(act)
        info = {"demo_action": demo_action}

        done = self._current_step >= self._max_steps  # Never hits this: np.all(obs["color"] == goal_obs["color"]) and np.all(obs["depth"] == goal_obs["depth"])  # TODO: this isn't really necessary...just doing it for video rendering purposes mostly
        self._current_step += 1

        obs = self._convert_observation(obs)

        """observation = self._convert_observation(self._current_colors[self._current_timestep])
        info = {"demo_action": demo_action}  # Just has pose0 and pose1, which I already have from action? TODO: self._current_infos[self._current_timestep]
        reward = self._current_rewards[self._current_timestep]
        done = self._current_timestep + 1 == len(self._current_actions) or self._current_actions[self._current_timestep + 1] is None

        self._current_timestep += 1"""

        return obs, reward, done, info
