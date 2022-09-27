import gym
import numpy as np
from ravens.environments.environment_mcts import EnvironmentMCTS as RavensVisualForesightEnvironment
from ravens.tasks.put_block_base_mcts import PutBlockBaseMCTS
import pickle
import os
import random
import torch
#from ravens_torch.dataset import Dataset
from cliport.dataset import RavensDataset as Dataset
from ravens.tasks import names
import random


class RavensSimEnvironment(gym.Env):
    def __init__(self, assets_root, task_name, data_dir, use_goal_image=False, seeds=None, n_demos=1000):
        super().__init__()

        # For goal generation
        cfg = RavensSimEnvironment.construct_cfg()
        self._dataset = Dataset(data_dir, cfg, n_demos=n_demos)  # TODO: config
        self.use_goal_image = use_goal_image
        self._seeds = seeds

        task_class = names[task_name]
        self._env = RavensVisualForesightEnvironment(assets_root=assets_root, task=task_class(pp=True), disp=False, hz=480) #  TODO: requires installation, maybe? Hanging, currently: use_egl=True)
        #self._env.reset()

        self._max_steps = 50  # TODO...
        self._current_step = 0

        observation_space = self._env.observation_space
        color_spaces = observation_space["color"]
        depth_spaces = observation_space["depth"]
        aggregated_dim = 0
        lows = []
        highs = []

        num_image_sets = 2 if use_goal_image else 1

        for _ in range(num_image_sets):
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

        combined_shape = [*color_spaces[0].shape[:-1], aggregated_dim]
        #combined_color_depth_space = gym.spaces.Box(low=combined_low, high=combined_high, shape=combined_shape, dtype=np.uint8)  # TODO: getting converted to np.float
        combined_color_depth_space = gym.spaces.Box(low=0, high=255, shape=(320, 160, 12), dtype=np.uint8)  # TODO: getting converted to np.float

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

        #self.reset()

    @staticmethod
    def construct_cfg():
        # TODO: omegaconf + config files. Some of these certainly don't belong here
        cfg = {}
        cfg["dataset"] = {"type": "single", "images": True, "cache": True, "augment": {"theta_sigma": 60}}
        cfg["train"] = {"exp_folder": "exps", "task": "packing-boxes-pairs-seen-colors",
                        "agent": "two_stream_clip_unet_lat_transporter", "n_demos": 1000, "n_rotations": 36,
                        "attn_stream_fusion_type": "add", "trans_stream_fusion_type": "conv",  "lang_fusion_type": 'mult',
                        "val_repeats": 1,
                        "save_steps": [1000, 2000, 3000, 4000, 5000, 7000, 10000, 20000, 40000, 80000, 120000, 160000, 200000, 300000, 400000, 500000, 600000, 800000, 1000000, 1200000],
                        "batchnorm": False,
                        "log": False,
                        "lr": 1e-4}
        return cfg

    def _convert_observation(self, observation):
        """all_camera_data = []

        for camera_id in range(len(observation["color"])):
            camera_data = np.concatenate((observation["color"][camera_id], np.expand_dims(observation["depth"][camera_id], -1)), axis=-1)
            all_camera_data.append(camera_data)

        converted_observation = {"image": np.concatenate(all_camera_data, axis=-1)}"""

        converted_observation = {"image": self._dataset.get_image(observation)}

        return converted_observation

    def _append_goal(self, observation):
        if self.use_goal_image:
            # To grab a goal observation
            _, goal_dict = self._dataset[random.randint(0, len(self._dataset))]

            #goal_observation = self._convert_observation(goal_dict['img'])  # TODO: this is after height map. Make consistent
            observation = {'image': np.concatenate((observation['image'], goal_dict['img']), axis=-1)}

        return observation

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

    def _convert_dict_to_unified_action(self, dict_action):
        unified_action = []
        for pose_id in ("pose0", "pose1"):
            for space in dict_action[pose_id]:
                unified_action.append(space)

        return torch.tensor(np.concatenate(unified_action))  # TODO: this really shouldn't convert to torch here, but it is very convenient

    def reset(self):
        if self._seeds is None:
            seed = None
        else:
            # Select a "truer" random seed from the set of allowed seeds
            random.seed(int.from_bytes(os.urandom(4), byteorder="little"))
            seed = random.choice(self._seeds)

        self._env.seed(seed)  # TODO: not actually done during demo collection...
        np.random.seed(seed)
        random.seed(seed)  # TODO: not actually done during demo collection...

        observation, base_urdf, base_size, base_id, objs_id, info = self._env.reset()
        self._current_step = 0
        observation = self._convert_observation(observation)
        observation = self._append_goal(observation)
        return observation

    def step(self, action):
        converted_action = self.convert_unified_action_to_dict(action)
        observation, reward, done, info = self._env.step(converted_action)

        pick_obs, place_obs = observation  # TODO: which one

        observation = self._convert_observation(pick_obs)
        observation = self._append_goal(observation)

        done = done or self._current_step >= self._max_steps
        self._current_step += 1

        return observation, reward, done, info


class RavensDemonstrationEnv(RavensSimEnvironment):
    # TODO: inheriting from the SimEnv just to grab the observation space and action space, lazily. It's probably
    # more heavy than desired
    def __init__(self, task_name, assets_root, data_dir, valid_dataset_indices, use_goal_image, n_demos=1000):
        super().__init__(assets_root=assets_root, task_name=task_name, data_dir=data_dir, use_goal_image=use_goal_image,
                         n_demos=n_demos)
        self._data_dir = data_dir
        #self._dataset = Dataset(data_dir)
        self._max_steps = 10  # Episodes don't have a done in demonstration-mode. TODO?
        self._current_step = 0
        self.use_goal_image = use_goal_image

        self._action_dir = os.path.join(self._data_dir, "action")
        self._color_dir = os.path.join(self._data_dir, "color")
        self._depth_dir = os.path.join(self._data_dir, "depth")
        self._info_dir = os.path.join(self._data_dir, "info")
        self._reward_dir = os.path.join(self._data_dir, "reward")

        self._trajectory_ids = os.listdir(self._action_dir)
        self._trajectory_ids.sort()  # TODO: actually check the seeds so we can be confident in alignment between Sim and Demo
        self._trajectory_ids = self._trajectory_ids[valid_dataset_indices[0]:valid_dataset_indices[1]]

        self._current_trajectory_id = None
        self._current_actions = None
        self._current_colors = None
        self._current_depths = None
        self._current_infos = None
        self._current_rewards = None

        self._current_timestep = 0

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
        observation = self._convert_observation(observation)
        observation = self._append_goal(observation)

        return observation

    def step(self, action):
        """(obs, act, reward, _), (goal_obs, _, _, _)  = self._dataset.sample()
        demo_action = self._convert_dict_to_unified_action(act)
        info = {"demo_action": demo_action}

        done = self._current_step >= self._max_steps  # Never hits this: np.all(obs["color"] == goal_obs["color"]) and np.all(obs["depth"] == goal_obs["depth"])  # TODO: this isn't really necessary...just doing it for video rendering purposes mostly
        self._current_step += 1

        obs = self._convert_observation(obs)"""

        raw_obs = {"color": self._current_colors[self._current_timestep + 1], "depth": self._current_depths[self._current_timestep + 1]}
        observation = self._convert_observation(raw_obs)
        observation = self._append_goal(observation)

        # TODO: Sim env is currently sampling a different one every time step. TODO: consistent with that? or consistent with GoalTransporterAgent comments
        #if self.use_goal_image:
        #    # To grab a goal observation
        #    #_, (goal_obs, _, _, _)  = self._dataset.sample()
        #    goal_obs = {"color": self._current_colors[-1], "depth": self._current_depths[-1]}
        #    goal_observation = self._convert_observation(goal_obs)
        #    observation = {'image': np.concatenate((observation['image'], goal_observation['image']), axis=-1)}

        demo_action = self._current_actions[self._current_timestep]
        demo_action = self._convert_dict_to_unified_action(demo_action)

        info = {"demo_action": demo_action}  # Just has pose0 and pose1, which I already have from action? TODO: self._current_infos[self._current_timestep]
        reward = self._current_rewards[self._current_timestep]
        done = self._current_timestep + 1 == len(self._current_actions) or self._current_actions[self._current_timestep + 1] is None

        self._current_timestep += 1

        return observation, reward, done, info
