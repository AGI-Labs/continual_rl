from typing import Optional, Union, List

import gym
import os
import numpy as np
import torch
import json
import h5py
from gym.core import RenderFrame


class ManiskillEnv(gym.Env):
    def __init__(self, task_name):
        super().__init__()

        import mani_skill2.envs
        env_kwargs = dict(obs_mode='rgbd', control_mode="pd_joint_pos")
        self._env = gym.make(task_name, **env_kwargs)

        observation_space = self._env.observation_space
        self.observation_space = gym.spaces.Dict({"state_vector": observation_space["agent"]["qpos"],
                                                  "image": observation_space["image"]["base_camera"]["rgb"]})
        self.action_space = self._env.action_space

    def _convert_observation(self, observation):
        # De-dupe with DemoEnv
        new_observation = {}
        new_observation["state_vector"] = observation["agent"]["qpos"]
        new_observation["image"] = observation["image"]["base_camera"]["rgb"]
        return new_observation

    def step(self, action):
        observation, reward, done, _ = self._env.step(action)
        observation = self._convert_observation(observation)
        return observation, reward, done, {}

    def reset(self):
        observation = self._env.reset()
        return self._convert_observation(observation)

    def render(self, mode="human"):
        pass


class ManiskillDemonstrationEnv(gym.Env):
    """
    This class is a wrapper around a dataset of (observation, action, reward, done) trajectories.
    The main difference between this environment and a "normal" environment is that the action taken is *not*
    the action passed in. The action actually taken is passed out in the info dictionary, and consumers should
    use that information appropriately.
    The format is the ManiSkill format.
    """

    def __init__(self, dataset_path, task_name, valid_dataset_indices):
        """
        The dataset_path should have a "parsed.pkl" file with most of the trajectory data, and a "data" folder with the
        image data.
        valid_dataset_indices is intended to allow the user to specify train/test splits. E.g. train might be (None, -100)
        and test might be (-100, None).
        """
        # TODO: I need to give some thought to how I want to run "eval" in this case. Maybe a "how close is it to the real action" metric?
        # TODO: remove the valid_dataset_indices if I don't end up using them, just playing it safe right now...
        super().__init__()
        dataset_path = os.path.join(dataset_path, task_name)
        self._dataset_path = dataset_path

        with open(os.path.join(self._dataset_path, "trajectory.json"), 'r') as dataset_file:
            self._dataset_metadata = json.load(dataset_file)

        env_info = self._dataset_metadata["env_info"]
        print(f"Env info: {env_info}")

        env_kwargs = env_info["env_kwargs"]
        env_kwargs["obs_mode"] = "rgbd"

        import mani_skill2.envs
        self._env = gym.make(env_info["env_id"], **env_info["env_kwargs"])
        self._episodes_metadata = self._dataset_metadata["episodes"][valid_dataset_indices[0]:valid_dataset_indices[1]]
        self._dataset_trajectories = h5py.File(os.path.join(self._dataset_path, "trajectory.h5"))

        self._current_episode_metadata = None
        self._current_trajectory_actions = None
        self._current_trajectory_step = None

        observation_space = self._env.observation_space
        #self.observation_space["state_vector"] = self.observation_space["agent"]["qpos"]  # TODO: which states?
        #self.observation_space["image"] = self.observation_space["image"]["base_camera"]["rgb"]
        #del self.observation_space["agent"]
        self.observation_space = gym.spaces.Dict({"state_vector": observation_space["agent"]["qpos"],
                                                  "image": observation_space["image"]["base_camera"]["rgb"]})
        self.action_space = self._env.action_space
        self._np_random = None  # Should be defined in gym.Env, but not in all versions it would seem (TODO)

        print(f"!!! Observation space: {self.observation_space.keys()}")

    def _convert_observation(self, observation):
        # Image already maps to image.
        new_observation = {}
        new_observation["state_vector"] = observation["agent"]["qpos"]
        new_observation["image"] = observation["image"]["base_camera"]["rgb"]
        return new_observation

    def _load_next_trajectory(self):
        episode_index = self._np_random.integers(0, len(self._episodes_metadata))

        self._current_episode_metadata = self._episodes_metadata[episode_index]
        episode_id = self._current_episode_metadata["episode_id"]
        self._current_trajectory = self._dataset_trajectories.get(f'traj_{episode_id}')
        self._current_trajectory_step = 0 #self._np_random.integers(0, len(self._current_trajectory['traj_id'])-1)  # TODO: What end
        self._current_trajectory_actions = torch.tensor(self._current_trajectory.get("actions"))

        observation = self._env.reset(**self._current_episode_metadata["reset_kwargs"])
        return self._convert_observation(observation)

    def step(self, action):
        """
        Note: action is ignored, and the action that was actually taken is returned in info.
        """
        if self._current_trajectory is None:
            self._load_next_trajectory()

        action = self._current_trajectory_actions[self._current_trajectory_step].detach().cpu().numpy()
        observation, reward, done, _ = self._env.step(action)
        observation = self._convert_observation(observation)
        self._current_trajectory_step += 1

        return observation, reward, done, {"demo_action": torch.tensor(action)}

    def reset(
        self,
        seed = None,
        return_info = False,
        options = None,
    ):
        # Per the reset API, the seed should only be reset if it hasn't yet been set
        if self._np_random is None:
            self._np_random = np.random.default_rng(seed)  # See: https://github.com/hyperopt/hyperopt/issues/838
            #self._np_random, seed = seeding.np_random(seed)
            #super().reset(seed=seed)  # Handles basic seeding of numpy. TODO: use self._np_random

        observation = self._load_next_trajectory()
        return observation

    def render(self, mode="human"):
        pass


if __name__ == "__main__":
    ManiskillDemonstrationEnv("/Users/spowers/Demonstrations/maniskill/rigid_body_envs", "AssemblingKits-v0",
                              valid_dataset_indices=(None, -100))
