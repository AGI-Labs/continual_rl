import gym
import pickle
import os
from PIL import Image
import numpy as np
import torch
from gym.utils import seeding
from continual_rl.envs.franka.util import LOW_JOINTS, HIGH_JOINTS


class RobotDemonstrationEnv(gym.Env):
    """
    This class is a wrapper around a dataset of (observation, action, reward, done) trajectories.
    The main difference between this environment and a "normal" environment is that the action taken is *not*
    the action passed in. The action actually taken is passed out in the info dictionary, and consumers should
    use that information appropriately.
    The format is the "Cloud Dataset" (Victoria Dean) format, for now at least.
    """

    def __init__(self, dataset_path, valid_dataset_indices):
        """
        The dataset_path should have a "parsed.pkl" file with most of the trajectory data, and a "data" folder with the
        image data.
        valid_dataset_indices is intended to allow the user to specify train/test splits. E.g. train might be (None, -100)
        and test might be (-100, None).
        """
        # TODO: I need to give some thought to how I want to run "eval" in this case. Maybe a "how close is it to the real action" metric?
        # TODO: remove the valid_dataset_indices if I don't end up using them, just playing it safe right now...
        super().__init__()
        parsed_pkl_path = os.path.join(dataset_path, "parsed.pkl")
        self._dataset_path = dataset_path
        self._dataset_trajectories = pickle.load(open(parsed_pkl_path, 'rb'))[valid_dataset_indices[0] : valid_dataset_indices[1]]
        print(f"Created demonstration env with {len(self._dataset_trajectories)} entries")

        self._current_trajectory = None
        self._current_trajectory_observations = None
        self._current_trajectory_step = None

        self.observation_space = gym.spaces.Dict({'state_vector': gym.spaces.Box(low=LOW_JOINTS, high=HIGH_JOINTS, shape=(7,), dtype=np.float32),
                                                  'image': gym.spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8) })

        # TODO: support (-inf, inf)? - really a monobeast nets problem
        self.action_space = gym.spaces.Box(low=LOW_JOINTS, high=HIGH_JOINTS, shape=(7,), dtype=np.float32)
        self._np_random = None  # Should be defined in gym.Env, but not in all versions it would seem (TODO)

    def _load_next_trajectory(self):
        trajectory_id = self._np_random.integers(0, len(self._dataset_trajectories))
        self._current_trajectory = self._dataset_trajectories[trajectory_id]
        self._current_trajectory_step = 0

        images = []
        for i in range(self._current_trajectory['observations'].shape[0]):
            path = os.path.join(self._dataset_path, "data", self._current_trajectory['traj_id'], self._current_trajectory['cam0c'][i])
            img = Image.open(path)  # Assuming RGB for now
            images.append(np.asarray(img))
            img.close()
        self._current_trajectory_observations = images

    def _get_current_obs(self):
        obs = {"image": self._current_trajectory_observations[self._current_trajectory_step],
               "state_vector": self._current_trajectory['jointstates'][self._current_trajectory_step]}  # TODO: 'jointstates' or 'observations'?
        return obs

    def step(self, action):
        """
        Note: action is ignored, and the action that was actually taken is returned in info.
        """
        if self._current_trajectory is None:
            self._load_next_trajectory()

        # The action is the *current step* action, but the reward, observation, and done are the *next* step (the
        # result of taking the action)
        action = self._current_trajectory["actions"][self._current_trajectory_step]
        action_delta = action - self._current_trajectory["jointstates"][self._current_trajectory_step]

        self._current_trajectory_step += 1
        reward = self._current_trajectory["rewards"][self._current_trajectory_step]
        observation = self._get_current_obs()
        done = self._current_trajectory["terminated"][self._current_trajectory_step]

        if done:
            self._current_trajectory = None

        return observation, reward, done, {"demo_action": torch.tensor(action_delta)}

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

        self._load_next_trajectory()
        observation = self._get_current_obs()
        return observation

    def render(self, mode="human"):
        pass
