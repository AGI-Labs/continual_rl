"""
From https://github.com/AGI-Labs/franka_control/blob/b369bb1bc4d83ebe10d493f253bfbace2cc585ef/franka_env.py
"""

import torch, os
import numpy as np
from gym import Env, spaces
from .util import robot_setup, Rate, LOW_JOINTS, HIGH_JOINTS, HOMES
from .camera import Camera
from collections import OrderedDict


class FrankaEnv(Env):
    def __init__(self, home, hz, gain_type, camera=True):
        self.observation_space = spaces.Box(low=0, high=255, shape=(120, 160, 3), dtype=np.uint8)
        #spaces.Box(
            #low=LOW_JOINTS, high=HIGH_JOINTS, dtype=np.float32
        #)
        self.action_space = spaces.Box(
            low=LOW_JOINTS, high=HIGH_JOINTS, dtype=np.float32
        )
        self.curr_step = 0

        self.home = home
        self.rate = Rate(hz)
        self.gain_type = gain_type
        self.camera = Camera() if camera else None
        self.reset()

    def step(self, ac):
        if ac is not None:
            self.robot.update_current_policy({"q_desired": torch.from_numpy(ac)})
        self.rate.sleep()
        return self._get_obs(), 0, False, {}

    def _get_obs(self):
        obs = OrderedDict()
        obs["q"] = self.robot.get_joint_positions().numpy()
        obs["qdot"] = self.robot.get_joint_velocities().numpy()
        obs["eep"] = np.concatenate([a.numpy() for a in self.robot.get_ee_pose()])
        if self.camera:
            c, d = self.camera.get_frame()
            obs["rgb"] = c
            obs["depth"] = d
        return obs["rgb"]  # TODO: return everything. This is all I need right now so being lazy

    def reset(self):
        self.robot, self.policy = robot_setup(self.home, self.gain_type)
        return self._get_obs()

    def close(self):
        return self.robot.terminate_current_policy()


class FrankaScoopEnv(FrankaEnv):
    def __init__(self):
        super().__init__(home=HOMES["scoop"], hz=30, gain_type="default", camera=True)
