"""
From https://github.com/AGI-Labs/franka_control/blob/b369bb1bc4d83ebe10d493f253bfbace2cc585ef/util.py
"""

import torch, time
import numpy as np


TIME = 10
HZ = 30
HOMES = {
    "pour": [0.1828, -0.4909, -0.0093, -2.4412, 0.2554, 3.3310, 0.5905],
    "scoop": [0.1828, -0.4909, -0.0093, -2.4412, 0.2554, 3.3310, 0.5905],
    "zip": [-0.1337, 0.3634, -0.1395, -2.3153, 0.1478, 2.7733, -1.1784],
    "insertion": [0.1828, -0.4909, -0.0093, -2.4412, 0.2554, 3.3310, 0.5905],
}
KQ_GAINS = {
    "record": [1, 1, 1, 1, 1, 1, 1],
    "default": [26.6667, 40.0000, 33.3333, 33.3333, 23.3333, 16.6667, 6.6667],
    "stiff": [240.0, 360.0, 300.0, 300.0, 210.0, 150.0, 60.0],
}
KQD_GAINS = {
    "record": [1, 1, 1, 1, 1, 1, 1],
    "default": [3.3333, 3.3333, 3.3333, 3.3333, 1.6667, 1.6667, 1.6667],
    "stiff": [30.0, 30.0, 30.0, 30.0, 15.0, 15.0, 15.0],
}
LOW_JOINTS = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
HIGH_JOINTS = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])


class Rate:
    """
    Maintains constant control rate for POMDP loop
    """

    def __init__(self, frequency):
        self._period = 1.0 / frequency
        self._last = time.time()

    def sleep(self):
        current_delta = time.time() - self._last
        sleep_time = max(0, self._period - current_delta)
        if sleep_time:
            time.sleep(sleep_time)
        self._last = time.time()


def robot_setup(home_pos, gain_type, franka_ip="172.16.0.1"):
    # Initialize robot interface and reset
    from polymetis import RobotInterface
    from continual_rl.envs.franka.pd_control import PDControl

    # Current server version is '839_gad68b678'
    # This is not quite available on conda, but 839_g0ea34d5f is
    robot = RobotInterface(ip_address=franka_ip, enforce_version=False)
    robot.set_home_pose(torch.Tensor(home_pos))
    robot.go_home()

    # Create and send PD Controller to Franka
    q_initial = robot.get_joint_positions()
    kq = torch.Tensor(KQ_GAINS[gain_type])
    kqd = torch.Tensor(KQD_GAINS[gain_type])
    pd_control = PDControl(joint_pos_current=q_initial, kq=kq, kqd=kqd)
    robot.send_torch_policy(pd_control, blocking=False)
    return robot, pd_control
