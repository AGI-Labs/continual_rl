import gym
import numpy as np
from ravens.environments.environment import Environment as RavensVisualForesightEnvironment
from ravens.tasks.put_block_base_mcts import PutBlockBaseMCTS


class RavensSimEnvironment(gym.Env):
    def __init__(self, assets_root):
        super().__init__()
        self._env = RavensVisualForesightEnvironment(assets_root=assets_root, task=PutBlockBaseMCTS(), disp=False)

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
        return self._convert_observation(observation), reward, done, info


class RavensDemonstrationEnv(gym.Env):
    def __init__(self):
        super().__init__()
