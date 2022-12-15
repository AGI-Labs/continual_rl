import gym
import numpy as np
from gym.spaces import Box, Discrete


class LiquidHandler(gym.Env):
    def __init__(self, grid_size=None, num_blocks=None, penalize_dist=True):
        if grid_size is None:
            grid_size = [10, 10]

        if num_blocks is None:
            num_blocks = [0, 1]

        self._num_blocks = num_blocks
        self._penalize_dist = penalize_dist
        self._max_steps = 100

        self._grid = np.zeros((grid_size[0], grid_size[1], len(self._num_blocks)))
        self._goals = np.zeros(self._grid.shape)
        self._blocks_in_grasp = np.zeros(len(self._num_blocks,))
        self._current_step = 0
        self._current_arm_pos = np.array([0, 0])  # TODO?

        shape = [self._grid.shape[2] + self._goals.shape[2] + self._blocks_in_grasp.shape[0], self._grid.shape[0], self._grid.shape[1]]
        self.observation_space = Box(low=0, high=1.0, shape=shape, dtype=np.int32)
        self.action_space = Discrete(n=np.prod(self._grid.shape[:2]))

    def _populate_grid(self, grid, blocks_to_fill):
        for block_id, num_blocks in enumerate(blocks_to_fill):
            for filled_id in range(num_blocks):
                filled_pos = np.random.randint(0, grid.shape[:2])
                grid[filled_pos[0]][filled_pos[1]][block_id] += 1
                #print(f"Block {block_id} filled pos: {filled_pos}")

    def _generate_observation(self):
        #return np.concatenate((self._grid.flatten(), self._goals.flatten()))
        tiled_blocks_in_grasp = np.tile(self._blocks_in_grasp, (self._grid.shape[0], self._grid.shape[1], 1))
        obs = np.concatenate((self._grid, self._goals, tiled_blocks_in_grasp), axis=-1)
        obs = obs.transpose((2, 0, 1))
        return obs

    def reset(
        self,
        *,
        seed = None,
        options = None,
    ):
        self._grid = np.zeros(self._grid.shape)
        self._goals = np.zeros(self._goals.shape)
        self._blocks_in_grasp = np.zeros(self._blocks_in_grasp.shape)
        self._current_step = 0
        self._current_arm_pos = np.array([0, 0])

        #print(f"Populating grid:")
        self._populate_grid(self._grid, self._num_blocks)

        #print(f"Populating goals:")
        self._populate_grid(self._goals, self._num_blocks)

        return self._generate_observation()

    def step(self, action):
        action_x = action // self._grid.shape[0]
        action_y = action % self._grid.shape[0]
        new_arm_pos = np.array([action_x, action_y])

        dist_traveled = np.sqrt(((new_arm_pos - self._current_arm_pos)**2).sum())
        self._current_arm_pos = new_arm_pos

        # If there are no blocks in the grasp, it'll perform a pick action
        if self._blocks_in_grasp.sum() == 0:
            # Penalize if we're picking up a goal that was already completed
            completed_goals = np.minimum(self._goals[action_x][action_y], self._grid[action_x][action_y])
            unnecessary_blocks = self._grid[action_x][action_y] - completed_goals

            # Reward for any non-goal blocks picked up (for net-zero consistency), and penalize for any goal blocks picked
            reward = unnecessary_blocks.sum() - completed_goals.sum()

            # Pick blocks
            #print(f"Picking from [{action_x}][{action_y}]")
            self._blocks_in_grasp = self._grid[action_x][action_y].copy()
            self._grid[action_x][action_y] *= 0
            #print(f"Blocks held: {self._blocks_in_grasp}")

            done = False
        else:
            # Otherwise, we're in a place state
            #print(f"Placing {self._blocks_in_grasp} into [{action_x}][{action_y}]")

            # How many blocks of each type we have left to get
            goal_blocks_left = np.clip(self._goals[action_x][action_y] - self._grid[action_x][action_y], a_min=0, a_max=None)

            # If we're placing more blocks than desired, only count the number desired
            # If we're placing fewer, only count the blocks placed
            goal_blocks_placed = np.minimum(self._blocks_in_grasp, goal_blocks_left)
            unnecessary_blocks = self._blocks_in_grasp - goal_blocks_placed
            #print(f"Reward from plack: {reward}")

            # Reward for the goal blocks, but penalize for placing extras unnecessarily
            reward = goal_blocks_placed.sum() - unnecessary_blocks.sum()

            # Place the blocks
            self._grid[action_x][action_y] += self._blocks_in_grasp.copy()
            self._blocks_in_grasp *= 0

            done = np.all(self._goals == self._grid)

        obs = self._generate_observation()
        self._current_step += 1
        done = done or self._current_step > self._max_steps

        if self._penalize_dist:
            max_dist = np.sqrt((np.array(self._grid.shape) ** 2).sum())
            reward -= 0.1 * dist_traveled / max_dist

        return obs, reward, done, {}


if __name__ == "__main__":
    env = LiquidHandler([2, 2])
    obs = env.reset()

    done = False

    while not done:
        obs, reward, done, info = env.step(env.action_space.sample())
