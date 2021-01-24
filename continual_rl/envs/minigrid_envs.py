import numpy as np
from gym_minigrid.envs.dynamicobstacles import DynamicObstaclesEnv
from gym_minigrid.envs import MiniGridEnv, WorldObj, Grid, Goal, COLORS
from gym_minigrid.rendering import fill_coords, point_in_rect


class DynamicObstaclesRandomEnv8x8(DynamicObstaclesEnv):
    def __init__(self):
        """
        See: https://github.com/maximecb/gym-minigrid/blob/master/gym_minigrid/envs/dynamicobstacles.py
        """
        super().__init__(size=8, n_obstacles=4, agent_start_pos=None)


class FakeGoal(WorldObj):
    def __init__(self, color='green'):
        super().__init__('lava', color)

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class SimpleChoiceEnv(MiniGridEnv):
    """
    Based on the EmptyEnv, only instead of one fixed-place goal, the green block is now negative, and the true
    goal is the red goal next to it.
    """

    def __init__(
        self,
        size=8,
        agent_start_pos=(1,1),
        agent_start_dir=0,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        super().__init__(
            grid_size=size,
            max_steps=4*size*size,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a green lava (looks like a goal, but it's a trap)
        lava = FakeGoal()
        self.put_obj(lava, width - 2, height - 2)

        # Place a goal square in the bottom-right corner. The goal is red, because we're being tricky
        goal = Goal()
        goal.color = 'red'
        self.put_obj(goal, width - 4, height - 2)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the red goal square"


class OddManOutEnv(MiniGridEnv):
    """
    In this env there are 3 possible goal locations. The one that is the odd color out is the one that, when selected,
    gives reward. The other 2 end the episode.
    """
    def __init__(
        self,
        correct_color, incorrect_color,
        size=8,
        agent_start_pos=(1,1),
        agent_start_dir=0,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self._num_choices = 3
        self._correct_color = correct_color
        self._incorrect_color = incorrect_color

        super().__init__(
            grid_size=size,
            max_steps=4*size*size,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_box_order(self):
        # n-1 incorrect options (represented with False), 1 correct option (represented with True)
        box_order = [False for _ in range(self._num_choices-1)]
        box_order.append(True)

        # Shuffles in-place
        random_state = np.random.RandomState()
        random_state.shuffle(box_order)
        return box_order

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Generate the random boxes and place them
        boxes = self._gen_box_order()

        for box_id, box_correct in enumerate(boxes):
            if box_correct:
                box = Goal()
                box.color = self._correct_color
            else:
                # Place a green lava (looks like a goal, but it's a trap)
                box = FakeGoal(color=self._incorrect_color)

            self.put_obj(box, width - 2 * (box_id + 1), height - 2)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = f"get to the {self._correct_color} goal square"
