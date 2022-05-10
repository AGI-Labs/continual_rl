import gym


class RobotDemonstrationEnv(gym.Env):
    """
    This class is a wrapper around a dataset of
    """
    def __init__(self):
        super().__init__()

    def step(self, action):
        pass

    def reset(
        self,
        seed = None,
        return_info = False,
        options = None,
    ):
        super().reset(seed)  # Handles basic seeding of numpy. TODO: use self._np_random
        pass
