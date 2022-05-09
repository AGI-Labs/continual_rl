from continual_rl.experiments.tasks.image_task import ImageTask
from continual_rl.utils.utils import Utils
import gym
import numpy as np


class MultiGoalToImageWrapper(gym.ObservationWrapper):
    """
    Converts the robot observation into concatenated (observation, goal) image, in the range [0, 255], uint8
    """
    def __init__(self, env):
        super().__init__(env)
        original_space = env.observation_space.spaces["observation"]
        new_shape = (original_space.shape[0], original_space.shape[1], 2 * original_space.shape[2])  # (observation, goal) concatenated in the last dim
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=new_shape, dtype=np.uint8)

    def observation(self, observation):
        current_obs = observation["observation"]
        goal = observation["desired_goal_img"]  # TODO: what is "achieved goal image"?
        return np.concatenate((current_obs, goal), axis=-1)


class MultiGoalRobotTask(ImageTask):
    def __init__(self, task_id, action_space_id, env_spec, num_timesteps, time_batch_size, eval_mode,
                 image_size, grayscale, continual_eval=True, resize_interp_method="INTER_AREA"):
        env_spec_multi = lambda: MultiGoalToImageWrapper(Utils.make_env(env_spec)[0])
        super().__init__(task_id, action_space_id, env_spec_multi, num_timesteps, time_batch_size, eval_mode,
                 image_size, grayscale, continual_eval, resize_interp_method)
