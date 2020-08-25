from continual_rl.experiments.experiment import Experiment
from continual_rl.experiments.tasks.image_task import ImageTask
from continual_rl.experiments.tasks.minigrid_task import MiniGridTask
from continual_rl.utils.env_wrappers import EpisodicLifeEnv
from continual_rl.utils.utils import Utils
from continual_rl.utils.vec_env_wrappers import VecNormalize
import gym


def get_available_experiments():
    experiments = {
        "breakout":
            Experiment(tasks=[
                ImageTask(action_space_id=0,
                          env_spec=lambda: EpisodicLifeEnv(Utils.make_env('BreakoutDeterministic-v4')),
                          num_timesteps=10000000, time_batch_size=4, eval_mode=False,
                          image_size=[84, 84], grayscale=True)
            ]),

        "recall_minigrid_empty8x8_unlock":
            Experiment(tasks=[MiniGridTask(action_space_id=0, env_spec='MiniGrid-Empty-8x8-v0', num_timesteps=150000,
                                           time_batch_size=1, eval_mode=False),
                              MiniGridTask(action_space_id=0, env_spec='MiniGrid-Unlock-v0', num_timesteps=500000,
                                           time_batch_size=1, eval_mode=False),
                              MiniGridTask(action_space_id=0, env_spec='MiniGrid-Empty-8x8-v0', num_timesteps=10000,
                                           time_batch_size=1, eval_mode=True)
                              ]),

        "coinrun_easy_unlimited":
            Experiment(tasks=[
                ImageTask(action_space_id=0,
                          env_spec=lambda: VecNormalize(gym.make('procgen:procgen-coinrun-v0', distribution_mode="easy")),
                          num_timesteps=10000000, time_batch_size=4,
                          eval_mode=False, image_size=[84, 84], grayscale=False)
            ])
    }

    return experiments
