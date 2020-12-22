from continual_rl.experiments.experiment import Experiment
from continual_rl.experiments.tasks.image_task import ImageTask
from continual_rl.experiments.tasks.minigrid_task import MiniGridTask
from continual_rl.utils.env_wrappers import wrap_deepmind, make_atari
from continual_rl.available_policies import LazyDict


def load_mini_atari_cycle():
    return Experiment(tasks=[
                ImageTask(action_space_id=0,
                          env_spec=lambda: wrap_deepmind(
                              make_atari('SpaceInvadersNoFrameskip-v4', max_episode_steps=10000),
                              clip_rewards=False,
                              frame_stack=False,  # Handled separately
                              scale=False,
                          ),
                          num_timesteps=10000000, time_batch_size=4, eval_mode=False,
                          image_size=[84, 84], grayscale=True),
                   ImageTask(action_space_id=2,
                             env_spec=lambda: wrap_deepmind(
                                 make_atari('KrullNoFrameskip-v4', max_episode_steps=10000),
                                 clip_rewards=False,
                                 frame_stack=False,  # Handled separately
                                 scale=False,
                             ), num_timesteps=10000000, time_batch_size=4, eval_mode=False,
                             image_size=[84, 84], grayscale=True),
                   ImageTask(action_space_id=4,
                             env_spec=lambda: wrap_deepmind(
                                 make_atari('BeamRiderNoFrameskip-v4', max_episode_steps=10000),
                                 clip_rewards=False,
                                 frame_stack=False,  # Handled separately
                                 scale=False,
                             ), num_timesteps=10000000, time_batch_size=4, eval_mode=False,
                             image_size=[84, 84], grayscale=True)
            ], continual_testing_freq=50000, cycle_count=5)


def load_minigrid_empty8x8_unlock():
    return Experiment(tasks=[MiniGridTask(action_space_id=0, env_spec='MiniGrid-Empty-8x8-v0', num_timesteps=150000,
                                           time_batch_size=1, eval_mode=False),
                              MiniGridTask(action_space_id=0, env_spec='MiniGrid-Unlock-v0', num_timesteps=500000,
                                           time_batch_size=1, eval_mode=False),
                              MiniGridTask(action_space_id=0, env_spec='MiniGrid-Empty-8x8-v0', num_timesteps=10000,
                                           time_batch_size=1, eval_mode=True)
                              ])


def get_available_experiments():

    experiments = LazyDict({
        "mini_atari_cycle": load_mini_atari_cycle,
        "minigrid_empty8x8_unlock": load_minigrid_empty8x8_unlock
    })

    return experiments
