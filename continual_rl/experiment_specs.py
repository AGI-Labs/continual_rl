from continual_rl.experiments.experiment import Experiment
from continual_rl.experiments.tasks.image_task import ImageTask
from continual_rl.experiments.tasks.minigrid_task import MiniGridTask


def get_available_experiments(output_dir):
    experiments = {
        "breakout":
            Experiment(tasks=[
                ImageTask(env_spec='BreakoutDeterministic-v4', num_timesteps=10000000, time_batch_size=4,
                          eval_mode=False, output_dir=output_dir, image_size=[84, 84], grayscale=True)
            ], output_dir=output_dir),

        "recall_minigrid_empty8x8_unlock":
            Experiment(tasks=[MiniGridTask(env_spec='MiniGrid-Empty-8x8-v0', num_timesteps=150000, time_batch_size=1,
                                           eval_mode=False, output_dir=output_dir),
                              MiniGridTask(env_spec='MiniGrid-Unlock-v0', num_timesteps=5000000, time_batch_size=1,
                                           eval_mode=False, output_dir=output_dir),
                              MiniGridTask(env_spec='MiniGrid-Empty-8x8-v0', num_timesteps=10000, time_batch_size=1,
                                           eval_mode=True, output_dir=output_dir)
                              ], output_dir=output_dir)
    }

    return experiments
