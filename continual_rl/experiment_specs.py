from continual_rl.experiments.experiment import Experiment
from continual_rl.experiments.tasks.minigrid_task import MiniGridTask


def get_available_experiments(output_dir):
    experiments = {
        "recall_minigrid_empty8x8_unlock_empty8x8":
            Experiment(tasks=[MiniGridTask(env_spec='MiniGrid-Empty-8x8-v0', num_timesteps=150000, time_batch_size=1,
                                           eval_mode=False, output_dir=output_dir),
                              MiniGridTask(env_spec='MiniGrid-Unlock-v0', num_timesteps=5000000, time_batch_size=1,
                                           eval_mode=False, output_dir=output_dir),
                              MiniGridTask(env_spec='MiniGrid-Empty-8x8-v0', num_timesteps=10000, time_batch_size=1,
                                           eval_mode=True, output_dir=output_dir)
                              ], output_dir=output_dir)
    }

    return experiments
