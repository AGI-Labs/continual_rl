from neural_map.logic.pattern_buffer.experiment import MiniGridTask, ImageTask, Experiment
from neural_map.logic.pattern_buffer.envs.incremental_classification_env import IncrementalClassificationEnv, DatasetIds


def get_available_experiments(output_dir):
    experiments = {
        "recall_minigrid_empty8x8_unlock_empty8x8":
            Experiment(tasks=[MiniGridTask(env_name='MiniGrid-Empty-8x8-v0', num_timesteps=150000, time_batch_size=1,
                                           eval_mode=False, output_dir=output_dir,
                                           large_data_dir=output_dir),
                              MiniGridTask(env_name='MiniGrid-Unlock-v0', num_timesteps=5000000, time_batch_size=1,
                                           eval_mode=False, output_dir=output_dir,
                                           large_data_dir=output_dir),
                              MiniGridTask(env_name='MiniGrid-Empty-8x8-v0', num_timesteps=10000, time_batch_size=1,
                                           eval_mode=True, output_dir=output_dir,
                                           large_data_dir=output_dir)
                              ], output_dir=output_dir)
    }

    return experiments
