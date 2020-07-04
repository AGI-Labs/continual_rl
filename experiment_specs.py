from experiments.experiment import Experiment
from experiments.tasks.image_task import ImageTask


def get_available_experiments(output_dir):
    experiments = {
        "breakout":
            Experiment(tasks=[
                ImageTask(env_spec='BreakoutDeterministic-v4', num_timesteps=10000000, time_batch_size=2,
                          eval_mode=False, output_dir=output_dir, image_size=[3, 84, 84])
            ], output_dir=output_dir)
    }

    return experiments
