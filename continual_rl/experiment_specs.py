from continual_rl.experiments.experiment import Experiment
from continual_rl.experiments.tasks.image_task import ImageTask


def get_available_experiments(output_dir):
    experiments = {
        "breakout":
            Experiment(tasks=[
                ImageTask(env_spec='BreakoutDeterministic-v4', num_timesteps=10000000, time_batch_size=4,
                          eval_mode=False, output_dir=output_dir, image_size=[3, 84, 84], grayscale=True)
            ], output_dir=output_dir)
    }

    return experiments
