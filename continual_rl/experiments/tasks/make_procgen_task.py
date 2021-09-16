import gym

from .image_task import ImageTask


def make_procgen(env_name, num_levels=0, start_level=0, distribution_mode="easy"):
    env = gym.make(
        f"procgen:procgen-{env_name}",
        num_levels=num_levels,
        start_level=start_level,
        distribution_mode=distribution_mode,
    )
    return env


def get_single_procgen_task(task_id, action_space_id, env_name, num_timesteps, eval_mode=False, **kwargs):
    return ImageTask(
        task_id=task_id,
        action_space_id=action_space_id,
        env_spec=lambda: make_procgen(env_name, **kwargs),
        num_timesteps=num_timesteps,
        time_batch_size=1,  # no framestack
        eval_mode=eval_mode,
        image_size=[64, 64],
        grayscale=False,
    )
