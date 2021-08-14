from .image_task import ImageTask

from crl_alfred import AlfredDemoBasedThorEnv
from crl_alfred.wrappers import ChannelConcatGoal

def get_alfred_demo_based_thor_task(
    which_set,
    demo_names,
    runs_per_demo=1,
    eval_mode=False,
    max_episode_steps=1000,
    continual_eval=True,
):
    task = ImageTask(
        action_space_id=0,  # shared action space
        env_spec=lambda: ChannelConcatGoal(AlfredDemoBasedThorEnv(which_set, demo_names)),
        num_timesteps=max_episode_steps * runs_per_demo * len(demo_names),
        time_batch_size=1,  # no framestack
        eval_mode=eval_mode,
        continual_eval=continual_eval,
        image_size=[64, 64],
        grayscale=False,
    )
    return task
