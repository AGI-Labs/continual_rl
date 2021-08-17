import os
import json

from .image_task import ImageTask

from crl_alfred import AlfredDemoBasedThorEnv
from crl_alfred.wrappers import ChannelConcatGoal


def get_alfred_demo_based_thor_task(
    which_set,
    demo_names,
    num_timesteps,
    eval_mode=False,
    max_episode_steps=1000,
    continual_eval=True,
):
    task = ImageTask(
        action_space_id=0,  # shared action space
        env_spec=lambda: ChannelConcatGoal(AlfredDemoBasedThorEnv(which_set, demo_names)),
        num_timesteps=num_timesteps,
        time_batch_size=1,  # no framestack
        eval_mode=eval_mode,
        continual_eval=continual_eval,
        image_size=[64, 64],
        grayscale=False,
    )
    return task


def create_alfred_tasks_from_sequence(num_timesteps, max_episode_steps=1000):
    # Load in the task sequences: note they depend on specific trajectories (TODO: where will we put the official trajectories?)
    metadata_path = os.path.join(os.path.dirname(__file__), "metadata")

    #with open(os.path.join(metadata_path, 'alfred_task_sequences.json'), 'r') as f:
    with open(os.path.join(metadata_path, 'alfred_task_sequences_debug.json'), 'r') as f:  # TODO: remove this
        task_sequences = json.load(f)

    tasks = []
    for task_data in task_sequences:
        # Construct the train task, where training occurs
        train_demos = task_data["train"]
        train_task = get_alfred_demo_based_thor_task(
            "train",
            train_demos,
            num_timesteps=num_timesteps,
            eval_mode=False,
            continual_eval=True,  # Check recall
            max_episode_steps=max_episode_steps,
        )
        tasks.append(train_task)

        # Construct the validation task, where we check generalization
        validation_demos = task_data["valid_seen"]
        validation_task = get_alfred_demo_based_thor_task(
            "valid_seen",
            validation_demos,
            num_timesteps=10000,  # TODO:...
            eval_mode=True,
            continual_eval=True,
            max_episode_steps=max_episode_steps,
        )
        tasks.append(validation_task)

    return tasks
