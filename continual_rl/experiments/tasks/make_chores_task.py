import os
import json

from .image_task import ImageTask


def get_chores_task(
    task_id,
    which_set,
    demo_names,
    num_timesteps,
    eval_mode=False,
    max_episode_steps=1000,
    continual_eval=True,
):
    from crl_alfred import AlfredDemoBasedThorEnv
    from crl_alfred.wrappers import ChannelConcatGoal

    task = ImageTask(
        task_id=task_id,
        action_space_id=0,  # shared action space
        env_spec=lambda: ChannelConcatGoal(AlfredDemoBasedThorEnv(which_set, demo_names, max_steps=max_episode_steps)),
        num_timesteps=num_timesteps,
        time_batch_size=1,  # no framestack
        eval_mode=eval_mode,
        continual_eval=continual_eval,
        image_size=[64, 64],
        grayscale=False,
        resize_interp_method="INTER_LINEAR"
    )
    return task


def create_chores_tasks_from_sequence(task_prefix, sequence_file_name, num_timesteps, max_episode_steps=1000):
    # Load in the task sequences: note they depend on specific trajectories
    metadata_path = os.path.join(os.path.dirname(__file__), "metadata")

    with open(os.path.join(metadata_path, sequence_file_name), 'r') as f:
        task_sequences = json.load(f)

    tasks = []
    for task_id, task_data in enumerate(task_sequences):
        # Construct the train task, where training occurs
        train_demos = task_data["train"]
        train_task = get_chores_task(
            f"{task_prefix}_{task_id}",
            "train",
            train_demos,
            num_timesteps=num_timesteps,
            eval_mode=False,
            continual_eval=True,  # Check recall
            max_episode_steps=max_episode_steps,
        )
        tasks.append(train_task)

        # Construct the validation task, where we check generalization
        validation_demos = task_data.get("valid_seen", None)
        if validation_demos is not None:
            validation_task = get_chores_task(
                f"{task_prefix}_{task_id}_valid_seen",
                "valid_seen",
                validation_demos,
                num_timesteps=1000,
                eval_mode=True,
                continual_eval=True,
                max_episode_steps=max_episode_steps,
            )
            tasks.append(validation_task)

    return tasks
