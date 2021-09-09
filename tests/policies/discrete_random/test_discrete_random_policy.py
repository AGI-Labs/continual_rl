import os
from pathlib import Path
from continual_rl.experiments.experiment import Experiment
from continual_rl.experiments.tasks.image_task import ImageTask
from continual_rl.policies.discrete_random.discrete_random_policy_config import DiscreteRandomPolicyConfig
from continual_rl.policies.discrete_random.discrete_random_policy import DiscreteRandomPolicy


class TestDiscreteRandomPolicy(object):

    def test_end_to_end_batch(self, set_tmp_directory, cleanup_experiment, request):
        """
        Not a unit test - a full (very short) run with Discrete Random for a sanity check that it's working.
        This is testing: DiscreteRandomPolicy, ImageTask
        """
        # Arrange
        experiment = Experiment(tasks=[
            ImageTask(task_id="some_id", action_space_id=0,
                      env_spec='BreakoutDeterministic-v4',
                      num_timesteps=10, time_batch_size=4, eval_mode=False,
                      image_size=[84, 84], grayscale=True)
        ])
        config = DiscreteRandomPolicyConfig()
        config.num_parallel_envs = 2  # To make it batched

        # Make a subfolder of the output directory that only this experiment is using, to avoid conflict
        output_dir = Path(request.node.experiment_output_dir, "discrete_random_batch")
        os.makedirs(output_dir)
        experiment.set_output_dir(output_dir)
        config.set_output_dir(output_dir)

        policy = DiscreteRandomPolicy(config, experiment.observation_space, experiment.action_spaces)

        # Act
        experiment.try_run(policy, summary_writer=None)

        # Assert
        assert Path(policy._config.output_dir, "core_process.log").is_file(), "Log file not created"

    def test_end_to_end_sync(self, set_tmp_directory, cleanup_experiment, request):
        """
        Not a unit test - a full (very short) run with Discrete Random for a sanity check that it's working.
        This is testing: DiscreteRandomPolicy, ImageTask
        """
        # Arrange
        experiment = Experiment(tasks=[
            ImageTask(task_id="end_to_end_sync", action_space_id=0,
                      env_spec='BreakoutDeterministic-v4',
                      num_timesteps=10, time_batch_size=4, eval_mode=False,
                      image_size=[84, 84], grayscale=True)
        ])
        config = DiscreteRandomPolicyConfig()
        config.num_parallel_envs = None  # To make it sync

        # Make a subfolder of the output directory that only this experiment is using, to avoid conflict
        output_dir = Path(request.node.experiment_output_dir, "discrete_random_sync")
        os.makedirs(output_dir)
        experiment.set_output_dir(output_dir)
        config.set_output_dir(output_dir)

        policy = DiscreteRandomPolicy(config, experiment.observation_space, experiment.action_spaces)

        # Act
        experiment.try_run(policy, summary_writer=None)

        # Assert
        assert Path(policy._config.output_dir, "core_process.log").is_file(), "Log file not created"
