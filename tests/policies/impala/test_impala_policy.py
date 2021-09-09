import pytest
import os
import numpy as np
from pathlib import Path
from torch.utils.tensorboard.writer import SummaryWriter
from continual_rl.experiments.experiment import Experiment
from continual_rl.experiments.tasks.image_task import ImageTask
from continual_rl.policies.impala.impala_policy_config import ImpalaPolicyConfig


class TestImpalaPolicy(object):

    @pytest.mark.skip(reason="Torchbeast not currently being initialized on the test server")
    def test_end_to_end_batch(self, set_tmp_directory, cleanup_experiment, request):
        """
        Not a unit test - a full (very short) run with Impala for a sanity check that it's working.
        This is testing: ImpalaPolicy, MiniGridTask, SummaryWriter
        """
        from continual_rl.policies.impala.impala_policy import ImpalaPolicy

        # Arrange
        experiment = Experiment(
            tasks=[
                ImageTask(task_id="some_id", action_space_id=0,
                          env_spec='BreakoutDeterministic-v4',
                          num_timesteps=10, time_batch_size=4, eval_mode=False,
                          image_size=[84, 84], grayscale=True)])
        config = ImpalaPolicyConfig()

        # Make a subfolder of the output directory that only this experiment is using, to avoid conflict
        output_dir = Path(request.node.experiment_output_dir, "impala_full_parallel")
        os.makedirs(output_dir)
        experiment.set_output_dir(output_dir)
        config.set_output_dir(output_dir)

        policy = ImpalaPolicy(config, experiment.observation_space, experiment.action_spaces)
        summary_writer = SummaryWriter(log_dir=experiment.output_dir)

        # Act
        experiment.try_run(policy, summary_writer=summary_writer)

        # Assert
        assert Path(policy._config.output_dir, "core_process.log").is_file(), "Log file not created"
        assert np.any(['event' in file_name for file_name in os.listdir(experiment.output_dir)]), \
            "Summary writer file not created"
