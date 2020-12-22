from continual_rl.experiments.environment_runners.environment_runner_base import EnvironmentRunnerBase
from continual_rl.utils.utils import Utils
from dotmap import DotMap
import os


class ImpalaEnvironmentRunner(EnvironmentRunnerBase):
    """
    IMPALA is kind of a special case. Rather than re-implementing the asychrony as they do it, we're just letting
    FAIR's torchbeast code take care of all of it, and this runner forms the thinnest wrapper it can, while still
    ensuring the same basic model is used each time, and the right environments are loaded.
    Quick refs:
    IMPALA paper: https://arxiv.org/pdf/1802.01561.pdf
    IMPALA base implementation: https://github.com/facebookresearch/torchbeast
    CLEAR paper: https://arxiv.org/pdf/1811.11682.pdf
    """

    def __init__(self, config, policy):
        super().__init__()
        self._config = config
        self._policy = policy
        self._result_generators = {}
        self._last_step_returned = 0

    @property
    def _logger(self):
        logger = Utils.create_logger(f"{self._config.output_dir}/impala.log")
        return logger

    def _create_task_flags(self, task_spec):
        flags = DotMap()
        flags.env_spec = task_spec.env_spec

        # Really just needs to not be "test_render", but these are the intended options
        flags.mode = "test" if task_spec.eval_mode else "train"

        # IMPALA is handling all training, thus task_base can't enforce the number of steps. Instead we just
        # tell IMPALA how long to run
        flags.total_steps = task_spec.num_timesteps

        return flags

    def _initialize_data_generator(self, task_spec):
        os.environ["OMP_NUM_THREADS"] = "1"  # TODO: check this works. It is necessary. (Note, doesnt' work...where can this be set automatically?)
        task_flags = self._create_task_flags(task_spec)

        if task_spec.eval_mode:
            num_episodes = task_spec.return_after_episode_num
            result_generator = self._policy.impala_trainer.test(task_flags, num_episodes=num_episodes)
        else:
            result_generator = self._policy.impala_trainer.train(task_flags)

        return result_generator

    def collect_data(self, task_spec):
        self._policy.set_action_space(task_spec.action_space_id)

        if task_spec not in self._result_generators:
            self._result_generators[task_spec] = self._initialize_data_generator(task_spec)

        result_generator = self._result_generators[task_spec]

        try:
            stats = next(result_generator)
        except StopIteration:
            stats = None

            if task_spec.eval_mode:  # If we want to start again, we'll have to re-initialize
                del self._result_generators[task_spec]

        all_env_data = []
        rewards_to_report = []
        logs_to_report = []

        if stats is not None:
            # Eval_mode only does one step of collection at a time, so this is the number of timesteps since last return
            if task_spec.eval_mode:
                timesteps = stats["step"]
            else:
                timesteps = stats["step"] - self._last_step_returned

            rewards_to_report = stats.get("episode_returns", [])

            if "total_loss" in stats:
                logs_to_report.append({"type": "scalar", "tag": "total_loss", "value": stats["total_loss"]})

            if "abs_max_vtrace_advantage" in stats:
                logs_to_report.append({"type": "scalar", "tag": "abs_max_vtrace_advantage", "value": stats["abs_max_vtrace_advantage"]})

            self._last_step_returned = stats["step"]
        else:
            # Forcibly end the task. (TODO: why is impala sometimes getting almost but not quite to the end?)
            timesteps = task_spec.num_timesteps - self._last_step_returned
            self._last_step_returned = task_spec.num_timesteps

        return timesteps, all_env_data, rewards_to_report, logs_to_report

    def cleanup(self):
        del self._result_generators
