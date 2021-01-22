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
        self._timesteps_since_last_render = 0

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
        task_flags = self._create_task_flags(task_spec)

        if task_spec.eval_mode:
            num_episodes = task_spec.return_after_episode_num
            result_generator = self._policy.impala_trainer.test(task_flags, num_episodes=num_episodes)
        else:
            result_generator = self._policy.impala_trainer.train(task_flags)

        return result_generator

    def _render_video(self, preprocessor, observations_to_render):
        """
        Only renders if it's time, per the render_collection_freq
        """
        video_log = None

        if self._config.render_freq is not None and self._timesteps_since_last_render >= self._config.render_freq:
            try:
                # As with resetting, remove the last element because it's from the next episode
                rendered_episode = preprocessor.render_episode(observations_to_render)
                video_log = {"type": "video",
                             "tag": "behavior_video",
                             "value": rendered_episode}
            except NotImplementedError:
                # If the task hasn't implemented rendering, it may simply not be feasible, so just
                # let it go.
                pass

            self._timesteps_since_last_render = 0

        return video_log

    def collect_data(self, task_spec):
        self._policy.set_action_space(task_spec.action_space_id)  # TODO: these aren't thread-safe. Make it so. (Same elsewhere. See sane_filebacked_cl_parallel
        self._policy.set_current_task_id(task_spec.task_id)

        assert len(self._result_generators) == 0 or task_spec in self._result_generators
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

            self._timesteps_since_last_render += timesteps
            rewards_to_report = stats.get("episode_returns", [])

            for key in stats.keys():
                if key.endswith("loss"):
                    logs_to_report.append({"type": "scalar", "tag": key, "value": stats[key]})

            if "video" in stats and stats["video"] is not None:
                video_log = self._render_video(task_spec.preprocessor, stats["video"])
                if video_log is not None:
                    logs_to_report.append(video_log)

            self._last_step_returned = stats["step"]
        else:
            # Forcibly end the task. (TODO: why is impala sometimes getting almost but not quite to the end?)
            timesteps = task_spec.num_timesteps - self._last_step_returned
            self._last_step_returned = task_spec.num_timesteps

        return timesteps, all_env_data, rewards_to_report, logs_to_report

    def cleanup(self):
        del self._result_generators
