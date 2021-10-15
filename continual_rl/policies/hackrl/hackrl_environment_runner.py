from continual_rl.experiments.environment_runners.environment_runner_base import EnvironmentRunnerBase
from continual_rl.utils.utils import Utils
from dotmap import DotMap
import torch


class HackRLEnvironmentRunner(EnvironmentRunnerBase):
    """
    TODO
    """

    def __init__(self, config, policy):
        super().__init__()
        self._config = config
        self._policy = policy
        self._timesteps_since_last_render = 0
        self._result_generator = None

    @property
    def _logger(self):
        logger = Utils.create_logger(f"{self._config.output_dir}/hackrl.log")
        return logger

    def _create_task_flags(self, task_spec):
        flags = DotMap()
        """flags.action_space_id = task_spec.action_space_id
        flags.task_id = task_spec.task_id
        flags.env_spec = task_spec.env_spec

        # Really just needs to not be "test_render", but these are the intended options
        flags.mode = "test" if task_spec.eval_mode else "train"""

        flags.env_spec = task_spec.env_spec
        
        # HackRL is handling all training, thus task_base can't enforce the number of steps. Instead we just
        # tell HackRL how long to run
        flags.total_steps = task_spec.num_timesteps

        return flags

    def _get_render_log(self, preprocessor, observations, tag):
        rendered_episode = preprocessor.render_episode(observations)
        video_log = {"type": "video",
                     "tag": tag,
                     "value": rendered_episode}
        return video_log

    def _render_video(self, preprocessor, observations_to_render, force_render):
        """
        Only renders if it's time, per the render_collection_freq, unless the force_render flag is set
        """
        video_logs = []

        if force_render or (self._config.render_freq is not None and self._timesteps_since_last_render >= self._config.render_freq):
            try:
                # TODO: the preprocessor should handle creating different videos, not the policy. Tracked by #108
                if isinstance(observations_to_render[0], torch.Tensor) and observations_to_render[0].shape[0] == 6:
                    actor_observatons = [obs[:3] for obs in observations_to_render]
                    goal_observatons = [obs[3:] for obs in observations_to_render]
                    video_logs.append(self._get_render_log(preprocessor, actor_observatons, "behavior_video"))
                    video_logs.append(self._get_render_log(preprocessor, goal_observatons, "goal_video"))
                else:
                    video_logs.append(self._get_render_log(preprocessor, observations_to_render, "behavior_video"))
            except NotImplementedError:
                # If the task hasn't implemented rendering, it may simply not be feasible, so just
                # let it go.
                pass

            self._timesteps_since_last_render = 0

        return video_logs

    def collect_data(self, task_spec):
        if self._result_generator is None:
            task_flags = self._create_task_flags(task_spec)
            self._result_generator = self._policy.learner.train(task_flags)

        last_steps_done = 0
        last_episodes_done = 0

        try:
            stats = next(self._result_generator)
        except StopIteration:
            stats = None

        # Compute the deltas ourselves for now, for convenience (TODO: this caused problems with monobeast...double check)
        steps_done = stats["steps_done"].result()
        timesteps_delta = steps_done - last_steps_done
        last_steps_done = steps_done

        episodes_done = stats["episodes_done"].result()
        episodes_done_delta = episodes_done - last_episodes_done
        last_episodes_done = episodes_done

        # Currently hackrl doesn't return individual episode results, so instead fake it by using the mean x #episodes
        # TODO: this would mess up standard deviation stats, and is overall kind of misleading....
        mean_episode_return = stats["mean_episode_return"].result()
        rewards_to_report = [mean_episode_return for _ in range(episodes_done_delta)]

        # Data for training, which we don't need, so we don't keep
        all_env_data = []

        # Report out everything hackrl is giving us. Might be unnecessary but... (TODO: video)
        logs_to_report = []
        for key in stats.keys():
            logs_to_report.append({"type": "scalar", "tag": key, "value": stats[key].result(), "timestep": steps_done})

        return timesteps_delta, all_env_data, rewards_to_report, logs_to_report 

        """assert len(self._result_generators) == 0 or task_spec in self._result_generators
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
        timesteps = 0

        if stats is not None:
            # Eval_mode only does one step of collection at a time, so this is the number of timesteps since last return
            if task_spec.eval_mode:
                timesteps = stats["step"]
            else:
                timesteps = stats["step_delta"]

            self._timesteps_since_last_render += timesteps
            logs_to_report.append({"type": "scalar", "tag": "timesteps_since_render",
                                   "value": self._timesteps_since_last_render})
            rewards_to_report = stats.get("episode_returns", [])

            for key in stats.keys():
                if key.endswith("loss") or key == "total_norm":
                    logs_to_report.append({"type": "scalar", "tag": key, "value": stats[key]})

            if "video" in stats and stats["video"] is not None:
                video_log = self._render_video(task_spec.preprocessor, stats["video"], force_render=task_spec.eval_mode)
                if video_log is not None:
                    logs_to_report.extend(video_log)

        return timesteps, all_env_data, rewards_to_report, logs_to_report"""

    def cleanup(self, task_spec):
        if not task_spec.eval_mode:
            self._policy.impala_trainer.cleanup()
        del self._result_generators
