from continual_rl.experiments.environment_runners.environment_runner_base import EnvironmentRunnerBase
from continual_rl.utils.utils import Utils
from dotmap import DotMap


class HackRLEnvironmentRunner(EnvironmentRunnerBase):
    """
    TODO
    """

    def __init__(self, config, policy):
        super().__init__()
        self._config = config
        self._policy = policy
        self._timesteps_since_last_render = 0
        self._last_timestep = 0
        self._last_episode_done = 0
        self._result_generator = None

    @property
    def _logger(self):
        logger = Utils.create_logger(f"{self._config.output_dir}/hackrl.log")
        return logger

    def _create_task_flags(self, task_spec):
        flags = DotMap()

        # TODO: testing. It seems like moolib is using the same seed? Explicitly setting a new one
        flags.env_spec = lambda: Utils.make_env(task_spec.env_spec, create_seed=True)[0]

        # Continual RL and hackRL both trying to control the number of steps readily leads to infinite loops
        # Make CRL be in charge, but base the hackRL on it, so if something goes wrong hackrl doesn't just go infinitely.
        flags.total_steps = task_spec.num_timesteps + 1e6  # TODO: what number?

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
                video_logs.append(self._get_render_log(preprocessor, observations_to_render, "behavior_video"))
            except NotImplementedError:
                # If the task hasn't implemented rendering, it may simply not be feasible, so just
                # let it go.
                pass

            self._timesteps_since_last_render = 0

        # TODO: more generally, and de-dupe with impala
        def get_hunger(observation):
            return 0 #observation["internal"].squeeze(0).squeeze(0)[7]
        hunger_delta = get_hunger(observations_to_render[-1]) - get_hunger(observations_to_render[0])
        video_logs.append({"type": "scalar", "tag": "hunger_delta", "value": hunger_delta})

        def get_ac(observation):
            return observation["blstats"].squeeze(0).squeeze(0)[16]
        ac_delta = get_ac(observations_to_render[-1]) - get_ac(observations_to_render[0])
        video_logs.append({"type": "scalar", "tag": "ac_delta", "value": ac_delta})

        video_logs.append({"type": "scalar", "tag": "episode_len", "value": len(observations_to_render)})
        
        return video_logs

    def collect_data(self, task_spec):
        if self._result_generator is None:
            task_flags = self._create_task_flags(task_spec)
            self._result_generator = self._policy.learner.train(task_flags)

        try:
            stats, trajectory_log = next(self._result_generator)
        except StopIteration:
            stats = None

        # Data for training, which we don't need, so we don't keep
        timesteps_delta = 0  # Might result in infinite looping, but doing it for right now anyway
        all_env_data = []
        rewards_to_report = []
        logs_to_report = []

        if stats is not None:
            # Compute the deltas ourselves for now, for convenience (TODO: this caused problems with monobeast...double check)
            steps_done = stats["steps_done"].result()
            timesteps_delta = steps_done - self._last_timestep
            self._last_timestep = steps_done

            episodes_done = stats["episodes_done"].result()
            episodes_done_delta = episodes_done - self._last_episode_done
            self._last_episode_done = episodes_done

            # Currently hackrl doesn't return individual episode results, so instead fake it by using the mean x #episodes
            # TODO: this would mess up standard deviation stats, and is overall kind of misleading....
            mean_episode_return = stats["mean_episode_return"].result()
            rewards_to_report = [mean_episode_return for _ in range(episodes_done_delta)]

            # Report out everything hackrl is giving us. Might be unnecessary but... (TODO: video)
            for key in stats.keys():
                logs_to_report.append({"type": "scalar", "tag": key, "value": stats[key].result(), "timestep": steps_done})

        self._timesteps_since_last_render += timesteps_delta
        if trajectory_log is not None:
            logs_to_report.extend(self._render_video(task_spec.preprocessor, trajectory_log, force_render=False))

        return timesteps_delta, all_env_data, rewards_to_report, logs_to_report 

    def cleanup(self, task_spec):
        self._policy.cleanup()
