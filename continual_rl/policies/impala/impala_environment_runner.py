from continual_rl.experiments.environment_runners.environment_runner_base import EnvironmentRunnerBase
from continual_rl.utils.utils import Utils
from dotmap import DotMap
import torch


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
        self._timesteps_since_last_render = 0

    @property
    def _logger(self):
        logger = Utils.create_logger(f"{self._config.output_dir}/impala.log")
        return logger

    def _create_task_flags(self, task_spec):
        flags = DotMap()
        flags.action_space_id = task_spec.action_space_id
        flags.task_id = task_spec.task_id
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
            num_episodes = num_episodes if num_episodes is not None else 10  # Collect 10 episodes at a time
            result_generator = self._policy.impala_trainer.test(task_flags, num_episodes=num_episodes)
        else:
            result_generator = self._policy.impala_trainer.train(task_flags)

        return result_generator

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

        # TODO: more generally, and de-dupe with hackrl
        def get_hunger(observation):
            return observation["internal"].squeeze(0).squeeze(0)[7]
        hunger_delta = get_hunger(observations_to_render[-1]) - get_hunger(observations_to_render[0])
        video_logs.append({"type": "scalar", "tag": "hunger_delta", "value": hunger_delta})

        # TODO: more generally. Also note this doesn't include the last value, so ac-env won't look quite right
        def get_ac(observation):
            return observation["blstats"].squeeze(0).squeeze(0)[16]
        ac_delta = get_ac(observations_to_render[-1]) - get_ac(observations_to_render[0])
        video_logs.append({"type": "scalar", "tag": "ac_delta", "value": ac_delta})

        def get_hp(observation):
            return observation["blstats"].squeeze(0).squeeze(0)[10]
        hp_delta = get_hp(observations_to_render[-1]) - get_hp(observations_to_render[0])
        video_logs.append({"type": "scalar", "tag": "hp_delta", "value": hp_delta})

        video_logs.append({"type": "scalar", "tag": "episode_len", "value": len(observations_to_render)})

        # Final values, for innate reward
        video_logs.append({"type": "scalar", "tag": "final_ac", "value": get_ac(observations_to_render[-1])})
        video_logs.append({"type": "scalar", "tag": "final_hp", "value": get_hp(observations_to_render[-1])})
        video_logs.append({"type": "scalar", "tag": "final_hunger", "value": get_hunger(observations_to_render[-1])})
        video_logs.append({"type": "scalar", "tag": "final_innate_reward", "value": observations_to_render[-1]["innate_reward"]})

        return video_logs

    def collect_data(self, task_spec):
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

        return timesteps, all_env_data, rewards_to_report, logs_to_report

    def cleanup(self, task_spec):
        if not task_spec.eval_mode:
            self._policy.impala_trainer.cleanup()
        del self._result_generators
