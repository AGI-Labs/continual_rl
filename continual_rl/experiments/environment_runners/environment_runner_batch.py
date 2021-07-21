import torch
import numpy as np
from collections import deque
from continual_rl.experiments.environment_runners.parallel_env import ParallelEnv
from continual_rl.experiments.environment_runners.environment_runner_base import EnvironmentRunnerBase
import copy


class EnvironmentRunnerBatch(EnvironmentRunnerBase):
    """
    Passes a batch of observations into the policy, gets a batch of actions out, and runs the environments in parallel.

    The arguments provided to __init__ are from the policy.
    The arguments provided to collect_data are from the task.
    """
    def __init__(self, policy, num_parallel_envs, timesteps_per_collection, render_collection_freq=None,
                 output_dir=None):
        super().__init__()
        self._policy = policy
        self._num_parallel_envs = num_parallel_envs
        self._timesteps_per_collection = timesteps_per_collection
        self._render_collection_freq = render_collection_freq  # In timesteps
        self._output_dir = output_dir

        self._parallel_env = None
        self._last_observations = None  # To allow returning mid-episode
        self._last_timestep_data = None  # Always stores the last thing seen, even across "dones"
        self._cumulative_rewards = np.array([0 for _ in range(num_parallel_envs)], dtype=np.float)

        # Used to determine what to save off to logs and when
        self._observations_to_render = []
        self._timesteps_since_last_render = 0
        self._total_timesteps = 0

    def _preprocess_raw_observations(self, preprocessor, raw_observations):
        return preprocessor.preprocess(raw_observations)

    def _initialize_envs(self, env_spec, preprocessor):
        if self._parallel_env is None:
            env_specs = [env_spec for _ in range(self._num_parallel_envs)]
            self._parallel_env = ParallelEnv(env_specs, self._output_dir)

        # Initialize the observation time-batch with n of the first observation.
        raw_observations = self._parallel_env.reset()
        processed_observations = self._preprocess_raw_observations(preprocessor, raw_observations)
        return processed_observations

    def _reset_env(self, env_id):
        """
        ParallelEnv doesn't readily expose manually resetting an environment, so doing that here.
        """
        if env_id == 0:
            observation = self._parallel_env.envs[0].reset()
        else:
            local = self._parallel_env.locals[env_id-1]
            local.send(("reset", None))
            observation = local.recv()

        return observation

    def _render_video(self, preprocessor):
        """
        Only renders if it's time, per the render_collection_freq
        """
        video_log = None

        if self._render_collection_freq is not None and \
                self._timesteps_since_last_render >= self._render_collection_freq:
            try:
                # As with resetting, remove the last element because it's from the next episode
                rendered_episode = preprocessor.render_episode(self._observations_to_render[:-1])
                video_log = {"type": "video",
                             "tag": "behavior_video",
                             "value": rendered_episode,
                             "timestep": self._total_timesteps}
            except NotImplementedError:
                # If the task hasn't implemented rendering, it may simply not be feasible, so just
                # let it go.
                pass

            self._timesteps_since_last_render = 0

        # Reset the observations to render except keep the last frame because it's from the next episode
        self._observations_to_render = [self._observations_to_render[-1]]
        return video_log

    def collect_data(self, task_spec):
        """
        Passes observations to the policy of shape [#envs, time, **env.observation_shape]
        """
        env_spec = task_spec.env_spec
        preprocessor = task_spec.preprocessor
        task_id = task_spec.task_id
        action_space_id = task_spec.action_space_id
        eval_mode = task_spec.eval_mode
        return_after_episode_num = task_spec.return_after_episode_num

        # If the task requires fewer collections than the policy specifies, only collect that number
        timesteps_to_collect = min(self._timesteps_per_collection, task_spec.num_timesteps)

        # The per-environment data is contained within each TimestepData object, stored within per_timestep_data
        per_timestep_data = []
        returns_to_report = []
        logs_to_report = []  # {tag, type ("video", "scalar"), value, timestep}
        num_timesteps = 0

        # Grabbed the saved-off observations, if applicable.
        if self._last_observations is None:
            processed_observations = self._initialize_envs(env_spec, preprocessor)
        else:
            processed_observations = self._last_observations

        for timestep_id in range(timesteps_to_collect):
            actions, timestep_data = self._policy.compute_action(processed_observations,
                                                                 task_id,
                                                                 action_space_id,
                                                                 self._last_timestep_data,
                                                                 eval_mode)

            # ParallelEnv automatically resets the env and returns the new observation when a "done" occurs
            result = self._parallel_env.step(actions)
            raw_observations, rewards, dones, infos = list(result)

            self._total_timesteps += self._num_parallel_envs
            self._last_timestep_data = timestep_data
            processed_observations = self._preprocess_raw_observations(preprocessor, raw_observations)
            self._last_observations = processed_observations  # Save it off so we can resume if we finish the collection

            # If we're expecting the environment to keep track of this for us (EpisodicLifeEnv) use that.
            # Otherwise accumulate ourselves
            if "episode_return" in infos[0]:
                for env_id, env_info in enumerate(infos):
                    # The episode return will be None if the episode is not yet over, but Nones can't be stored in
                    # numpy arrays, so convert to np.nan.
                    val_to_store = env_info["episode_return"] if env_info["episode_return"] is not None else np.nan
                    self._cumulative_rewards[env_id] = val_to_store
            else:
                self._cumulative_rewards += np.array(rewards)

            # For logging video, take the first env's most recent observation and save it.
            # Without the deepcopy, the reset overwrites the end of observations_to_render
            self._observations_to_render.append(copy.deepcopy(processed_observations[0][-1]))
            self._timesteps_since_last_render += self._num_parallel_envs

            for env_id, done in enumerate(dones):
                if done:
                    # It may not be a "real" done (e.g. EpisodicLifeEnv), so only log it out if it is
                    if not np.isnan(self._cumulative_rewards[env_id]):
                        returns_to_report.append(self._cumulative_rewards[env_id])

                    self._cumulative_rewards[env_id] = 0

                    # Save off observations to enable viewing behavior
                    if env_id == 0:
                        render_log = self._render_video(preprocessor)
                        if render_log is not None:
                            logs_to_report.append(render_log)

            # Finish populating the info to store with the collected data
            timestep_data.reward = rewards
            timestep_data.done = dones
            timestep_data.info = infos
            per_timestep_data.append(timestep_data)
            num_timesteps += self._num_parallel_envs

            if return_after_episode_num is not None and len(returns_to_report) >= return_after_episode_num:
                break

        # Tasks expect a list of lists for timestep data, to support different forms of parallelization, so return
        # per_timestep_data as a list
        return num_timesteps, [per_timestep_data], returns_to_report, logs_to_report

    def cleanup(self, task_spec):
        self._parallel_env.close()
