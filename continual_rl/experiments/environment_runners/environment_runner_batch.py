import torch
import numpy as np
from collections import deque
from torch_ac.utils.penv import ParallelEnv
from continual_rl.experiments.environment_runners.environment_runner_base import EnvironmentRunnerBase
from continual_rl.utils.utils import Utils


class EnvironmentRunnerBatch(EnvironmentRunnerBase):
    """
    Passes a batch of observations into the policy, gets a batch of actions out, and runs the environments in parallel.

    The arguments provided to __init__ are from the policy.
    The arguments provided to collect_data are from the task.
    """
    def __init__(self, policy, num_parallel_envs, timesteps_per_collection, render_collection_freq=None):
        super().__init__()
        self._policy = policy
        self._num_parallel_envs = num_parallel_envs
        self._timesteps_per_collection = timesteps_per_collection
        self._render_collection_freq = render_collection_freq  # In timesteps

        self._parallel_env = None
        self._observations = None
        self._last_info_to_store = None  # Always stores the last thing seen, even across "dones"
        self._cumulative_rewards = np.array([0 for _ in range(num_parallel_envs)], dtype=np.float)

        # Used to determine what to save off to logs and when
        self._observations_to_render = []
        self._env_0_timesteps_since_last_render = 0
        self._total_timesteps = 0

    def _preprocess_raw_observations(self, preprocessor, raw_observations):
        return torch.stack([preprocessor(raw_observation) for raw_observation in raw_observations])

    def _initialize_envs(self, env_spec, time_batch_size, preprocessor):
        envs = [Utils.make_env(env_spec) for _ in range(self._num_parallel_envs)]
        self._parallel_env = ParallelEnv(envs)

        # Initialize the observation time-batch with n of the first observation.
        raw_observations = self._parallel_env.reset()
        processed_observations = self._preprocess_raw_observations(preprocessor, raw_observations)

        self._observations = self._initialize_observations(processed_observations, time_batch_size)

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

    def _initialize_observations(self, processed_observations, time_batch_size):
        observations = deque(maxlen=time_batch_size)

        for _ in range(time_batch_size):
            observations.append(processed_observations)

        return observations

    def _reset_observations_for_env(self, reset_observation, time_batch_size, env_id):
        for time_id in range(time_batch_size):
            self._observations[time_id][env_id] = reset_observation

    def collect_data(self, time_batch_size, env_spec, preprocessor, action_space_id, episode_renderer=None):
        """
        Passes observations to the policy of shape [#envs, time, **env.observation_shape]
        """
        # The per-environment data is contained within the info_to_stores stored within per_timestep_data
        per_timestep_data = []
        rewards_to_report = []
        logs_to_report = []  # {tag, type ("video", "scalar"), value, timestep}

        if self._parallel_env is None:
            self._initialize_envs(env_spec, time_batch_size, preprocessor)

        for timestep_id in range(self._timesteps_per_collection):
            stacked_observations = torch.stack(list(self._observations), dim=1)
            actions, info_to_store = self._policy.compute_action(stacked_observations,
                                                                 action_space_id,
                                                                 self._last_info_to_store)

            # ParallelEnv automatically resets the env and returns the new observation when a "done" occurs
            result = self._parallel_env.step(actions)
            raw_observations, rewards, dones, infos = list(result)

            self._total_timesteps += self._num_parallel_envs
            self._observations.append(self._preprocess_raw_observations(preprocessor, raw_observations))
            self._last_info_to_store = info_to_store
            self._cumulative_rewards += np.array(rewards)

            # For logging video, take the most recent first env's observation and save it. Once we finish an episode, if
            # we've exceeded the render frequency (in timesteps) we will save off the most recent episode's video
            self._observations_to_render.append(self._observations[-1][0])
            self._env_0_timesteps_since_last_render += 1

            for env_id, done in enumerate(dones):
                if done:
                    # The last observation was populated from the new environment. Grab it and reset the rest from it.
                    new_observation = self._observations[-1][env_id]
                    self._reset_observations_for_env(new_observation, time_batch_size, env_id)

                    rewards_to_report.append(self._cumulative_rewards[env_id])
                    self._cumulative_rewards[env_id] *= 0  # Simple method to ensure the shape is right but the total is 0

                    # Save off observations to enable viewing behavior
                    if env_id == 0:
                        if self._render_collection_freq is not None and episode_renderer is not None and \
                                self._env_0_timesteps_since_last_render > self._render_collection_freq:
                            try:
                                rendered_episode = episode_renderer(self._observations_to_render)
                                logs_to_report.append({"type": "video",
                                                       "tag": "behavior_video",
                                                       "value": rendered_episode,
                                                       "timestep": self._total_timesteps})
                            except NotImplementedError:
                                # If the task hasn't implemented rendering, it may simply not be feasible, so just
                                # let it go.
                                pass

                            self._env_0_timesteps_since_last_render = 0

                        self._observations_to_render.clear()

            # Finish populating the info to store with the collected data
            info_to_store.reward = rewards
            info_to_store.done = dones
            per_timestep_data.append(info_to_store)

        timesteps = self._num_parallel_envs * self._timesteps_per_collection

        # Tasks expect a list of lists for timestep data, to support different forms of parallelization, so return
        # per_timestep_data as a list
        return timesteps, [per_timestep_data], rewards_to_report, logs_to_report
