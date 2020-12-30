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
        self._observations = None
        self._last_timestep_data = None  # Always stores the last thing seen, even across "dones"
        self._cumulative_rewards = np.array([0 for _ in range(num_parallel_envs)], dtype=np.float)

        # Used to determine what to save off to logs and when
        self._observations_to_render = []
        self._timesteps_since_last_render = 0
        self._total_timesteps = 0

    def _preprocess_raw_observations(self, preprocessor, raw_observations):
        return torch.stack([preprocessor.preprocess(raw_observation) for raw_observation in raw_observations])

    def _initialize_envs(self, env_spec, time_batch_size, preprocessor):
        env_specs = [env_spec for _ in range(self._num_parallel_envs)]
        self._parallel_env = ParallelEnv(env_specs, self._output_dir)

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

    def collect_data(self, task_spec):
        """
        Passes observations to the policy of shape [#envs, time, **env.observation_shape]
        """
        time_batch_size = task_spec.time_batch_size
        env_spec = task_spec.env_spec
        preprocessor = task_spec.preprocessor
        action_space_id = task_spec.action_space_id
        eval_mode = task_spec.eval_mode

        # If the task requires fewer collections than the policy specifies, only collect that number
        timesteps_to_collect = min(self._timesteps_per_collection, task_spec.num_timesteps)

        # The per-environment data is contained within each TimestepData object, stored within per_timestep_data
        per_timestep_data = []
        rewards_to_report = []
        logs_to_report = []  # {tag, type ("video", "scalar"), value, timestep}
        num_timesteps = 0

        if self._parallel_env is None:
            self._initialize_envs(env_spec, time_batch_size, preprocessor)

        for timestep_id in range(timesteps_to_collect):
            stacked_observations = torch.stack(list(self._observations), dim=1)
            actions, timestep_data = self._policy.compute_action(stacked_observations,
                                                                 action_space_id,
                                                                 self._last_timestep_data,
                                                                 eval_mode)

            # ParallelEnv automatically resets the env and returns the new observation when a "done" occurs
            result = self._parallel_env.step(actions)
            raw_observations, rewards, dones, infos = list(result)

            self._total_timesteps += self._num_parallel_envs
            self._observations.append(self._preprocess_raw_observations(preprocessor, raw_observations))
            self._last_timestep_data = timestep_data
            self._cumulative_rewards += np.array(rewards)

            # For logging video, take the most recent first env's observation and save it. Once we finish an episode, if
            # we've exceeded the render frequency (in timesteps) we will save off the most recent episode's video
            # Without the deepcopy, the reset overwrites the end of observations_to_render
            self._observations_to_render.append(copy.deepcopy(self._observations[-1][0]))
            self._timesteps_since_last_render += self._num_parallel_envs

            for env_id, done in enumerate(dones):
                if done:
                    # The last observation was populated from the new environment. Grab it and reset the rest from it.
                    new_observation = self._observations[-1][env_id]
                    self._reset_observations_for_env(new_observation, time_batch_size, env_id)

                    rewards_to_report.append(self._cumulative_rewards[env_id])
                    self._cumulative_rewards[env_id] *= 0  # Simple method to ensure the shape is right but the total is 0

                    # Save off observations to enable viewing behavior
                    if env_id == 0:
                        if self._render_collection_freq is not None and \
                                self._timesteps_since_last_render >= self._render_collection_freq:
                            try:
                                # As with resetting, remove the last element because it's from the next episode
                                rendered_episode = preprocessor.render_episode(self._observations_to_render[:-1])
                                logs_to_report.append({"type": "video",
                                                       "tag": "behavior_video",
                                                       "value": rendered_episode,
                                                       "timestep": self._total_timesteps})
                            except NotImplementedError:
                                # If the task hasn't implemented rendering, it may simply not be feasible, so just
                                # let it go.
                                pass

                            self._timesteps_since_last_render = 0

                        # Reset the observations to render except keep the last frame because it's from the next episode
                        self._observations_to_render = [self._observations_to_render[-1]]

            # Finish populating the info to store with the collected data
            timestep_data.reward = rewards
            timestep_data.done = dones
            timestep_data.info = infos
            per_timestep_data.append(timestep_data)
            num_timesteps += self._num_parallel_envs

        # Tasks expect a list of lists for timestep data, to support different forms of parallelization, so return
        # per_timestep_data as a list
        return num_timesteps, [per_timestep_data], rewards_to_report, logs_to_report
