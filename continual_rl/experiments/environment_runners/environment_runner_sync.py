from continual_rl.experiments.environment_runners.environment_runner_base import EnvironmentRunnerBase
from continual_rl.experiments.environment_runners.environment_runner_batch import EnvironmentRunnerBatch


class EnvironmentRunnerSync(EnvironmentRunnerBase):
    """
    An episode collection class that will collect the data synchronously, using one environment.
    This class is currently rather naively using the fact that Batch's first env is synchronous (based on ParallelEnv).

    The arguments provided to __init__ are from the policy.
    The arguments provided to collect_data are from the task.
    """
    def __init__(self, policy, timesteps_per_collection, render_collection_freq=None):
        super().__init__()
        self._batch_runner = EnvironmentRunnerBatch(policy=policy, timesteps_per_collection=timesteps_per_collection,
                                                    num_parallel_envs=1, render_collection_freq=render_collection_freq)

    def collect_data(self, time_batch_size, env_spec, preprocessor, action_space_id, episode_renderer=None):
        """
        Provides actions to the policy in the form [1, time, *env.observation_shape]
        Basically the same API as batch, but with a batch size of 1.
        """
        return self._batch_runner.collect_data(time_batch_size, env_spec, preprocessor, action_space_id)
