from continual_rl.experiments.environment_runners.environment_runner_sync import EnvironmentRunnerSync


class PlayEnvironmentRunner(EnvironmentRunnerSync):
    """
    This Environment Runner is a Sync runner that adds in some key-control and rendering functionality.
    """
    def __init__(self, policy, timesteps_per_collection, on_key_press, on_key_release, output_dir=None):
        super().__init__(policy, timesteps_per_collection, render_collection_freq=None, output_dir=output_dir)
        self._on_key_press = on_key_press
        self._on_key_release = on_key_release
        self._policy = policy

    def collect_data(self, task_spec):
        collection_results = super().collect_data(task_spec)

        self._batch_runner._parallel_env._local_env.render()  # To create the window the event handlers need
        self._batch_runner._parallel_env._local_env.unwrapped.viewer.window.on_key_press = self._on_key_press
        self._batch_runner._parallel_env._local_env.unwrapped.viewer.window.on_key_release = self._on_key_release
        self._policy.on_env_ready(self._batch_runner._parallel_env._local_env)

        return collection_results
