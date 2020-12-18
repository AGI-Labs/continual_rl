from continual_rl.experiments.environment_runners.environment_runner_base import EnvironmentRunnerBase


class MockEnvironmentRunner(EnvironmentRunnerBase):
    def collect_data(self, task_spec):
        timesteps =
        return timesteps, all_env_data, rewards_to_report, logs_to_report
