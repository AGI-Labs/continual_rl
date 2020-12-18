from continual_rl.experiments.environment_runners.environment_runner_base import EnvironmentRunnerBase
from tests.common_mocks.mock_policy.mock_timestep_data import MockTimestepData


class MockEnvironmentRunner(EnvironmentRunnerBase):
    def collect_data(self, task_spec):
        timesteps = 10
        all_env_data = [[MockTimestepData({"foo": 1}), MockTimestepData({"foo": 2})],
                        [MockTimestepData({"foo": 3}), MockTimestepData({"foo": 4})]]
        rewards_to_report = [10, 11, 12]
        logs_to_report = [{"type": "scalar", "value": 456}]

        return timesteps, all_env_data, rewards_to_report, logs_to_report
