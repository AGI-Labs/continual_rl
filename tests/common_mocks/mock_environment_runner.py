from continual_rl.experiments.environment_runners.environment_runner_base import EnvironmentRunnerBase
from tests.common_mocks.mock_policy.mock_timestep_data import MockTimestepData


class MockEnvironmentRunner(EnvironmentRunnerBase):
    def __init__(self):
        super().__init__()
        self._call_count = 0
        self.cleanup_called = False

    def collect_data(self, task_spec):
        timesteps = 10
        all_env_data = [[MockTimestepData({"foo": 1}), MockTimestepData({"foo": 2})],
                        [MockTimestepData({"foo": 3}), MockTimestepData({"foo": 4})]]
        rewards_to_report = [self._call_count, self._call_count+1, self._call_count+2]
        logs_to_report = [{"step_count": 456+self._call_count},
                          {"eval_mode": task_spec.eval_mode}]
        self._call_count += 1

        return timesteps, all_env_data, rewards_to_report, logs_to_report

    def cleanup(self, task_spec):
        self.cleanup_called = True
