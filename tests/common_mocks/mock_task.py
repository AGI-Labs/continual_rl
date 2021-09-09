from continual_rl.experiments.tasks.task_base import TaskBase
from tests.common_mocks.mock_preprocessor import MockPreprocessor


class MockTask(TaskBase):
    def __init__(self, task_id, action_space_id, env_spec, action_space, time_batch_size, num_timesteps, eval_mode):
        preprocessor = MockPreprocessor()
        super().__init__(task_id, action_space_id, preprocessor, env_spec, preprocessor.observation_space, action_space,
                         num_timesteps, eval_mode)
