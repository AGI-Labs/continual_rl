from continual_rl.experiments.tasks.task_base import TaskBase


class MockTask(TaskBase):
    def __init__(self, action_space_id, env_spec, observation_size, action_space, time_batch_size, num_timesteps, eval_mode):
        super().__init__(action_space_id, env_spec, observation_size, action_space, time_batch_size, num_timesteps, eval_mode)

    def preprocess(self, observation):
        return observation
