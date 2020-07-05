from abc import ABC, abstractmethod
import types
import gym


class TaskBase(ABC):
    def __init__(self, env_spec, obs_size, action_size, num_timesteps, time_batch_size, eval_mode, output_dir):
        pass

    @abstractmethod
    def preprocess(self, observation):
        pass

    def run(self, policy, task_id, summary_writer):
        raise NotImplementedError("Coming soon")

    @classmethod
    def _make_env(cls, env_spec):
        if isinstance(env_spec, types.LambdaType):
            env = env_spec()
        else:
            env = gym.make(env_spec)

        return env
