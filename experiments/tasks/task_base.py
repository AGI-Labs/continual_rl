from abc import ABC, abstractmethod


class TaskBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def preprocess(self, observation):
        pass

    def run(self, policy, task_id, summary_writer):
        raise NotImplementedError("Coming soon")
