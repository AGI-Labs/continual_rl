

class TaskBase(object):
    def __init__(self):
        pass

    def run(self, policy, task_id, summary_writer):
        raise NotImplementedError("Task's run method not implemented")
