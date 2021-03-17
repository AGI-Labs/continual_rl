from continual_rl.policies.timestep_data_base import TimestepDataBase


class MockTimestepData(TimestepDataBase):
    def __init__(self, data_to_store, memory=None):
        super().__init__()
        self.data_to_store = data_to_store
        self.memory = memory
