from continual_rl.policies.timestep_data_base import TimestepDataBase


class NdpmTimestepData(TimestepDataBase):
    def __init__(self, observation):
        super().__init__()
        self.observation = observation
