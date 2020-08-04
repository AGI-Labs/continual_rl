from continual_rl.policies.timestep_data_base import TimestepDataBase


class PrototypeTimestepData(TimestepDataBase):
    def __init__(self):
        super().__init__()
