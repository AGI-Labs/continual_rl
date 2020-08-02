from continual_rl.policies.timestep_data_base import TimestepDataBase


class ImpalaTimestepData(TimestepDataBase):
    def __init__(self, agent_state, action):
        super().__init__()
        self.agent_state = agent_state
        self.action = action
