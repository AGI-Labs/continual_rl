from continual_rl.policies.info_to_store_base import InfoToStoreBase


class ImpalaInfoToStore(InfoToStoreBase):
    def __init__(self, agent_state, action):
        super().__init__()
        self.agent_state = agent_state
        self.action = action
