from continual_rl.policies.info_to_store_base import InfoToStoreBase


class MockInfoToStore(InfoToStoreBase):
    def __init__(self, data_to_store):
        super().__init__()
        self.data_to_store = data_to_store
