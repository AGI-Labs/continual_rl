from continual_rl.policies.timestep_data_base import TimestepDataBase


class SaneTimestepDataSingle(TimestepDataBase):
    def __init__(self, data_blob, per_episode_storage):
        super().__init__()
        self.data_blob = data_blob  # TODO: temp
        self.per_episode_storage = per_episode_storage


class SaneTimestepDataBatch(SaneTimestepDataSingle):
    def __init__(self, data_blob, per_episode_storage):
        super().__init__(data_blob, per_episode_storage)
        self.creation_buffer = None

    def convert_to_array_of_singles(self):
        env_sorted_data_blobs = []

        for env_id, data_blob_entry in enumerate(self.data_blob):
            single_storage = SaneTimestepDataSingle(data_blob_entry,
                                                    self.per_episode_storage[env_id])
            single_storage.reward = self.reward[env_id]
            single_storage.done = self.done[env_id]
            env_sorted_data_blobs.append(single_storage)

        return env_sorted_data_blobs
