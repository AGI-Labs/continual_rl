import os
import json


class RunMetadata(object):
    def __init__(self, output_dir):
        self._metadata = None
        self._output_dir = output_dir
        self.load()

    @property
    def cycle_id(self):
        return self._metadata.get("cycle_id", 0)

    @property
    def task_id(self):
        return self._metadata.get("task_id", 0)

    @property
    def task_timesteps(self):
        return self._metadata.get("task_timesteps", 0)

    @property
    def total_train_timesteps(self):
        return self._metadata.get("total_train_timesteps", 0)

    def _get_path(self):
        return os.path.join(self._output_dir, "run_metadata.json")

    def load(self):
        path = self._get_path()
        if os.path.exists(path):
            with open(path, "r") as metadata_file:
                self._metadata = json.load(metadata_file)
        else:
            self._metadata = {}

    def save(self, cycle_id, task_id, task_timesteps, total_train_timesteps):
        self._metadata["cycle_id"] = cycle_id
        self._metadata["task_id"] = task_id
        self._metadata["task_timesteps"] = task_timesteps
        self._metadata["total_train_timesteps"] = total_train_timesteps

        path = self._get_path()
        with open(path, "w+") as metadata_file:
            json.dump(self._metadata, metadata_file)
