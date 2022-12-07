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

    @property
    def current_continual_eval_id(self):
        """
        To not re-run completed evals if we die part-way through a set, and to not do a training set before eval,
        if we were part-way. If None, an eval was not in process
        """
        return self._metadata.get("current_continual_eval_id", 0)  # TODO: initializing to 0 so we start with a continual eval set of runs (is this clear enough?)

    @property
    def last_continual_testing_step(self):
        return self._metadata.get("last_continual_testing_step", -1e8)  # TODO: what's a reasonable default here?

    def _get_path(self):
        return os.path.join(self._output_dir, "run_metadata.json")

    def load(self):
        path = self._get_path()
        if os.path.exists(path):
            with open(path, "r") as metadata_file:
                self._metadata = json.load(metadata_file)
        else:
            self._metadata = {}

    def save(self, cycle_id, task_id, task_timesteps, total_train_timesteps, continual_eval_id,
             last_continual_testing_step):
        self._metadata["cycle_id"] = cycle_id
        self._metadata["task_id"] = task_id
        self._metadata["task_timesteps"] = task_timesteps
        self._metadata["total_train_timesteps"] = total_train_timesteps
        self._metadata["current_continual_eval_id"] = continual_eval_id
        self._metadata["last_continual_testing_step"] = last_continual_testing_step

        path = self._get_path()
        with open(path, "w+") as metadata_file:
            json.dump(self._metadata, metadata_file)
