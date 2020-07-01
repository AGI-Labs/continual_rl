import os
import json
import copy
import time
import subprocess


class UnknownExperimentConfigEntry(Exception):
    pass


class ConfigBase(object):
    """
    This is the base class for the experiment configuration loader.
    It will automatically load a JSON file that is a list of dicts. Each dict is assumed to be a
    separate experiment that gets parsed by the particular implementation of this class.
    We will get the next experiment that has not yet been started, and return it.
    """
    def __init__(self, config_path, output_dir):
        self._config_path = config_path
        self._output_dir = output_dir  # General output dir where all experiments currently being run live
        self.experiment_output_dir = None  # Accessible output dir for the current run of the experiment

    def _load_single_experiment(self, config_json):
        """
        Load the parameters from the input json object into the current object (self).
        Should assert UnknownExperimentConfigEntry if something unknown was found.
        """
        raise NotImplementedError()

    @classmethod
    def _get_script_dir_commit_hash(cls):
        current_working_dir = os.getcwd()
        script_dir = os.path.realpath(__file__)
        print("Script file path: {}".format(script_dir))
        os.chdir(os.path.dirname(script_dir))

        commit = subprocess.check_output(["git", "describe", "--always"]).strip()

        os.chdir(current_working_dir)
        return commit

    @classmethod
    def _write_json_config_file(cls, experiment_json, output_path):  # TODO: create the json from class members instead...
        experiment_json = copy.deepcopy(experiment_json)
        experiment_json["commit"] = str(cls._get_script_dir_commit_hash())
        experiment_json["timestamp"] = str(time.time())

        output_file_path = os.path.join(output_path, "experiment.json")

        with open(output_file_path, "w") as output_file:
            output_file.write(json.dumps(experiment_json))

    def load_next_experiment(self):
        """
        Read the next entry from the config file and load it into this configuration object.
        Returns None if there is nothing further to load.
        """
        # Instead of dumping directly into the output directory, we'll make a folder with the same name as the experiment file.
        # This allows for multiple experiment sets
        json_experiment_name = os.path.basename(os.path.splitext(self._config_path)[0])
        output_directory = os.path.join(self._output_dir, json_experiment_name)

        try:
            os.makedirs(output_directory)
        except FileExistsError:
            pass

        with open(self._config_path) as json_file:
            json_raw = json_file.read()
            experiments = json.loads(json_raw)

        existing_experiments = os.listdir(output_directory)
        next_experiment_id = None

        for experiment_id in range(len(experiments)):
            if not str(experiment_id) in existing_experiments:
                next_experiment_id = experiment_id
                break

        experiment_config = None

        if next_experiment_id is not None:
            experiment_output_dir = os.path.join(output_directory, str(next_experiment_id))
            os.makedirs(experiment_output_dir)

            # Make the output dir accessible on the config itself, so more things can be put there.
            self.experiment_output_dir = experiment_output_dir

            experiment_json = experiments[experiment_id]
            self._load_single_experiment(copy.deepcopy(experiment_json))
            self._write_json_config_file(experiment_json, experiment_output_dir)
            print("Starting job in location: {}".format(experiment_output_dir))

            experiment_config = self  # TODO: this is actually kind of confusing. Should just return a new object probably?

        return experiment_config
