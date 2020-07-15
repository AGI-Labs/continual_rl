import os
import subprocess
import copy
import json
import datetime


class ExperimentNotFoundException(Exception):
    def __init__(self, error_msg):
        super().__init__(error_msg)


class PolicyNotFoundException(Exception):
    def __init__(self, error_msg):
        super().__init__(error_msg)


class IllFormedConfig(Exception):
    def __init__(self, error_msg):
        super().__init__(error_msg)


class ConfigurationLoader(object):
    """
    Loads a configuration from a raw dictionary into the appropriate experiment spec and policy config objects.
    Also sets up the output directory where the new experiment will be stored.
    """

    def __init__(self, available_policies, available_experiments):
        self._available_policies = available_policies
        self._available_experiments = available_experiments

    def _get_policy_and_experiment_from_raw_config(self, raw_config, experiment_output_dir):
        """
        The config dictionary will tell us which policy to use, which will allow us to populate the correct config file.
        """
        # Extract the spec of the experiment we will be running
        experiment_id = raw_config.pop("experiment")

        if experiment_id not in self._available_experiments:
            raise ExperimentNotFoundException(f"Experiment {experiment_id} not found in available experiments.")

        experiment = self._available_experiments[experiment_id]

        # Extract the configuration of the policy we will be running
        policy_id = raw_config.pop("policy")

        if policy_id not in self._available_policies:
            raise PolicyNotFoundException(f"Policy {policy_id} not found in available experiments.")

        policy_class = self._available_policies[policy_id].policy
        policy_config_class = self._available_policies[policy_id].config
        policy_config = policy_config_class().load_from_dict(raw_config)

        # Pass the config to the policy - assumes the initialization signature of all policies is simply
        # PolicyType(PolicyConfig)
        policy = policy_class(policy_config, experiment.observation_size, experiment.action_sizes)

        return experiment, policy

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
    def _write_json_log_file(cls, experiment_json, output_path):
        experiment_json = copy.deepcopy(experiment_json)
        experiment_json["commit"] = str(cls._get_script_dir_commit_hash())
        experiment_json["timestamp"] = str(datetime.datetime.utcnow())

        output_file_path = os.path.join(output_path, "experiment.json")

        with open(output_file_path, "w") as output_file:
            output_file.write(json.dumps(experiment_json))

    def load_next_experiment_from_config(self, output_dir, config_path):
        """
        Reads the configuration dictionary from the config_path, and loads the next entry to run.
        Returns None if there is nothing further to load.

        This will throw a JSONDecodeError if the file is not valid JSON. May also raise IllFormedConfig,
        ExperimentNotFoundException, PolicyNotFoundException.
        """
        # Instead of dumping directly into the output directory, we'll make a folder with the same name as the
        # experiment file. This allows for multiple experiment sets
        json_experiment_name = os.path.basename(os.path.splitext(config_path)[0])
        output_directory = os.path.join(output_dir, json_experiment_name)

        with open(config_path) as json_file:
            json_raw = json_file.read()
            experiments = json.loads(json_raw)

        return self.load_next_experiment_from_dicts(output_directory, experiments, subdirectory_from_timestamp=False)

    def load_next_experiment_from_dicts(self, experiment_base_directory, experiments, subdirectory_from_timestamp=True):
        """
        Given a list of experiments (i.e. a list of dictionaries), load the next one. Its results will be saved in
        experiment_output_directory.

        If subdirectory_from_timestamp is true, we will create a new output directory regardless, according to
        output_dir/<policy>_<experiment>_<timestamp>.
        Otherwise we will create subdirectories, one for each index of the list: output_dir/0, output_dir/1, etc,
        with one per separate configuration entry.

        Each experiment configuration dictionary must have a "policy" entry and an "experiment" entry, at minimum.

        May raise IllFormedConfig, ExperimentNotFoundException, PolicyNotFoundException.
        """
        if not isinstance(experiments, list):
            raise IllFormedConfig("Configuration is expected to be a list of dictionaries. "
                                  "The object found is not a list.")

        if subdirectory_from_timestamp:
            assert len(experiments) == 1, "Multiple experiments available, but exactly one was expected."

            # Run this experiment no matter what, in a folder based on the timestamp
            next_experiment_id = 0
            experiment = experiments[next_experiment_id]

            # Colons are disallowed in Windows, so format as 'Jul_14_2020_06.27.22' (Month day year hour min sec)
            timestamp = datetime.datetime.now().strftime("%b_%d_%Y_%H.%M.%S")
            output_name = f"{experiment['policy']}_{experiment['experiment']}_{timestamp}"
            experiment_output_dir = os.path.join(experiment_base_directory, output_name)
        else:
            # Load up the first experiment we haven't yet started
            # Start by grabbing all the folders (representing old experiments) that currently exist
            # These will be, '0', '1', '2', '3', etc.
            if os.path.exists(experiment_base_directory):
                existing_experiments = os.listdir(experiment_base_directory)
            else:
                existing_experiments = []
            next_experiment_id = None

            # Find the first experiment (ie numbered folder) that doesn't yet exist
            # We do it this way so that if folders '0' and '2' exist, we will run '1' now.
            for experiment_id in range(len(experiments)):
                if not str(experiment_id) in existing_experiments:
                    next_experiment_id = experiment_id
                    break
            experiment_output_dir = os.path.join(experiment_base_directory, str(next_experiment_id))

        experiment = None
        policy = None

        # Inflate the configuration from the raw json
        if next_experiment_id is not None:
            experiment_json = experiments[next_experiment_id]

            # We don't create the experiment folder until we've verified everything is good to go, which we
            # don't know until after we've popped everything off the experiment json.
            # So duplicate it, so we have the original, before modification.
            experiment_json_clone = copy.deepcopy(experiment_json)

            if not isinstance(experiment_json, dict):
                raise IllFormedConfig("The configuration for an experiment should be a dictionary.")

            experiment, policy = self._get_policy_and_experiment_from_raw_config(
                raw_config=experiment_json, experiment_output_dir=experiment_output_dir)

            # Finally, if we've found an experiment to start, create its output directory and
            # log some metadata information into an "experiments.json" file in the output directory
            os.makedirs(experiment_output_dir)
            self._write_json_log_file(experiment_json_clone, experiment_output_dir)

            # Set the directories after they've been created
            experiment.set_output_dir(experiment_output_dir)
            policy.set_output_dir(experiment_output_dir)

            print("Starting job in location: {}".format(experiment_output_dir))

        return experiment, policy
