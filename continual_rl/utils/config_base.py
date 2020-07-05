import os
import json
import copy
import time
import subprocess
from abc import ABC, abstractmethod


class UnknownExperimentConfigEntry(Exception):
    pass


class ConfigBase(ABC):
    """
    This is the base class for the experiment configuration loader.
    It will automatically load a JSON file that is a list of dicts. Each dict is assumed to be a
    separate experiment that gets parsed by the particular implementation of this class.
    We will get the next experiment that has not yet been started, and return it.
    """
    def __init__(self):
        self.experiment_output_dir = None  # Accessible output dir for the current run of the experiment

    @abstractmethod
    def _load_from_dict_internal(self, config_dict):
        """
        Load the parameters from the input dict object into the current object (self).
        Pop each parameter off so the caller of this method knows it was successfully consumed.
        """
        pass

    def load_from_dict(self, config_dict):
        """
        Load the parameters from the input dict object into the current object (self).
        Will assert UnknownExperimentConfigEntry if something unknown was found.
        """
        loaded_config = self._load_from_dict_internal(config_dict)

        if len(config_dict) > 0:
            raise UnknownExperimentConfigEntry("Dict still had elements after parsing: {}".format(config_dict.keys()))

        return loaded_config

    def set_experiment_output_dir(self, experiment_output_dir):
        self.experiment_output_dir = experiment_output_dir
