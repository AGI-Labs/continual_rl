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
    def __init__(self, output_dir):
        self._output_dir = output_dir  # General output dir where all experiments currently being run live
        self.experiment_output_dir = None  # Accessible output dir for the current run of the experiment

    @abstractmethod
    def load_single_experiment_from_config(self, config_json):
        """
        Load the parameters from the input json object into the current object (self).
        Should assert UnknownExperimentConfigEntry if something unknown was found.
        """
        pass

    def set_experiment_output_dir(self, experiment_output_dir):
        self.experiment_output_dir = experiment_output_dir
