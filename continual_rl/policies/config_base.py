import distutils.util
from abc import ABC, abstractmethod
from continual_rl.utils.common_exceptions import OutputDirectoryNotSetException


class UnknownExperimentConfigEntry(Exception):
    pass


class MismatchTypeException(Exception):
    pass


class ConfigBase(ABC):
    """
    This is the base class for the experiment configuration loader.
    It will automatically load a JSON file that is a list of dicts. Each dict is assumed to be a
    separate experiment that gets parsed by the particular implementation of this class.
    We will get the next experiment that has not yet been started, and return it.
    """
    def __init__(self):
        self._output_dir = None  # Output dir for the current run of the experiment, accessible as self.output_dir
        self.timesteps_per_save = 3e4

    def set_output_dir(self, set_output_dir):
        self._output_dir = set_output_dir

    @property
    def output_dir(self):
        if self._output_dir is None:
            raise OutputDirectoryNotSetException("Config output directory not set. Call set_output_dir.")
        return self._output_dir

    def _auto_load_class_parameters(self, config_dict):
        """
        This is a helper function that automatically grabs all parameters in this class from the configuration
        dictionary, using their exact names, if they are there.
        It attempts to maintain the type used in the default, but will be unable to do so if the default is None,
        and it will be up to the caller to cast to the correct type as appropriate.

        It is best-effort, and if complex parsing is desired, better to do it manually (or at least check).
        """
        for key, value in self.__dict__.items():
            # Get the class of the default (e.g. int) and cast to it (if not None)
            default_val = self.__dict__[key]
            dict_val = config_dict.pop(key, value)

            # bool("false") returns True, unfortunately. So it requires a bit more fancy logic.
            if isinstance(default_val, bool) and isinstance(dict_val, str):
                self.__dict__[key] = bool(distutils.util.strtobool(dict_val))
            elif isinstance(default_val, list) and isinstance(dict_val, str):
                raise MismatchTypeException("Parsing lists from string is not currently supported, and will do unexpected things.")
            else:
                type_to_cast_to = type(default_val) if default_val is not None else lambda x: x

                try:
                    self.__dict__[key] = type_to_cast_to(dict_val)
                except ValueError:
                    raise MismatchTypeException(f"Config expected type {type_to_cast_to} but dictionary had type {type(dict_val)}")

        return self

    @abstractmethod
    def _load_from_dict_internal(self, config_dict):
        """
        Load the parameters from the input dict object into the current object (self).
        Pop each parameter off so the caller of this method knows it was successfully consumed.

        Consider using _auto_load_class_parameters if the desired mapping is simple (config param is the same in the
        json and in the class).

        Should return the loaded Config object.
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
