import argparse
from continual_rl.utils.configuration_loader import ConfigurationLoader


class ArgumentMissingException(Exception):
    def __init__(self, error_str):
        super().__init__(error_str)


class ArgparseManager(object):
    """
    Handles processing the command line, and then calls the ConfigurationLoader to actually load the experiment
    and policy as appropriate.
    """

    def __init__(self):
        self.command_line_mode_parser = self._create_command_line_mode_parser()
        self.config_mode_parser = self._create_config_mode_parser()

    @classmethod
    def _create_command_line_mode_parser(cls):
        # All other arguments will be converted to a dictionary and used the same way as if it were a configuration
        command_line_parser = argparse.ArgumentParser()
        command_line_parser.add_argument("--output-dir", help="The output directory where this experiment's results"
                                                              "(logs and models) will be stored.",
                                         type=str, default="tmp")

        return command_line_parser

    @classmethod
    def _create_config_mode_parser(cls):
        """
        If the "config-file" mode is run, these are the arguments expected.
        Example: python main.py --config-file path/to/my_config.json --output-dir path/to/output
        """
        config_parser = argparse.ArgumentParser()
        config_parser.add_argument('--config-file', type=str, help='The full path to the JSON file containing the '
                                                                   'experiment configs.')
        config_parser.add_argument("--output-dir", help="The output directory where logs and models for all experiments"
                                                        " generated by this config file are stored.",
                                   type=str, default="tmp")
        return config_parser

    @classmethod
    def parse(cls, raw_args):
        argparser = ArgparseManager()

        # If we successfully parse a config_file, enter config-mode, otherwise default to command-line mode
        args, extras = argparser.config_mode_parser.parse_known_args(raw_args)

        if args.config_file is not None:
            assert len(extras) == 0, f"Unknown arguments found: {extras}"
            print(f"Entering config mode using file {args.config_file} and output directory {args.output_dir}")

            experiment, policy = ConfigurationLoader.load_next_experiment_from_config(args.output_dir,
                                                                                      args.config_file)
        else:
            args, extras = argparser.command_line_mode_parser.parse_known_args(raw_args)

            # Extras is a list in the form ["--arg1", "val1", "--arg2", "val2"]. Convert it to a dictionary
            raw_experiment = {extras[i].replace('--', ''): extras[i + 1] for i in range(0, len(extras), 2)}

            if "experiment" not in raw_experiment:
                raise ArgumentMissingException("--experiment required in command-line mode")

            if "policy" not in raw_experiment:
                raise ArgumentMissingException("--policy required in command-line mode")

            # load_next_experiment is expecting a list of experiment configs, so put our experiment in a list
            experiment, policy = ConfigurationLoader.load_next_experiment_from_dicts(args.output_dir,
                                                                                     [raw_experiment])

        return experiment, policy
