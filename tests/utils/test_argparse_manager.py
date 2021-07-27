import pytest
import shutil
from pathlib import Path
import re
import json
from json import JSONDecodeError
import continual_rl.utils.argparse_manager as argparse_manager
from continual_rl.utils.configuration_loader import ExperimentNotFoundException, PolicyNotFoundException, IllFormedConfig
from continual_rl.utils.argparse_manager import ArgparseManager, ArgumentMissingException
from continual_rl.available_policies import PolicyStruct
from continual_rl.policies.config_base import UnknownExperimentConfigEntry
from continual_rl.experiments.experiment import Experiment
from tests.common_mocks.mock_policy.mock_policy import MockPolicy
from tests.common_mocks.mock_policy.mock_policy_config import MockPolicyConfig


class TestArgparseManager(object):
    """
    These are not exactly unit tests, because most of them are implicitly also testing ConfigurationLoader.
    This is intended - we're testing that the full flow of loading is working.
    """

    @pytest.fixture
    def setup_mocks(self, set_tmp_directory, monkeypatch):
        # First param in the lambda is "self" because it's an instance method
        monkeypatch.setattr(Experiment, "_get_action_spaces", lambda _, x: {0: 5, 1: 3})
        monkeypatch.setattr(Experiment, "_get_common_attribute", lambda _, x: 4)

        def mock_get_available_policies(*args, **kwargs):
            mock_policy = PolicyStruct(MockPolicy, MockPolicyConfig)
            return {"mock_policy": mock_policy}

        def mock_get_available_experiments(*args, **kwargs):
            mock_experiment = Experiment(tasks=[])
            return {"mock_experiment": mock_experiment}

        monkeypatch.setattr(argparse_manager, "get_available_policies", mock_get_available_policies)
        monkeypatch.setattr(argparse_manager, "get_available_experiments", mock_get_available_experiments)

    @classmethod
    def _get_file_matching_regex(cls, directory, regex):
        pattern = re.compile(regex)
        file_found = None

        for file_path in Path(directory).iterdir():
            if pattern.match(str(file_path.name)):
                file_found = file_path.name
                break
        return file_found

    def test_command_line_parser_simple_success(self, setup_mocks, cleanup_experiment, request):
        """
        Argparser should successfully retrieve the correct policy and experiment from the provided args, and the
        experiment output directory should be successfully setup using the default output directory.
        """
        # Arrange
        args = ["--policy", "mock_policy", "--experiment", "mock_experiment", "--test_param", "some value",
                "--output-dir", request.node.experiment_output_dir]

        # Act
        experiment, policy = ArgparseManager.parse(args)

        # Assert
        # Policy checks
        assert isinstance(policy, MockPolicy), "Policy not successfully retrieved"
        assert policy._config.test_param == "some value", "Policy config not successfully set"
        assert policy._config.test_param_2 == "also unfilled", "Policy config not default when expected"

        # Experiment checks
        # Sanity checks based on one of the parameters set by the mock
        assert isinstance(experiment, Experiment)
        assert experiment.observation_space == 4, "Experiment not successfully retrieved"

        # Output dir checks
        assert "mock_policy" in policy._config.output_dir, "Directory does not contain the policy name"
        assert "mock_experiment" in policy._config.output_dir, "Directory does not contain the experiment name"
        assert Path(policy._config.output_dir).is_dir()
        assert self._get_file_matching_regex(policy._config.output_dir, "experiment.*\.json") is not None

    def test_config_file_simple_success(self, setup_mocks, cleanup_experiment, request):
        """
        Argparser should successfully retrieve the correct policy and experiment from the config file, and the
        experiment output directory should be successfully setup using the default output directory.
        """
        # Arrange
        config_file_path = Path(__file__).parent.absolute().joinpath("mocks", "mock_config.json")
        args = ["--config-file", f"{config_file_path}", "--output-dir", request.node.experiment_output_dir]

        # Act
        experiment, policy = ArgparseManager.parse(args)

        # Assert
        # Policy checks
        assert isinstance(policy, MockPolicy), "Policy not successfully retrieved"
        assert policy._config.test_param == "some config value", "Policy config not successfully set"
        assert policy._config.test_param_2 == "also unfilled", "Policy config not default when expected"

        # Experiment checks
        # Sanity checks based on one of the parameters set by the mock
        assert isinstance(experiment, Experiment)
        assert experiment.observation_space == 4, "Experiment not successfully retrieved"

        # Output dir checks
        assert "mock_config" in policy._config.output_dir, "Directory does not contain the config file name"
        assert Path(policy._config.output_dir).is_dir()
        assert self._get_file_matching_regex(policy._config.output_dir, "experiment.*\.json") is not None

    def test_command_line_parser_no_experiment(self, setup_mocks, cleanup_experiment, request):
        """
        Argparser should fail due to missing "experiment" argument
        """
        # Arrange
        args = ["--policy", "mock_policy", "--test_param", "some value",
                "--output-dir", request.node.experiment_output_dir]

        # Act & Assert
        with pytest.raises(ArgumentMissingException):
            ArgparseManager.parse(args)

    def test_command_line_parser_no_policy(self, setup_mocks, cleanup_experiment, request):
        """
        Argparser should fail due to missing "policy" argument
        """
        # Arrange
        args = ["--experiment", "mock_experiment", "--test_param", "some value",
                "--output-dir", request.node.experiment_output_dir]

        # Act & Assert
        with pytest.raises(ArgumentMissingException):
            ArgparseManager.parse(args)

    def test_command_line_missing_policy(self, setup_mocks, cleanup_experiment, request):
        """
        Argparser should fail due to attempting to retrieve a policy that does not exist.
        """
        # Arrange
        args = ["--policy", "missing_policy", "--experiment", "mock_experiment", "--test_param", "some value",
                "--output-dir", request.node.experiment_output_dir]

        # Act & assert
        with pytest.raises(PolicyNotFoundException):
            ArgparseManager.parse(args)

    def test_command_line_missing_experiment(self, setup_mocks, cleanup_experiment, request):
        """
        Argparser should fail due to attempting to retrieve an experiment that does not exist.
        """
        # Arrange
        args = ["--policy", "mock_policy", "--experiment", "missing_experiment", "--test_param", "some value",
                "--output-dir", request.node.experiment_output_dir]

        # Act & assert
        with pytest.raises(ExperimentNotFoundException):
            ArgparseManager.parse(args)

    def test_command_line_invalid_argument(self, setup_mocks, cleanup_experiment, request):
        """
        Argparser should fail due to having a parameter that is unknown.
        """
        # Arrange
        args = ["--policy", "mock_policy", "--experiment", "mock_experiment", "--unknown_param", "some value",
                "--output-dir", request.node.experiment_output_dir]

        # Act & assert
        with pytest.raises(UnknownExperimentConfigEntry):
            ArgparseManager.parse(args)

    def test_config_file_invalid_argument(self, setup_mocks, cleanup_experiment, request):
        """
        Argparser should fail due to having a parameter that is unknown.
        """
        # Arrange
        config_file_path = Path(__file__).parent.absolute().joinpath("mocks", "mock_config_invalid_param.json")
        args = ["--config-file", f"{config_file_path}", "--output-dir", request.node.experiment_output_dir]

        # Act & assert
        with pytest.raises(UnknownExperimentConfigEntry):
            ArgparseManager.parse(args)

    def test_config_file_multi_call_simple_success(self, setup_mocks, cleanup_experiment, request):
        """
        When we've run out of configs to run, we should return Nones
        """
        # Arrange
        config_file_path = Path(__file__).parent.absolute().joinpath("mocks", "mock_config.json")
        args = ["--config-file", f"{config_file_path}", "--output-dir", request.node.experiment_output_dir]

        # Act
        _, policy_to_clean = ArgparseManager.parse(args)
        experiment, policy = ArgparseManager.parse(args)

        # Assert
        assert experiment is None, "Though we are finished with experiments, an experiment was returned"
        assert policy is None, "Though we are finished with experiments, a policy was returned"

    def test_multiple_config_file_simple_success(self, setup_mocks, cleanup_experiment, request):
        """
        Argparser should successfully retrieve both experiment configurations.
        """
        # Arrange
        config_file_path = Path(__file__).parent.absolute().joinpath("mocks", "mock_config_multi.json")
        args = ["--config-file", f"{config_file_path}", "--output-dir", request.node.experiment_output_dir]

        # Act
        experiment_0, policy_0 = ArgparseManager.parse(args)
        experiment_1, policy_1 = ArgparseManager.parse(args)

        # Assert
        # Policy checks
        assert isinstance(policy_0, MockPolicy), "Policy 0 not successfully retrieved"
        assert policy_0._config.test_param == "first", "Policy 0 config not successfully set"
        assert isinstance(policy_1, MockPolicy), "Policy 1 not successfully retrieved"
        assert policy_1._config.test_param == "second", "Policy 1 config not successfully set"

        # Experiment checks
        # Sanity checks based on one of the parameters set by the mock
        assert isinstance(experiment_0, Experiment)
        assert isinstance(experiment_1, Experiment)
        assert experiment_0.observation_space == 4, "Experiment not successfully retrieved"
        assert experiment_1.observation_space == 4, "Experiment not successfully retrieved"

        # Output dir checks
        assert "mock_config" in policy_0._config.output_dir, "Output path does not contain the config file name"
        assert "0" in policy_0._config.output_dir[-1:], "Output path does not contain the experiment id"
        assert "mock_config" in policy_1._config.output_dir, "Output path does not contain the config file name"
        assert "1" in policy_1._config.output_dir[-1:], "Output path does not contain the experiment id"
        assert Path(policy_0._config.output_dir).is_dir()
        assert Path(policy_1._config.output_dir).is_dir()
        assert self._get_file_matching_regex(policy_0._config.output_dir, "experiment.*\.json") is not None
        assert self._get_file_matching_regex(policy_1._config.output_dir, "experiment.*\.json") is not None

    def test_multiple_config_file_experiment_removed(self, setup_mocks, cleanup_experiment, request):
        """
        Argparser should successfully retrieve the first experiment after it's been removed.
        This experiment retrieves experiment 0, then experiment 1, then removes experiment 0, and makes sure
        that the next time an experiment is grabbed, it's that first one again
        """
        # Arrange
        config_file_path = Path(__file__).parent.absolute().joinpath("mocks", "mock_config_multi.json")
        args = ["--config-file", f"{config_file_path}", "--output-dir", request.node.experiment_output_dir]

        # Act
        experiment_0, policy_0 = ArgparseManager.parse(args)

        # Grab the second experiment
        _, _ = ArgparseManager.parse(args)

        # Remove the first experiment retrieved, so we expect the next one retrieved to be the first one again
        policy_0_path = f"{Path(policy_0._config.output_dir).parent}"
        shutil.rmtree(policy_0_path)

        experiment_1, policy_1 = ArgparseManager.parse(args)

        # Assert
        # Policy checks
        assert isinstance(policy_0, MockPolicy), "Policy 0 not successfully retrieved"
        assert policy_0._config.test_param == "first", "Policy 0 config not successfully set"
        assert isinstance(policy_1, MockPolicy), "Policy 1 not successfully retrieved"
        assert policy_1._config.test_param == "first", "Policy 1 config not successfully set"

        # Experiment checks
        # Sanity checks based on one of the parameters set by the mock
        assert isinstance(experiment_0, Experiment)
        assert isinstance(experiment_1, Experiment)
        assert experiment_0.observation_space == 4, "Experiment not successfully retrieved"
        assert experiment_1.observation_space == 4, "Experiment not successfully retrieved"

        # Output dir checks
        assert "mock_config" in policy_0._config.output_dir, "Output path does not contain the config file name"
        assert "0" in policy_0._config.output_dir[-1:], "Output path does not contain the experiment id"
        assert "mock_config" in policy_1._config.output_dir, "Output path does not contain the config file name"
        assert "0" in policy_1._config.output_dir[-1:], "Output path does not contain the experiment id"

    def test_config_file_ill_formed_list_check(self, setup_mocks, cleanup_experiment, request):
        """
        Argparser should fail informatively if the config file is ill-formed. In this case, it is not a list of
        experiments, just a single experiment.
        """
        # Arrange
        config_file_path = Path(__file__).parent.absolute().joinpath("mocks", "mock_config_ill_formed_0.json")
        args = ["--config-file", f"{config_file_path}", "--output-dir", request.node.experiment_output_dir]

        # Act & Assert
        with pytest.raises(IllFormedConfig):
            ArgparseManager.parse(args)

    def test_config_file_ill_formed_dict_check(self, setup_mocks, cleanup_experiment, request):
        """
        Argparser should fail informatively if the config file is ill-formed. In this case, experiments are not
        dictionaries.
        """
        # Arrange
        config_file_path = Path(__file__).parent.absolute().joinpath("mocks", "mock_config_ill_formed_1.json")
        args = ["--config-file", f"{config_file_path}", "--output-dir", request.node.experiment_output_dir]

        # Act & Assert
        with pytest.raises(IllFormedConfig):
            ArgparseManager.parse(args)

    def test_config_file_ill_formed_not_json(self, setup_mocks, cleanup_experiment, request):
        """
        Argparser should fail informatively if the config file is ill-formed. In this case, the file is not even json.
        """
        # Arrange
        config_file_path = Path(__file__).parent.absolute().joinpath("mocks", "mock_config_ill_formed_2.json")
        args = ["--config-file", f"{config_file_path}", "--output-dir", request.node.experiment_output_dir]

        # Act & Assert
        with pytest.raises(JSONDecodeError):
            ArgparseManager.parse(args)

    def test_command_line_parser_non_default_output(self, setup_mocks, cleanup_experiment, request):
        """
        The experiments should be initialized in the custom output directory.
        This is a little moot now that I'm always setting a custom output dir, but leaving it for a second test.
        """
        # Arrange
        output_dir = "this_dir_will_be_deleted"
        args = ["--policy", "mock_policy", "--experiment", "mock_experiment", "--test_param", "some value",
                "--output-dir", output_dir]

        # Act
        experiment, policy = ArgparseManager.parse(args)

        # For cleanup
        request.node.experiment_output_dir = output_dir

        # Assert
        # Output dir checks
        assert output_dir in policy._config.output_dir, "Output directory not created in the correct location"
        assert "mock_policy" in policy._config.output_dir, "Directory does not contain the policy name"
        assert "mock_experiment" in policy._config.output_dir, "Directory does not contain the experiment name"
        assert Path(policy._config.output_dir).is_dir()
        assert self._get_file_matching_regex(policy._config.output_dir, "experiment.*\.json") is not None

    def test_config_file_non_default_output(self, setup_mocks, cleanup_experiment, request):
        """
        Argparser should successfully retrieve the correct policy and experiment from the config file, and the
        experiment output directory should be successfully setup using the default output directory.
        """
        # Arrange
        output_dir = "this_dir_will_be_deleted_config"
        config_file_path = Path(__file__).parent.absolute().joinpath("mocks", "mock_config.json")
        args = ["--config-file", f"{config_file_path}", "--output-dir", output_dir]

        # Act
        experiment, policy = ArgparseManager.parse(args)

        # For cleanup. In this case we want to cleanup the parent (top level config folder)
        request.node.experiment_output_dir = output_dir

        # Assert
        # Output dir checks
        assert output_dir in policy._config.output_dir, "Output directory not created in the correct location"
        assert Path(policy._config.output_dir).is_dir()
        assert self._get_file_matching_regex(policy._config.output_dir, "experiment.*\.json") is not None

    def test_config_file_experiment_json(self, setup_mocks, cleanup_experiment, request):
        """
        Argparser should successfully retrieve the correct policy and experiment from the config file, and the
        experiment output directory should be successfully setup using the default output directory.
        """
        # Arrange
        config_file_path = Path(__file__).parent.absolute().joinpath("mocks", "mock_config.json")
        args = ["--config-file", f"{config_file_path}", "--output-dir", request.node.experiment_output_dir]

        # Act
        experiment, policy = ArgparseManager.parse(args)

        # Assert
        # Read in our experiment metadata file, so we can verify it
        meta_data_file_name = self._get_file_matching_regex(policy._config.output_dir, "experiment.*\.json")
        meta_data_path = Path(policy._config.output_dir, meta_data_file_name)
        with open(meta_data_path) as json_file:
            json_raw = json_file.read()
            saved_meta_data = json.loads(json_raw)

        # Output dir checks
        assert saved_meta_data["experiment"] == "mock_experiment", "Meta data experiment not saved properly"
        assert saved_meta_data["policy"] == "mock_policy", "Meta data policy not saved properly"
        assert saved_meta_data["test_param"] == "some config value", "Meta data custom param not saved properly"
        assert "continual_rl_commit" in saved_meta_data, "Meta data does not contain commit hash"
        assert "timestamp" in saved_meta_data, "Meta data does not contain timestamp"

    def test_command_line_experiment_json(self, setup_mocks, cleanup_experiment, request):
        """
        Argparser should successfully retrieve the correct policy and experiment from the config file, and the
        experiment output directory should be successfully setup using the default output directory.
        """
        # Arrange
        args = ["--policy", "mock_policy", "--experiment", "mock_experiment", "--test_param", "some value",
                "--output-dir", request.node.experiment_output_dir]

        # Act
        experiment, policy = ArgparseManager.parse(args)

        # Assert
        # Read in our experiment metadata file, so we can verify it
        meta_data_file_name = self._get_file_matching_regex(policy._config.output_dir, "experiment.*\.json")
        meta_data_path = Path(policy._config.output_dir, meta_data_file_name)
        with open(meta_data_path) as json_file:
            json_raw = json_file.read()
            saved_meta_data = json.loads(json_raw)

        # Output dir checks
        assert saved_meta_data["experiment"] == "mock_experiment", "Meta data experiment not saved properly"
        assert saved_meta_data["policy"] == "mock_policy", "Meta data policy not saved properly"
        assert saved_meta_data["test_param"] == "some value", "Meta data custom param not saved properly"
        assert "continual_rl_commit" in saved_meta_data, "Meta data does not contain commit hash"
        assert "timestamp" in saved_meta_data, "Meta data does not contain timestamp"
