import pytest
import shutil
from pathlib import Path
from mock import patch
from continual_rl.utils.argparse_manager import ArgparseManager
from continual_rl.available_policies import PolicyStruct
from tests.utils.mocks.mock_policy.mock_policy import MockPolicy
from tests.utils.mocks.mock_policy.mock_policy_config import MockPolicyConfig


class TestArgparseManager(object):

    @pytest.fixture
    def cleanup_experiment(self, request):
        # Courtesy: https://stackoverflow.com/questions/44716237/pytest-passing-data-for-cleanup
        def cleanup():
            path_to_remove = request.node.experiment_output_dir
            print(f"Removing {path_to_remove}")
            shutil.rmtree(path_to_remove)

        request.addfinalizer(cleanup)

    @patch('continual_rl.utils.configuration_loader.get_available_policies')
    @patch('continual_rl.utils.configuration_loader.get_available_experiments')
    def test_command_line_parser_simple_success(self, get_available_experiments_mock, get_available_policies_mock,
                                                cleanup_experiment, request):
        """
        Argparser should successfully retrieve the correct policy and experiment from the provided args, and the
        experiment output directory should be successfully setup using the default output directory.
        """
        # Arrange
        # Setup our available policies and experiments so we're not using the real ones
        mock_policy = PolicyStruct(MockPolicy, MockPolicyConfig)
        mock_experiment = {"this is": "not important"}
        get_available_policies_mock.return_value = {"mock_policy": mock_policy}
        get_available_experiments_mock.return_value = {"mock_experiment": mock_experiment}

        args = ["--policy", "mock_policy", "--experiment", "mock_experiment", "--test_param", "some value"]

        # Act
        experiment, policy = ArgparseManager.parse(args)

        # For cleanup
        request.node.experiment_output_dir = policy._config.experiment_output_dir

        # Assert
        assert isinstance(policy, MockPolicy), "Policy not successfully retrieved"
        assert policy._config.test_param == "some value", "Policy config not successfully set"
        assert policy._config.test_param_2 == "also unfilled", "Policy config not default when expected"
        assert experiment is mock_experiment, "Experiment not successfully retrieved"
        assert Path(policy._config.experiment_output_dir).is_dir()
        assert Path(policy._config.experiment_output_dir, "experiment.json").is_file()

    @patch('continual_rl.utils.configuration_loader.get_available_policies')
    @patch('continual_rl.utils.configuration_loader.get_available_experiments')
    def test_config_file_simple_success(self, get_available_experiments_mock, get_available_policies_mock,
                                        cleanup_experiment, request):
        """
        Argparser should successfully retrieve the correct policy and experiment from the config file, and the
        experiment output directory should be successfully setup using the default output directory.
        """
        # Arrange
        # Setup our available policies and experiments so we're not using the real ones
        mock_policy = PolicyStruct(MockPolicy, MockPolicyConfig)
        mock_experiment = {"this is": "not important"}
        get_available_policies_mock.return_value = {"mock_policy": mock_policy}
        get_available_experiments_mock.return_value = {"mock_experiment": mock_experiment}

        config_file_path = Path(__file__).parent.absolute().joinpath("mocks", "mock_config.json")
        args = ["--config-file", f"{config_file_path}"]

        # Act
        experiment, policy = ArgparseManager.parse(args)

        # For cleanup
        request.node.experiment_output_dir = policy._config.experiment_output_dir

        # Assert
        assert isinstance(policy, MockPolicy), "Policy not successfully retrieved"
        assert policy._config.test_param == "some config value", "Policy config not successfully set"
        assert policy._config.test_param_2 == "also unfilled", "Policy config not default when expected"
        assert experiment is mock_experiment, "Experiment not successfully retrieved"
        assert "mock_config" in policy._config.experiment_output_dir, "Directory does not contain the config file name"
        assert Path(policy._config.experiment_output_dir).is_dir()
        assert Path(policy._config.experiment_output_dir, "experiment.json").is_file()
