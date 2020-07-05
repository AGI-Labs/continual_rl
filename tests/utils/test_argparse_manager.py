import pytest
import shutil
from pathlib import Path
import continual_rl.utils.configuration_loader as configuration_loader
from continual_rl.utils.argparse_manager import ArgparseManager
from continual_rl.available_policies import PolicyStruct
from tests.utils.mocks.mock_policy.mock_policy import MockPolicy
from tests.utils.mocks.mock_policy.mock_policy_config import MockPolicyConfig


class TestArgparseManager(object):

    @pytest.fixture
    def setup_mocks(self, monkeypatch):
        def mock_get_available_policies(*args, **kwargs):
            mock_policy = PolicyStruct(MockPolicy, MockPolicyConfig)
            return {"mock_policy": mock_policy}

        def mock_get_available_experiments(*args, **kwargs):
            mock_experiment = {"this is": "some spec"}
            return {"mock_experiment": mock_experiment}

        monkeypatch.setattr(configuration_loader, "get_available_policies", mock_get_available_policies)
        monkeypatch.setattr(configuration_loader, "get_available_experiments", mock_get_available_experiments)

    @pytest.fixture
    def cleanup_experiment(self, request):
        # Courtesy: https://stackoverflow.com/questions/44716237/pytest-passing-data-for-cleanup
        def cleanup():
            path_to_remove = request.node.experiment_output_dir
            print(f"Removing {path_to_remove}")
            shutil.rmtree(path_to_remove)

        request.addfinalizer(cleanup)

    def test_command_line_parser_simple_success(self, setup_mocks, cleanup_experiment, request):
        """
        Argparser should successfully retrieve the correct policy and experiment from the provided args, and the
        experiment output directory should be successfully setup using the default output directory.
        """
        # Arrange
        args = ["--policy", "mock_policy", "--experiment", "mock_experiment", "--test_param", "some value"]

        # Act
        experiment, policy = ArgparseManager.parse(args)

        # For cleanup
        request.node.experiment_output_dir = policy._config.experiment_output_dir

        # Assert
        # Policy checks
        assert isinstance(policy, MockPolicy), "Policy not successfully retrieved"
        assert policy._config.test_param == "some value", "Policy config not successfully set"
        assert policy._config.test_param_2 == "also unfilled", "Policy config not default when expected"

        # Experiment checks
        assert experiment["this is"] == "some spec", "Experiment not successfully retrieved"

        # Output dir checks
        assert "mock_policy" in policy._config.experiment_output_dir, "Directory does not contain the policy name"
        assert "mock_experiment" in policy._config.experiment_output_dir, "Directory does not contain the experiment name"
        assert Path(policy._config.experiment_output_dir).is_dir()
        assert Path(policy._config.experiment_output_dir, "experiment.json").is_file()

    def test_config_file_simple_success(self, setup_mocks, cleanup_experiment, request):
        """
        Argparser should successfully retrieve the correct policy and experiment from the config file, and the
        experiment output directory should be successfully setup using the default output directory.
        """
        # Arrange
        config_file_path = Path(__file__).parent.absolute().joinpath("mocks", "mock_config.json")
        args = ["--config-file", f"{config_file_path}"]

        # Act
        experiment, policy = ArgparseManager.parse(args)

        # For cleanup
        request.node.experiment_output_dir = policy._config.experiment_output_dir

        # Assert
        # Policy checks
        assert isinstance(policy, MockPolicy), "Policy not successfully retrieved"
        assert policy._config.test_param == "some config value", "Policy config not successfully set"
        assert policy._config.test_param_2 == "also unfilled", "Policy config not default when expected"

        # Experiment checks
        assert experiment["this is"] == "some spec", "Experiment not successfully retrieved"

        # Output dir checks
        assert "mock_config" in policy._config.experiment_output_dir, "Directory does not contain the config file name"
        assert Path(policy._config.experiment_output_dir).is_dir()
        assert Path(policy._config.experiment_output_dir, "experiment.json").is_file()
