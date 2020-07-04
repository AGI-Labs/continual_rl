from mock import patch
from utils.argparse_manager import ArgparseManager, ArgumentMissingException
from available_policies import PolicyStruct
from utils.test.mocks.mock_policy.mock_policy import MockPolicy
from utils.test.mocks.mock_policy.mock_policy_config import MockPolicyConfig


class TestArgparseManager(object):

    @patch('utils.configuration_loader.get_available_policies')
    @patch('utils.configuration_loader.get_available_experiments')
    def test_command_line_parser_simple_success(self, get_available_experiments_mock, get_available_policies_mock):
        """
        Argparser should successfully retrieve the correct policy and experiment from the provided args.
        """
        # Arrange
        mock_policy = PolicyStruct(MockPolicy, MockPolicyConfig)
        mock_experiment = {"this is": "not important"}

        get_available_policies_mock.return_value = {"mock_policy": mock_policy}
        get_available_experiments_mock.return_value = {"mock_experiment": mock_experiment}

        args = ["--policy", "mock_policy", "--experiment", "mock_experiment", "--test_param", "some value"]

        # Act
        experiment, policy = ArgparseManager.parse(args)

        # Assert
        assert isinstance(policy, MockPolicy), "Policy not successfully retrieved"
        assert policy._config.test_param == "some value", "Policy config not successfully set"
        assert experiment is mock_experiment, "Experiment not successfully retrieved"

        # TODO: test output directory creation, and also clean it up
