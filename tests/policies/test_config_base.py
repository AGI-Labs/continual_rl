import pytest
from continual_rl.policies.config_base import ConfigBase, UnknownExperimentConfigEntry, MismatchTypeException


class MockPolicyConfig(ConfigBase):

    def __init__(self):
        super().__init__()
        self.test_param = 1984
        self.test_param_2 = "Brave New World"
        self.foo = 42.314
        self.bar = None
        self.some_bool = True
        self.some_array = [2, 2]
        self.some_array_2 = [2, 2]

    def _load_from_dict_internal(self, config_dict):
        return self._auto_load_class_parameters(config_dict)


class TestConfigBase(object):

    def test_auto_load_class_parameters_simple_success(self):
        """
        ConfigBase provides an auto-loading feature, so not every class needs to pop individually.
        This tests that functionality, specifically a simple case of it.
        """
        # Arrange
        config = MockPolicyConfig()
        config_dict = {"test_param_2": "Fahrenheit 451"}

        # Act
        loaded_config = config.load_from_dict(config_dict)

        # Assert
        assert loaded_config.test_param == 1984
        assert loaded_config.test_param_2 == "Fahrenheit 451"
        assert loaded_config.foo == 42.314
        assert loaded_config.bar is None

    def test_auto_load_class_parameters_invalid_param_failure(self):
        """
        Tests that an invalid param fails
        """
        # Arrange
        config = MockPolicyConfig()
        config_dict = {"invalid_param": "Fahrenheit 451"}

        # Act & Assert
        with pytest.raises(UnknownExperimentConfigEntry):
            config.load_from_dict(config_dict)

    def test_auto_load_class_parameters_type_mismatch_failure(self):
        """
        Tests that a parameter with a type that is incompatible with the default fails
        """
        # Arrange
        config = MockPolicyConfig()
        config_dict = {"test_param": "Fahrenheit 451"}

        # Act & Assert
        with pytest.raises(MismatchTypeException):
            config.load_from_dict(config_dict)

    def test_auto_load_class_parameters_str_to_bool(self):
        """
        bool("false") returns True, so special logic is required.
        """
        # Arrange
        config = MockPolicyConfig()
        config_dict = {"some_bool": "false"}  # For instance from command line

        # Act & Assert
        config.load_from_dict(config_dict)

        # Assert
        assert not config.some_bool, "Bool parsed incorrectly - expected False, but returned True"

    def test_auto_load_class_parameters_arrays(self):
        """
        bool("false") returns True, so special logic is required.
        """
        # Arrange
        config = MockPolicyConfig()
        config_dict = {"some_array": [20, 4, (2, 1)]}

        # Act
        config.load_from_dict(config_dict)

        # Assert
        assert config.some_array == [20, 4, (2, 1)], "Array parsed incorrectly (simple mode)"

    def test_auto_load_class_parameters_array_from_str_asserts(self):
        """
        bool("false") returns True, so special logic is required.
        """
        # Arrange
        config = MockPolicyConfig()
        config_dict = {"some_array_2": "[20, 4, (2, 1)]"}

        # Act & Assert
        with pytest.raises(MismatchTypeException):
            config.load_from_dict(config_dict)
