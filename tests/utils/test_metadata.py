"""Unit tests for the metadata module.

This test module validates the functionality of metadata classes, particularly
ClassifierSettings and its handling of None values for optional parameters.
"""

import pytest

from jabs_postprocess.utils.metadata import (
    ClassifierSettings,
    FeatureSettings,
    DEFAULT_INTERPOLATE,
    DEFAULT_STITCH,
    DEFAULT_MIN_BOUT,
)


class TestClassifierSettingsInitialization:
    """Test ClassifierSettings initialization with various parameter combinations."""

    def test_all_explicit_values(self):
        """Test ClassifierSettings with all parameters explicitly set."""
        settings = ClassifierSettings(
            behavior="grooming",
            interpolate=10,
            stitch=15,
            min_bout=20,
        )

        assert settings.behavior == "grooming"
        assert settings.interpolate == 10
        assert settings.stitch == 15
        assert settings.min_bout == 20

    def test_all_none_values_use_defaults(self):
        """Test that None values default to constants.

        This is the core fix for issue #45 - None values should be converted
        to their default constants rather than remaining None.
        """
        settings = ClassifierSettings(
            behavior="grooming",
            interpolate=None,
            stitch=None,
            min_bout=None,
        )

        assert settings.behavior == "grooming"
        assert settings.interpolate == DEFAULT_INTERPOLATE
        assert settings.stitch == DEFAULT_STITCH
        assert settings.min_bout == DEFAULT_MIN_BOUT

    def test_mixed_none_and_explicit_values(self):
        """Test ClassifierSettings with some None and some explicit values."""
        settings = ClassifierSettings(
            behavior="walking",
            interpolate=10,
            stitch=None,
            min_bout=25,
        )

        assert settings.behavior == "walking"
        assert settings.interpolate == 10
        assert settings.stitch == DEFAULT_STITCH  # Should use default
        assert settings.min_bout == 25

    def test_zero_values_are_preserved(self):
        """Test that explicit zero values are not treated as None."""
        settings = ClassifierSettings(
            behavior="feeding",
            interpolate=0,
            stitch=0,
            min_bout=0,
        )

        assert settings.interpolate == 0
        assert settings.stitch == 0
        assert settings.min_bout == 0

    @pytest.mark.parametrize(
        "interpolate,stitch,min_bout",
        [
            (None, None, None),
            (5, None, None),
            (None, 10, None),
            (None, None, 15),
            (5, 10, None),
            (5, None, 15),
            (None, 10, 15),
            (5, 10, 15),
        ],
    )
    def test_all_parameter_combinations(self, interpolate, stitch, min_bout):
        """Test all combinations of None and explicit values."""
        settings = ClassifierSettings(
            behavior="test",
            interpolate=interpolate,
            stitch=stitch,
            min_bout=min_bout,
        )

        # Verify each parameter is either the provided value or the default
        expected_interpolate = (
            interpolate if interpolate is not None else DEFAULT_INTERPOLATE
        )
        expected_stitch = stitch if stitch is not None else DEFAULT_STITCH
        expected_min_bout = min_bout if min_bout is not None else DEFAULT_MIN_BOUT

        assert settings.interpolate == expected_interpolate
        assert settings.stitch == expected_stitch
        assert settings.min_bout == expected_min_bout


class TestClassifierSettingsComparison:
    """Test that ClassifierSettings values work in comparison operations.

    These tests verify the fix for issue #45 - the bug occurred when
    None values were compared with integers in filter_by_settings().
    """

    def test_comparison_with_defaults_from_none(self):
        """Test that default values from None can be compared with integers."""
        settings = ClassifierSettings(
            behavior="test",
            interpolate=None,
            stitch=None,
            min_bout=None,
        )

        # These comparisons would raise TypeError if values were None
        assert settings.interpolate > 0
        assert settings.stitch > 0
        assert settings.min_bout > 0
        assert settings.interpolate >= 0
        assert settings.stitch <= 100
        assert settings.min_bout == DEFAULT_MIN_BOUT

    def test_comparison_with_explicit_values(self):
        """Test that explicit values work in comparisons."""
        settings = ClassifierSettings(
            behavior="test",
            interpolate=10,
            stitch=15,
            min_bout=20,
        )

        assert settings.interpolate > 5
        assert settings.stitch > 10
        assert settings.min_bout > 15
        assert settings.interpolate == 10
        assert settings.stitch == 15
        assert settings.min_bout == 20

    def test_comparison_with_zero_values(self):
        """Test that zero values work in comparisons."""
        settings = ClassifierSettings(
            behavior="test",
            interpolate=0,
            stitch=0,
            min_bout=0,
        )

        assert settings.interpolate >= 0
        assert settings.stitch >= 0
        assert settings.min_bout >= 0
        assert not (settings.interpolate > 0)
        assert not (settings.stitch > 0)
        assert not (settings.min_bout > 0)


class TestClassifierSettingsProperties:
    """Test ClassifierSettings property accessors."""

    def test_behavior_property(self):
        """Test that behavior property returns the correct value."""
        settings = ClassifierSettings("grooming", None, None, None)
        assert settings.behavior == "grooming"

    def test_interpolate_property(self):
        """Test that interpolate property returns the correct value."""
        settings = ClassifierSettings("test", 10, None, None)
        assert settings.interpolate == 10

    def test_stitch_property(self):
        """Test that stitch property returns the correct value."""
        settings = ClassifierSettings("test", None, 15, None)
        assert settings.stitch == 15

    def test_min_bout_property(self):
        """Test that min_bout property returns the correct value."""
        settings = ClassifierSettings("test", None, None, 20)
        assert settings.min_bout == 20

    def test_all_properties_with_none(self):
        """Test that all properties return defaults when initialized with None."""
        settings = ClassifierSettings("test", None, None, None)
        assert settings.behavior == "test"
        assert settings.interpolate == DEFAULT_INTERPOLATE
        assert settings.stitch == DEFAULT_STITCH
        assert settings.min_bout == DEFAULT_MIN_BOUT


class TestClassifierSettingsStringRepresentation:
    """Test ClassifierSettings string methods."""

    def test_str_with_explicit_values(self):
        """Test __str__ with explicit values."""
        settings = ClassifierSettings("grooming", 10, 15, 20)
        result = str(settings)

        assert "grooming" in result
        assert "10" in result
        assert "15" in result
        assert "20" in result

    def test_str_with_none_values_shows_defaults(self):
        """Test __str__ with None values shows defaults."""
        settings = ClassifierSettings("walking", None, None, None)
        result = str(settings)

        assert "walking" in result
        assert str(DEFAULT_INTERPOLATE) in result
        assert str(DEFAULT_STITCH) in result
        assert str(DEFAULT_MIN_BOUT) in result

    def test_repr_equals_str(self):
        """Test that __repr__ returns the same as __str__."""
        settings = ClassifierSettings("test", 5, 10, 15)
        assert repr(settings) == str(settings)


class TestFeatureSettingsInheritance:
    """Test that FeatureSettings inherits the None-handling behavior.

    FeatureSettings extends ClassifierSettings and should benefit from
    the same None-handling fix.
    """

    @pytest.fixture
    def temp_config_file(self, tmp_path):
        """Create a temporary YAML config file for testing."""
        config_file = tmp_path / "test_config.yaml"
        config_content = """
behavior: test_behavior
definition:
  - feature1 > 10
  - feature2 < 5
"""
        config_file.write_text(config_content)
        return config_file

    def test_feature_settings_with_none_values(self, temp_config_file):
        """Test FeatureSettings with None values uses defaults."""
        settings = FeatureSettings(
            config_file=str(temp_config_file),
            interpolate=None,
            stitch=None,
            min_bout=None,
        )

        # Should inherit None-handling from ClassifierSettings
        assert settings.interpolate == DEFAULT_INTERPOLATE
        assert settings.stitch == DEFAULT_STITCH
        assert settings.min_bout == DEFAULT_MIN_BOUT

    def test_feature_settings_with_explicit_values(self, temp_config_file):
        """Test FeatureSettings with explicit values overrides defaults."""
        settings = FeatureSettings(
            config_file=str(temp_config_file),
            interpolate=10,
            stitch=15,
            min_bout=20,
        )

        assert settings.interpolate == 10
        assert settings.stitch == 15
        assert settings.min_bout == 20

    def test_feature_settings_config_file_defaults(self, tmp_path):
        """Test that FeatureSettings can get defaults from config file."""
        config_file = tmp_path / "test_config.yaml"
        config_content = """
behavior: test_behavior
definition:
  - feature1 > 10
interpolate: 7
stitch: 12
min_bout: 18
"""
        config_file.write_text(config_content)

        settings = FeatureSettings(
            config_file=str(config_file),
            interpolate=None,
            stitch=None,
            min_bout=None,
        )

        # Should use values from config file
        assert settings.interpolate == 7
        assert settings.stitch == 12
        assert settings.min_bout == 18

    def test_feature_settings_explicit_overrides_config(self, tmp_path):
        """Test that explicit values override config file values."""
        config_file = tmp_path / "test_config.yaml"
        config_content = """
behavior: test_behavior
definition:
  - feature1 > 10
interpolate: 7
stitch: 12
min_bout: 18
"""
        config_file.write_text(config_content)

        settings = FeatureSettings(
            config_file=str(config_file),
            interpolate=100,
            stitch=200,
            min_bout=300,
        )

        # Explicit values should override config file
        assert settings.interpolate == 100
        assert settings.stitch == 200
        assert settings.min_bout == 300
