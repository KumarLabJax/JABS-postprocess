"""Unit tests for the generate_behavior_tables module.

This test module validates the functionality of the behavior table generation process
in JABS (Just Another Behavior Scorer). The main functions under test are:

1. process_behavior_tables - Processes a single behavior, extracting bout information
   and creating summary tables with configurable parameters

2. process_multiple_behaviors - Processes multiple behaviors in batch, validating
   proper parameter handling across different behavior configurations

The tests use mocking extensively to isolate the code under test from external
dependencies and validate proper interaction between components.
"""

from unittest.mock import MagicMock, patch

import pytest

from jabs_postprocess.generate_behavior_tables import (
    process_behavior_tables,
    process_multiple_behaviors,
)


@pytest.fixture
def mock_project():
    """Create a mock JabsProject.

    This fixture provides a mock JabsProject instance with predefined
    return values for get_bouts() and to_summary_table() methods, allowing
    tests to run without actual data processing.
    """
    mock = MagicMock()
    mock_bout_table = MagicMock()
    mock_bin_table = MagicMock()

    mock.get_bouts.return_value = mock_bout_table
    mock_bout_table.to_summary_table.return_value = mock_bin_table

    return mock


@pytest.fixture
def mock_find_behaviors():
    """Mock the JabsProject.find_behaviors method.

    This fixture patches the find_behaviors method to return a predefined
    list of behaviors, allowing tests to verify behavior validation without
    requiring actual project data.
    """
    with patch(
        "jabs_postprocess.generate_behavior_tables.JabsProject.find_behaviors"
    ) as mock:
        mock.return_value = ["grooming", "walking", "feeding"]
        yield mock


def test_process_behavior_tables_default_params(mock_project):
    """Test process_behavior_tables with default parameters.

    Validates that when process_behavior_tables is called with only the
    required parameters (project_folder and behavior), it:
    1. Creates ClassifierSettings with appropriate defaults
    2. Loads the JabsProject correctly
    3. Retrieves bout data and generates summary tables
    4. Saves files with expected naming convention
    5. Returns the correct tuple of filenames
    """
    with (
        patch(
            "jabs_postprocess.generate_behavior_tables.JabsProject.from_prediction_folder",
            return_value=mock_project,
        ) as mock_from_folder,
        patch(
            "jabs_postprocess.generate_behavior_tables.ClassifierSettings"
        ) as mock_settings,
    ):
        # Act
        result = process_behavior_tables(
            project_folder="/path/to/project", behavior="grooming"
        )

        # Assert
        mock_settings.assert_called_once_with("grooming", None, None, None)
        mock_from_folder.assert_called_once_with(
            "/path/to/project", mock_settings.return_value, None
        )
        mock_project.get_bouts.assert_called_once()
        mock_project.get_bouts.return_value.to_file.assert_called_once_with(
            "behavior_grooming_bouts.csv", False
        )
        mock_project.get_bouts.return_value.to_summary_table.assert_called_once_with(60)
        mock_project.get_bouts.return_value.to_summary_table.return_value.to_file.assert_called_once_with(
            "behavior_grooming_summaries.csv", False
        )
        assert result == (
            "behavior_grooming_bouts.csv",
            "behavior_grooming_summaries.csv",
        )


def test_process_behavior_tables_custom_params(mock_project):
    """Test process_behavior_tables with custom parameters.

    Verifies that process_behavior_tables correctly handles all optional parameters:
    1. Behavior name, output prefix, bin size, feature folder location
    2. Interpolation size, stitching gap, minimum bout length
    3. File overwrite flag

    This test ensures that all customization options work as expected and are
    passed to the correct underlying components.
    """
    with (
        patch(
            "jabs_postprocess.generate_behavior_tables.JabsProject.from_prediction_folder",
            return_value=mock_project,
        ) as mock_from_folder,
        patch(
            "jabs_postprocess.generate_behavior_tables.ClassifierSettings"
        ) as mock_settings,
    ):
        # Act
        result = process_behavior_tables(
            project_folder="/path/to/project",
            behavior="walking",
            out_prefix="custom",
            out_bin_size=120,
            feature_folder="/custom/features",
            interpolate_size=5,
            stitch_gap=3,
            min_bout_length=10,
            overwrite=True,
        )

        # Assert
        mock_settings.assert_called_once_with("walking", 5, 3, 10)
        mock_from_folder.assert_called_once_with(
            "/path/to/project", mock_settings.return_value, "/custom/features"
        )
        mock_project.get_bouts.assert_called_once()
        mock_project.get_bouts.return_value.to_file.assert_called_once_with(
            "custom_walking_bouts.csv", True
        )
        mock_project.get_bouts.return_value.to_summary_table.assert_called_once_with(
            120
        )
        mock_project.get_bouts.return_value.to_summary_table.return_value.to_file.assert_called_once_with(
            "custom_walking_summaries.csv", True
        )
        assert result == ("custom_walking_bouts.csv", "custom_walking_summaries.csv")


@pytest.mark.parametrize(
    "interpolate_size,stitch_gap,min_bout_length",
    [
        (0, 0, 0),
        (1, 1, 1),
        (100, 50, 25),
        (None, None, None),
    ],
)
def test_process_behavior_tables_param_combinations(
    mock_project, interpolate_size, stitch_gap, min_bout_length
):
    """Test process_behavior_tables with different parameter combinations.

    This parametrized test verifies that the function handles various combinations
    of bout processing parameters correctly:
    - Zero values
    - Small positive values
    - Large positive values
    - None values (default behavior)

    Each combination is verified to be correctly passed to the ClassifierSettings constructor.
    """
    with (
        patch(
            "jabs_postprocess.generate_behavior_tables.JabsProject.from_prediction_folder",
            return_value=mock_project,
        ),
        patch(
            "jabs_postprocess.generate_behavior_tables.ClassifierSettings"
        ) as mock_settings,
    ):
        # Act
        process_behavior_tables(
            project_folder="/path/to/project",
            behavior="grooming",
            interpolate_size=interpolate_size,
            stitch_gap=stitch_gap,
            min_bout_length=min_bout_length,
        )

        # Assert
        mock_settings.assert_called_once_with(
            "grooming", interpolate_size, stitch_gap, min_bout_length
        )


def test_process_behavior_tables_empty_project_folder():
    """Test process_behavior_tables with empty project folder.

    Validates proper error handling when the project folder doesn't exist
    or is empty. The function should propagate the ValueError from the
    JabsProject.from_prediction_folder method.
    """
    with patch(
        "jabs_postprocess.generate_behavior_tables.JabsProject.from_prediction_folder"
    ) as mock_from_folder:
        mock_from_folder.side_effect = ValueError(
            "Project folder is empty or does not exist"
        )

        # Act & Assert
        with pytest.raises(
            ValueError, match="Project folder is empty or does not exist"
        ):
            process_behavior_tables(project_folder="", behavior="grooming")


def test_process_behavior_tables_error_during_processing(mock_project):
    """Test process_behavior_tables handles errors during bout processing.

    Verifies that any exceptions raised during the bout processing are properly
    propagated to the caller rather than being silently caught. This ensures
    that users receive appropriate error information.
    """
    with patch(
        "jabs_postprocess.generate_behavior_tables.JabsProject.from_prediction_folder",
        return_value=mock_project,
    ):
        # Mock an error during bout processing
        mock_project.get_bouts.side_effect = RuntimeError("Failed to process bouts")

        # Act & Assert
        with pytest.raises(RuntimeError, match="Failed to process bouts"):
            process_behavior_tables(
                project_folder="/path/to/project", behavior="grooming"
            )


@pytest.mark.parametrize(
    "behaviors,expected_calls",
    [
        # Single behavior with default settings
        (
            [{"behavior": "grooming"}],
            [
                {
                    "project_folder": "/path/to/project",
                    "behavior": "grooming",
                    "out_prefix": "behavior",
                    "out_bin_size": 60,
                    "feature_folder": None,
                    "interpolate_size": None,
                    "stitch_gap": None,
                    "min_bout_length": None,
                    "overwrite": False,
                }
            ],
        ),
        # Multiple behaviors with different settings
        (
            [
                {"behavior": "grooming", "interpolate_size": 5},
                {"behavior": "walking", "stitch_gap": 3, "min_bout_length": 10},
            ],
            [
                {
                    "project_folder": "/path/to/project",
                    "behavior": "grooming",
                    "out_prefix": "behavior",
                    "out_bin_size": 60,
                    "feature_folder": None,
                    "interpolate_size": 5,
                    "stitch_gap": None,
                    "min_bout_length": None,
                    "overwrite": False,
                },
                {
                    "project_folder": "/path/to/project",
                    "behavior": "walking",
                    "out_prefix": "behavior",
                    "out_bin_size": 60,
                    "feature_folder": None,
                    "interpolate_size": None,
                    "stitch_gap": 3,
                    "min_bout_length": 10,
                    "overwrite": False,
                },
            ],
        ),
        # Empty behaviors list
        ([], []),
        # Multiple behaviors with same settings
        (
            [
                {"behavior": "grooming", "interpolate_size": 5},
                {"behavior": "walking", "interpolate_size": 5},
                {"behavior": "feeding", "interpolate_size": 5},
            ],
            [
                {
                    "project_folder": "/path/to/project",
                    "behavior": "grooming",
                    "out_prefix": "behavior",
                    "out_bin_size": 60,
                    "feature_folder": None,
                    "interpolate_size": 5,
                    "stitch_gap": None,
                    "min_bout_length": None,
                    "overwrite": False,
                },
                {
                    "project_folder": "/path/to/project",
                    "behavior": "walking",
                    "out_prefix": "behavior",
                    "out_bin_size": 60,
                    "feature_folder": None,
                    "interpolate_size": 5,
                    "stitch_gap": None,
                    "min_bout_length": None,
                    "overwrite": False,
                },
                {
                    "project_folder": "/path/to/project",
                    "behavior": "feeding",
                    "out_prefix": "behavior",
                    "out_bin_size": 60,
                    "feature_folder": None,
                    "interpolate_size": 5,
                    "stitch_gap": None,
                    "min_bout_length": None,
                    "overwrite": False,
                },
            ],
        ),
    ],
)
def test_process_multiple_behaviors(behaviors, expected_calls, mock_find_behaviors):
    """Test process_multiple_behaviors with different behavior configurations.

    This parametrized test validates that process_multiple_behaviors correctly:
    1. Handles various behavior input configurations (single, multiple, empty)
    2. Correctly passes behavior-specific parameters to process_behavior_tables
    3. Validates behavior existence against available behaviors
    4. Returns the expected list of output filenames

    The test covers:
    - Single behavior with default settings
    - Multiple behaviors with different settings per behavior
    - Empty behaviors list
    - Multiple behaviors with identical settings
    """
    # Arrange
    with patch(
        "jabs_postprocess.generate_behavior_tables.process_behavior_tables"
    ) as mock_process:
        mock_process.side_effect = lambda **kwargs: (
            f"{kwargs['out_prefix']}_{kwargs['behavior']}_bouts.csv",
            f"{kwargs['out_prefix']}_{kwargs['behavior']}_summaries.csv",
        )

        # Act
        result = process_multiple_behaviors(
            project_folder="/path/to/project", behaviors=behaviors
        )

        # Assert
        assert mock_find_behaviors.call_count == 1
        assert mock_process.call_count == len(behaviors)

        actual_calls = [call[1] for call in mock_process.call_args_list]
        for i, expected in enumerate(expected_calls):
            for key, value in expected.items():
                assert actual_calls[i][key] == value

        expected_result = [
            (
                f"behavior_{b['behavior']}_bouts.csv",
                f"behavior_{b['behavior']}_summaries.csv",
            )
            for b in behaviors
        ]
        assert result == expected_result


def test_process_multiple_behaviors_custom_params(mock_find_behaviors):
    """Test process_multiple_behaviors with custom prefix, bin size, and feature folder.

    Verifies that global parameters provided to process_multiple_behaviors
    (output prefix, bin size, feature folder, and overwrite flag) are correctly
    passed to each individual process_behavior_tables call.

    This ensures that common settings can be applied globally while still allowing
    behavior-specific overrides.
    """
    # Arrange
    behaviors = [{"behavior": "grooming"}, {"behavior": "walking"}]

    with patch(
        "jabs_postprocess.generate_behavior_tables.process_behavior_tables"
    ) as mock_process:
        mock_process.side_effect = lambda **kwargs: (
            f"{kwargs['out_prefix']}_{kwargs['behavior']}_bouts.csv",
            f"{kwargs['out_prefix']}_{kwargs['behavior']}_summaries.csv",
        )

        # Act
        result = process_multiple_behaviors(
            project_folder="/path/to/project",
            behaviors=behaviors,
            out_prefix="custom",
            out_bin_size=120,
            feature_folder="/custom/features",
            overwrite=True,
        )

        # Assert
        assert mock_process.call_count == 2

        for i, behavior in enumerate(behaviors):
            assert (
                mock_process.call_args_list[i][1]["project_folder"]
                == "/path/to/project"
            )
            assert mock_process.call_args_list[i][1]["behavior"] == behavior["behavior"]
            assert mock_process.call_args_list[i][1]["out_prefix"] == "custom"
            assert mock_process.call_args_list[i][1]["out_bin_size"] == 120
            assert (
                mock_process.call_args_list[i][1]["feature_folder"]
                == "/custom/features"
            )
            assert mock_process.call_args_list[i][1]["overwrite"] is True

        expected_result = [
            ("custom_grooming_bouts.csv", "custom_grooming_summaries.csv"),
            ("custom_walking_bouts.csv", "custom_walking_summaries.csv"),
        ]
        assert result == expected_result


def test_process_multiple_behaviors_behavior_not_found(mock_find_behaviors):
    """Test process_multiple_behaviors raises error for non-existent behavior.

    Validates that process_multiple_behaviors correctly:
    1. Checks each requested behavior against the list of available behaviors
    2. Raises a descriptive ValueError when a behavior is not found

    This prevents users from attempting to process non-existent behaviors
    which would fail later with less clear error messages.
    """
    # Arrange
    behaviors = [{"behavior": "grooming"}, {"behavior": "invalid_behavior"}]

    # Act & Assert
    with pytest.raises(ValueError, match="invalid_behavior not in experiment folder"):
        process_multiple_behaviors("/path/to/project", behaviors)


def test_process_multiple_behaviors_missing_behavior_key():
    """Test process_multiple_behaviors raises error when behavior key is missing.

    Verifies that the function properly validates the structure of each
    behavior specification dictionary, requiring a 'behavior' key to identify
    which behavior to process.

    This provides early error detection for malformed behavior configurations.
    """
    # Arrange
    behaviors = [{"behavior": "grooming"}, {"not_behavior_key": "walking"}]

    # Act & Assert
    with pytest.raises(KeyError, match="Behavior name required"):
        process_multiple_behaviors("/path/to/project", behaviors)


def test_process_multiple_behaviors_error_propagation(mock_find_behaviors):
    """Test that errors from process_behavior_tables are propagated.

    Verifies that errors occurring during individual behavior processing
    are properly propagated up the call stack, ensuring that failures
    are visible to the user and not silently ignored.
    """
    # Arrange
    behaviors = [{"behavior": "grooming"}, {"behavior": "walking"}]

    with patch(
        "jabs_postprocess.generate_behavior_tables.process_behavior_tables"
    ) as mock_process:
        # Mock an error during processing the second behavior
        def side_effect(**kwargs):
            if kwargs["behavior"] == "walking":
                raise RuntimeError("Failed to process walking behavior")
            return (
                f"{kwargs['out_prefix']}_{kwargs['behavior']}_bouts.csv",
                f"{kwargs['out_prefix']}_{kwargs['behavior']}_summaries.csv",
            )

        mock_process.side_effect = side_effect

        # Act & Assert
        with pytest.raises(RuntimeError, match="Failed to process walking behavior"):
            process_multiple_behaviors("/path/to/project", behaviors)


def test_process_multiple_behaviors_no_available_behaviors():
    """Test process_multiple_behaviors when no behaviors are available.

    Verifies proper error handling when find_behaviors returns an empty list,
    indicating that no behaviors are available for processing in the project.

    This ensures users receive clear feedback when their project data lacks
    behavior information rather than encountering confusing behavior-not-found errors.
    """
    # Arrange
    behaviors = [{"behavior": "grooming"}]

    with patch(
        "jabs_postprocess.generate_behavior_tables.JabsProject.find_behaviors"
    ) as mock_find:
        mock_find.return_value = []

        # Act & Assert
        with pytest.raises(ValueError, match="grooming not in experiment folder"):
            process_multiple_behaviors("/path/to/project", behaviors)


def test_process_multiple_behaviors_empty_list(mock_find_behaviors):
    """Test process_multiple_behaviors with empty list returns empty list.

    Verifies that providing an empty behaviors list is handled gracefully
    by returning an empty result list without attempting to process anything.

    This ensures consistent behavior when no behaviors are specified.
    """
    # Arrange
    behaviors = []

    with patch(
        "jabs_postprocess.generate_behavior_tables.process_behavior_tables"
    ) as mock_process:
        # Act
        result = process_multiple_behaviors("/path/to/project", behaviors)

        # Assert
        assert mock_process.call_count == 0
        assert result == []
