"""
Tests for the `heuristic_classify` module.

This test file validates the functionality of the heuristic classification process in JABS,
focusing on the `process_heuristic_classification` function which is responsible for:
1. Loading behavior configurations
2. Processing behavioral data with configurable parameters
3. Generating and saving bout tables and summary tables

The tests cover various parameter combinations, backward compatibility,
and verify that the function correctly interacts with the underlying
JABS components (FeatureSettings, JabsProject, etc.).
"""
from unittest.mock import MagicMock, patch

import pytest

from jabs_postprocess.heuristic_classify import process_heuristic_classification


@pytest.fixture
def mock_project():
    """
    Fixture providing a mock JabsProject.
    
    Creates a mock JabsProject with mocked bout table and bin table methods,
    simulating the project's data processing capabilities without requiring
    actual file operations or data processing.
    """
    mock_project = MagicMock()
    mock_bout_table = MagicMock()
    mock_bin_table = MagicMock()
    
    mock_project.get_bouts.return_value = mock_bout_table
    mock_bout_table.to_summary_table.return_value = mock_bin_table
    
    return mock_project


@pytest.fixture
def mock_feature_settings():
    """
    Fixture providing a mock FeatureSettings.
    
    Creates a mock FeatureSettings object with a test behavior attribute,
    allowing tests to verify behavior classification parameters without
    loading actual configuration files.
    """
    mock_settings = MagicMock()
    mock_settings.behavior = "test_behavior"
    return mock_settings


@pytest.mark.parametrize(
    "project_folder,behavior_config,feature_folder,out_prefix,out_bin_size,overwrite,interpolate_size,stitch_gap,min_bout_length",
    [
        # Basic test case with defaults
        ("test_project", "test_config.yaml", "features", "behavior", 60, False, None, None, None),
        # Custom parameters
        ("test_project", "test_config.yaml", "custom_features", "custom", 30, True, 5, 10, 15),
        # Edge case: minimum values
        ("test_project", "test_config.yaml", "", "b", 1, False, 0, 0, 0),
    ],
)
def test_process_heuristic_classification(
    mock_project, mock_feature_settings,
    project_folder, behavior_config, feature_folder, out_prefix, out_bin_size, 
    overwrite, interpolate_size, stitch_gap, min_bout_length
):
    """
    Test process_heuristic_classification with various parameter combinations.
    
    This comprehensive test validates the heuristic classification process with:
    1. Basic test case with default parameters
    2. Custom parameters for all configurable options
    3. Edge case with minimum values for numerical parameters
    
    The test verifies that:
    - FeatureSettings is initialized with correct parameters
    - JabsProject is created from the feature folder correctly
    - Bout tables are properly generated and saved
    - Summary tables are created with the specified bin size
    - File naming follows the expected pattern based on output prefix and behavior
    
    Args:
        mock_project: Fixture providing a mock JabsProject
        mock_feature_settings: Fixture providing a mock FeatureSettings
        project_folder: Path to the project folder
        behavior_config: Path to the behavior configuration file
        feature_folder: Path to the feature folder
        out_prefix: Prefix for output files
        out_bin_size: Size of bins for summary tables
        overwrite: Whether to overwrite existing files
        interpolate_size: Size for interpolation (or None for default)
        stitch_gap: Gap size for stitching (or None for default)
        min_bout_length: Minimum bout length (or None for default)
    """
    # Arrange
    with patch("jabs_postprocess.heuristic_classify.FeatureSettings") as mock_settings_cls, \
         patch("jabs_postprocess.heuristic_classify.JabsProject") as mock_project_cls:
        
        mock_settings_cls.return_value = mock_feature_settings
        mock_project_cls.from_feature_folder.return_value = mock_project
        
        # Mock the tables for output verification
        mock_bout_table = mock_project.get_bouts.return_value
        mock_bin_table = mock_bout_table.to_summary_table.return_value
        
        expected_bout_file = f"{out_prefix}_{mock_feature_settings.behavior}_bouts.csv"
        expected_bin_file = f"{out_prefix}_{mock_feature_settings.behavior}_summaries.csv"
        
        # Act
        process_heuristic_classification(
            project_folder=project_folder,
            behavior_config=behavior_config,
            feature_folder=feature_folder,
            out_prefix=out_prefix,
            out_bin_size=out_bin_size,
            overwrite=overwrite,
            interpolate_size=interpolate_size,
            stitch_gap=stitch_gap,
            min_bout_length=min_bout_length
        )
        
        # Assert
        # Check that FeatureSettings was initialized with the correct parameters
        mock_settings_cls.assert_called_once_with(
            behavior_config, interpolate_size, stitch_gap, min_bout_length
        )
        
        # Check that JabsProject.from_feature_folder was called correctly
        mock_project_cls.from_feature_folder.assert_called_once_with(
            project_folder, mock_feature_settings, feature_folder
        )
        
        # Check that get_bouts was called
        mock_project.get_bouts.assert_called_once()
        
        # Check that to_file was called with the right parameters for bout table
        mock_bout_table.to_file.assert_called_once_with(expected_bout_file, overwrite)
        
        # Check that to_summary_table was called with the right bin size
        mock_bout_table.to_summary_table.assert_called_once_with(out_bin_size)
        
        # Check that to_file was called with the right parameters for bin table
        mock_bin_table.to_file.assert_called_once_with(expected_bin_file, overwrite)


def test_process_heuristic_classification_backwards_compatibility(mock_project, mock_feature_settings):
    """
    Test backward compatibility of process_heuristic_classification.
    
    This test verifies that the function correctly applies default values
    when optional parameters are not provided, ensuring backward compatibility
    with existing code that may rely on default behavior.
    
    The test specifically checks:
    - Default values are correctly applied when optional parameters are omitted
    - The function can be called with minimal required parameters
    - The default feature folder ('features') is used when not specified
    
    This ensures that updates to the function preserve existing behavior
    for code that depends on previous parameter conventions.
    
    Args:
        mock_project: Fixture providing a mock JabsProject
        mock_feature_settings: Fixture providing a mock FeatureSettings
    """
    # Arrange
    with patch("jabs_postprocess.heuristic_classify.FeatureSettings") as mock_settings_cls, \
         patch("jabs_postprocess.heuristic_classify.JabsProject") as mock_project_cls:
        
        mock_settings_cls.return_value = mock_feature_settings
        mock_project_cls.from_feature_folder.return_value = mock_project
        
        # Act
        process_heuristic_classification(
            project_folder="test_project",
            behavior_config="test_config.yaml"
        )
        
        # Assert
        # Check that FeatureSettings was called with None values for optional parameters
        mock_settings_cls.assert_called_once_with(
            "test_config.yaml", None, None, None
        )
        
        # Check that JabsProject.from_feature_folder was called with the default feature_folder
        mock_project_cls.from_feature_folder.assert_called_once_with(
            "test_project", mock_feature_settings, "features"
        )
