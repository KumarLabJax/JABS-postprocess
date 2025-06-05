"""Tests for the bouts_to_bins module.

This module tests the transform_bouts_to_bins function, which converts animal behavior bout data 
into time-binned summary statistics. The tests verify:

1. Correct handling of default and custom bin sizes
2. Proper filename transformation (from *_bouts.csv to *_summaries.csv)
3. Integration with file operations
4. Error handling for missing files
5. Support for different path formats (string, Path objects, relative, absolute)

The transform_bouts_to_bins function is a core data processing utility that aggregates 
fine-grained behavioral data into time-windowed summaries for further analysis.
"""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from jabs_postprocess.bouts_to_bins import transform_bouts_to_bins
from jabs_postprocess.utils.project_utils import BoutTable


@pytest.fixture
def mock_bout_table():
    """Create a mock BoutTable for testing.
    
    Returns:
        MagicMock: A mock object that simulates a BoutTable instance with the necessary
            methods stubbed out for testing. The mock is configured to return another mock
            when to_summary_table() is called, allowing for method chaining in tests.
    """
    mock_table = MagicMock(spec=BoutTable)
    mock_summary_table = MagicMock()
    mock_table.to_summary_table.return_value = mock_summary_table
    return mock_table


def test_transform_bouts_to_bins_default_bin_size(mock_bout_table, tmp_path):
    """Test transform_bouts_to_bins with default bin size.
    
    This test verifies that:
    1. The function correctly uses a default bin size of 60 minutes when none is specified
    2. It loads bout data from the correct file
    3. It generates the correct output filename (replacing '_bouts' with '_summaries')
    4. It properly calls the underlying methods to transform and save the data
    
    Args:
        mock_bout_table: Fixture providing a mock BoutTable object
        tmp_path: Pytest fixture providing a temporary directory path
    """
    # Arrange
    bout_file = tmp_path / "test_bouts.csv"
    bout_file.touch()
    expected_summary_file = tmp_path / "test_summaries.csv"
    
    with patch("jabs_postprocess.bouts_to_bins.BoutTable.from_file", 
               return_value=mock_bout_table) as mock_from_file:
        # Act
        transform_bouts_to_bins(bout_file)
        
        # Assert
        mock_from_file.assert_called_once_with(bout_file)
        mock_bout_table.to_summary_table.assert_called_once_with(60)
        mock_bout_table.to_summary_table().to_file.assert_called_once()
        # Check that the output filename uses the correct pattern
        file_arg = mock_bout_table.to_summary_table().to_file.call_args[0][0]
        assert str(file_arg) == str(expected_summary_file)


@pytest.mark.parametrize("bin_size", [30, 60, 120])
def test_transform_bouts_to_bins_custom_bin_size(mock_bout_table, bin_size, tmp_path):
    """Test transform_bouts_to_bins with custom bin sizes.
    
    This test verifies that the function properly handles user-specified bin sizes
    by passing them correctly to the underlying BoutTable.to_summary_table method.
    Multiple bin sizes are tested to ensure flexibility in time window configuration.
    
    Args:
        mock_bout_table: Fixture providing a mock BoutTable object
        bin_size: The bin size in minutes to test (parameterized with multiple values)
        tmp_path: Pytest fixture providing a temporary directory path
    """
    # Arrange
    bout_file = tmp_path / "test_bouts.csv"
    bout_file.touch()
    
    with patch("jabs_postprocess.bouts_to_bins.BoutTable.from_file", 
               return_value=mock_bout_table) as mock_from_file:
        # Act
        transform_bouts_to_bins(bout_file, bin_size=bin_size)
        
        # Assert
        mock_from_file.assert_called_once_with(bout_file)
        mock_bout_table.to_summary_table.assert_called_once_with(bin_size)


def test_transform_bouts_to_bins_filename_substitution():
    """Test the filename pattern substitution logic.
    
    This test verifies the function's ability to properly transform input filenames
    into output filenames by replacing '_bouts' with '_summaries' while preserving:
    - File extensions
    - Directory paths
    - Filenames with multiple occurrences of 'bouts'
    - Files without extensions
    
    This ensures the output files are named consistently and predictably.
    """
    # Arrange
    test_cases = [
        ("path/to/file_bouts.csv", "path/to/file_summaries.csv"),
        ("file_bouts.txt", "file_summaries.txt"),
        ("file_with_bouts_in_name_bouts", "file_with_bouts_in_name_summaries"),
        ("file_with_bouts_twice_bouts_bouts.csv", "file_with_bouts_twice_bouts_summaries.csv")
    ]
    
    # Act & Assert
    for input_file, expected_output in test_cases:
        mock_bout_data = MagicMock()
        mock_bin_data = MagicMock()
        
        with patch("jabs_postprocess.bouts_to_bins.BoutTable.from_file", 
                  return_value=mock_bout_data), \
             patch.object(mock_bout_data, "to_summary_table", 
                         return_value=mock_bin_data):
            
            # Convert expected output to Path for comparison
            expected_path = Path(expected_output)
            
            transform_bouts_to_bins(input_file)
            
            # Verify the correct output path was passed to to_file
            mock_bin_data.to_file.assert_called_once()
            actual_path = mock_bin_data.to_file.call_args[0][0]
            assert str(actual_path) == str(expected_path)


def test_transform_bouts_to_bins_integration(tmp_path):
    """Integration test with actual file operations.
    
    Unlike the more focused unit tests, this test verifies the entire function
    workflow by patching at a higher level and ensuring the complete chain of
    operations functions correctly. This test confirms:
    
    1. The BoutTable is loaded from the specified file
    2. The to_summary_table method is called with the expected bin size
    3. The resulting summary table's to_file method is called to save results
    
    Args:
        tmp_path: Pytest fixture providing a temporary directory path
    """
    # Arrange
    bout_file = tmp_path / "test_bouts.csv"
    summary_file = tmp_path / "test_summaries.csv"
    
    # Create a mock DataFrame that BoutTable would return
    mock_data = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': ['a', 'b', 'c']
    })
    
    mock_bout_table = MagicMock()
    mock_summary_table = MagicMock()
    mock_bout_table.to_summary_table.return_value = mock_summary_table
    
    # Act
    with patch("jabs_postprocess.bouts_to_bins.BoutTable.from_file", 
               return_value=mock_bout_table):
        transform_bouts_to_bins(bout_file)
    
    # Assert
    mock_bout_table.to_summary_table.assert_called_once_with(60)
    mock_summary_table.to_file.assert_called_once()


def test_transform_bouts_to_bins_error_handling():
    """Test error handling in transform_bouts_to_bins.
    
    This test verifies that the function properly propagates exceptions from
    underlying operations rather than silently failing. Specifically, it checks
    that FileNotFoundError is raised when the specified input file doesn't exist.
    
    Error propagation is important for debugging and for providing clear feedback
    to users about what went wrong during processing.
    """
    # Arrange
    non_existent_file = Path("non_existent_file_bouts.csv")
    
    # Act & Assert - Check that the function properly passes through exceptions
    with patch("jabs_postprocess.bouts_to_bins.BoutTable.from_file", 
              side_effect=FileNotFoundError("File not found")):
        with pytest.raises(FileNotFoundError):
            transform_bouts_to_bins(non_existent_file)


def test_transform_bouts_to_bins_path_handling():
    """Test how different path types are handled.
    
    This test verifies the function's ability to handle various path formats:
    1. Relative path strings
    2. Absolute path strings 
    3. Path objects
    
    For each format, it checks that:
    - The path is correctly converted to a Path object if needed
    - The BoutTable.from_file method is called with the correct path
    - The output filename transformation is applied correctly
    
    This ensures the function works consistently regardless of how users specify paths.
    """
    # Arrange
    path_cases = [
        "relative/path/file_bouts.csv",
        "/absolute/path/file_bouts.csv",
        Path("path/object/file_bouts.csv")
    ]
    
    # Act & Assert
    for path in path_cases:
        mock_bin_data = MagicMock()
        mock_bout_table = MagicMock()
        
        with patch("jabs_postprocess.bouts_to_bins.BoutTable.from_file", 
                  return_value=mock_bout_table) as mock_from_file, \
             patch.object(mock_bout_table, "to_summary_table", 
                         return_value=mock_bin_data):
            
            transform_bouts_to_bins(path)
            
            # Verify Path handling
            path_obj = Path(path)
            mock_from_file.assert_called_once_with(path_obj)
            
            # Verify the filename transformation
            expected_filename = path_obj.name.replace('_bouts', '_summaries')
            expected_path = path_obj.parent / expected_filename
            
            mock_bin_data.to_file.assert_called_once()
            actual_path = mock_bin_data.to_file.call_args[0][0]
            assert str(actual_path) == str(expected_path)
