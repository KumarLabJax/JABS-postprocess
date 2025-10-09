"""Unit tests for the merge_multiple_tables CLI command.

This test module validates the functionality of the merge_multiple_tables CLI command.
The command scans a folder for behavior table files, groups them by behavior name, and
merges each group separately. This is useful for combining results from multiple
experiments.

Key functionality tested:
1. Basic table merging with auto-detected behaviors
2. Behavior filtering and selection
3. File pattern matching and table discovery
4. Output file generation and overwrite behavior
5. Error handling for missing folders and invalid files
6. Table grouping by behavior name extraction
"""

from pathlib import Path
from unittest.mock import patch
import pytest
import pandas as pd

from jabs_postprocess.cli.main import app


class TestMergeMultipleTables:
    """Test class for the merge_multiple_tables CLI command."""

    @pytest.mark.parametrize("behavior_count", [1, 2, 4])
    @pytest.mark.parametrize("overwrite", [True, False])
    @patch("jabs_postprocess.cli.main.generate_behavior_tables")
    @patch("pandas.read_csv")
    def test_merge_multiple_tables_basic(
        self,
        mock_read_csv,
        mock_generate_module,
        runner,
        mock_table_folder,
        behavior_count,
        overwrite,
    ):
        """Test basic table merging functionality.

        Args:
            mock_read_csv: Mock pandas read_csv function
            mock_generate_module: Mock generate_behavior_tables module
            runner: CLI test runner
            mock_table_folder: Mock table folder with files
            behavior_count: Number of behaviors to simulate
            overwrite: Whether to enable overwrite
        """
        # Arrange
        table_folder, files = mock_table_folder

        # Mock pandas read_csv to return behavior names
        def mock_csv_reader(file_path, nrows=None):
            file_name = Path(file_path).name
            if "behavior_0" in file_name:
                return pd.DataFrame({"Behavior": ["behavior_0"]})
            elif "behavior_1" in file_name:
                return pd.DataFrame({"Behavior": ["behavior_1"]})
            else:
                return pd.DataFrame({"Behavior": ["behavior_2"]})

        mock_read_csv.side_effect = mock_csv_reader

        # Mock merge function return value
        mock_results = {
            f"behavior_{i}": (f"merged_bout_{i}.csv", f"merged_bin_{i}.csv")
            for i in range(behavior_count)
        }
        mock_generate_module.merge_multiple_behavior_tables.return_value = mock_results

        cmd_args = [
            "merge-multiple-tables",
            "--table-folder",
            str(table_folder),
            "--output-prefix",
            "merged_behavior"
        ]

        if overwrite:
            cmd_args.append("--overwrite")

        # Act
        result = runner.invoke(app, cmd_args)

        # Assert
        assert result.exit_code == 0

        # Verify merge function was called
        mock_generate_module.merge_multiple_behavior_tables.assert_called_once()
        call_args = mock_generate_module.merge_multiple_behavior_tables.call_args

        assert call_args.kwargs["output_prefix"] == "merged_behavior"
        assert call_args.kwargs["overwrite"] == overwrite
        assert "table_groups" in call_args.kwargs

        # Verify output messages
        assert (
            f"Successfully merged tables for {behavior_count} behaviors:"
            in result.stdout
        )

    @pytest.mark.parametrize("table_pattern", ["*.csv", "*_bout.csv", "behavior_*.csv"])
    @patch("jabs_postprocess.cli.main.generate_behavior_tables")
    @patch("pandas.read_csv")
    def test_merge_multiple_tables_custom_pattern(
        self,
        mock_read_csv,
        mock_generate_module,
        runner,
        mock_table_folder,
        table_pattern,
    ):
        """Test table merging with custom file patterns.

        Args:
            mock_read_csv: Mock pandas read_csv function
            mock_generate_module: Mock generate_behavior_tables module
            runner: CLI test runner
            mock_table_folder: Mock table folder with files
            table_pattern: File pattern to test
        """
        # Arrange
        table_folder, files = mock_table_folder
        mock_read_csv.return_value = pd.DataFrame({"Behavior": ["test_behavior"]})
        mock_generate_module.merge_multiple_behavior_tables.return_value = {
            "test_behavior": ("bout.csv", "bin.csv")
        }

        cmd_args = [
            "merge-multiple-tables",
            "--table-folder",
            str(table_folder),
            "--table-pattern",
            table_pattern,
        ]

        # Act
        result = runner.invoke(app, cmd_args)

        # Assert
        assert result.exit_code == 0
        mock_generate_module.merge_multiple_behavior_tables.assert_called_once()

    @pytest.mark.parametrize(
        "selected_behaviors", [["behavior_0"], ["behavior_1", "behavior_2"]]
    )
    @patch("jabs_postprocess.cli.main.generate_behavior_tables")
    @patch("pandas.read_csv")
    def test_merge_multiple_tables_behavior_filtering(
        self,
        mock_read_csv,
        mock_generate_module,
        runner,
        mock_table_folder,
        selected_behaviors,
    ):
        """Test table merging with behavior filtering.

        Args:
            mock_read_csv: Mock pandas read_csv function
            mock_generate_module: Mock generate_behavior_tables module
            runner: CLI test runner
            mock_table_folder: Mock table folder with files
            selected_behaviors: List of behaviors to filter for
        """
        # Arrange
        table_folder, files = mock_table_folder

        # Mock pandas to return different behaviors for different files
        def mock_csv_reader(file_path, nrows=None):
            file_name = Path(file_path).name
            if "behavior_0" in file_name:
                return pd.DataFrame({"Behavior": ["behavior_0"]})
            elif "behavior_1" in file_name:
                return pd.DataFrame({"Behavior": ["behavior_1"]})
            else:
                return pd.DataFrame({"Behavior": ["behavior_2"]})

        mock_read_csv.side_effect = mock_csv_reader

        # Mock results for selected behaviors only
        mock_results = {
            behavior: (f"{behavior}_bout.csv", f"{behavior}_bin.csv")
            for behavior in selected_behaviors
        }
        mock_generate_module.merge_multiple_behavior_tables.return_value = mock_results

        cmd_args = [
            "merge-multiple-tables",
            "--table-folder",
            str(table_folder),
        ]

        for behavior in selected_behaviors:
            cmd_args.extend(["--behaviors", behavior])

        # Act
        result = runner.invoke(app, cmd_args)

        # Assert
        assert result.exit_code == 0

        call_args = mock_generate_module.merge_multiple_behavior_tables.call_args
        table_groups = call_args.kwargs["table_groups"]

        # Verify only selected behaviors are included
        assert set(table_groups.keys()) == set(selected_behaviors)

    @pytest.mark.parametrize(
        "output_prefix", ["custom_prefix", "experiment_1", "merged_data"]
    )
    @patch("jabs_postprocess.cli.main.generate_behavior_tables")
    @patch("pandas.read_csv")
    def test_merge_multiple_tables_custom_prefix(
        self,
        mock_read_csv,
        mock_generate_module,
        runner,
        mock_table_folder,
        output_prefix,
    ):
        """Test table merging with custom output prefix.

        Args:
            mock_read_csv: Mock pandas read_csv function
            mock_generate_module: Mock generate_behavior_tables module
            runner: CLI test runner
            mock_table_folder: Mock table folder with files
            output_prefix: Custom output prefix to test
        """
        # Arrange
        table_folder, files = mock_table_folder
        mock_read_csv.return_value = pd.DataFrame({"Behavior": ["test_behavior"]})
        mock_generate_module.merge_multiple_behavior_tables.return_value = {
            "test_behavior": ("bout.csv", "bin.csv")
        }

        cmd_args = [
            "merge-multiple-tables",
            "--table-folder",
            str(table_folder),
            "--output-prefix",
            output_prefix,
        ]

        # Act
        result = runner.invoke(app, cmd_args)

        # Assert
        assert result.exit_code == 0

        call_args = mock_generate_module.merge_multiple_behavior_tables.call_args
        assert call_args.kwargs["output_prefix"] == output_prefix

    def test_merge_multiple_tables_nonexistent_folder(self, runner, nonexistent_folder):
        """Test error handling for nonexistent table folder.

        Args:
            runner: CLI test runner
            nonexistent_folder: Path to nonexistent folder
        """
        # Arrange & Act
        result = runner.invoke(
            app,
            [
                "merge-multiple-tables",
                "--table-folder",
                str(nonexistent_folder),
            ],
        )

        # Assert
        assert result.exit_code == 1
        assert f"Error: Table folder not found: {nonexistent_folder}" in result.stdout

    def test_merge_multiple_tables_empty_folder(self, runner, mock_project_folder):
        """Test error handling for folder with no matching files.

        Args:
            runner: CLI test runner
            mock_project_folder: Path to empty folder
        """
        # Arrange & Act
        result = runner.invoke(
            app,
            [
                "merge-multiple-tables",
                "--table-folder",
                str(mock_project_folder),
            ],
        )

        # Assert
        assert result.exit_code == 1
        assert "Error: No table files found matching pattern" in result.stdout

    @patch("pandas.read_csv")
    def test_merge_multiple_tables_invalid_csv_files(
        self, mock_read_csv, runner, mock_table_folder
    ):
        """Test handling of invalid CSV files that cannot be read.

        Args:
            mock_read_csv: Mock pandas read_csv function
            runner: CLI test runner
            mock_table_folder: Mock table folder with files
        """
        # Arrange
        table_folder, files = mock_table_folder
        mock_read_csv.side_effect = pd.errors.EmptyDataError("No data")

        cmd_args = [
            "merge-multiple-tables",
            "--table-folder",
            str(table_folder),
        ]

        # Act
        result = runner.invoke(app, cmd_args)

        # Assert
        assert result.exit_code == 1
        assert "Error: No valid behavior tables found to merge" in result.stdout
        assert "Warning: Could not read behavior from" in result.stdout

    @patch("jabs_postprocess.cli.main.generate_behavior_tables")
    @patch("pandas.read_csv")
    def test_merge_multiple_tables_file_exists_error(
        self, mock_read_csv, mock_generate_module, runner, mock_table_folder
    ):
        """Test handling of FileExistsError during merge operation.

        Args:
            mock_read_csv: Mock pandas read_csv function
            mock_generate_module: Mock generate_behavior_tables module
            runner: CLI test runner
            mock_table_folder: Mock table folder with files
        """
        # Arrange
        table_folder, files = mock_table_folder
        mock_read_csv.return_value = pd.DataFrame({"Behavior": ["test_behavior"]})
        mock_generate_module.merge_multiple_behavior_tables.side_effect = (
            FileExistsError("Output file already exists")
        )

        cmd_args = [
            "merge-multiple-tables",
            "--table-folder",
            str(table_folder),
        ]

        # Act
        result = runner.invoke(app, cmd_args)

        # Assert
        assert result.exit_code == 1
        assert "Error: Output file already exists" in result.stdout
        assert "Use --overwrite to force overwrite" in result.stdout

    @patch("jabs_postprocess.cli.main.generate_behavior_tables")
    @patch("pandas.read_csv")
    def test_merge_multiple_tables_unexpected_error(
        self, mock_read_csv, mock_generate_module, runner, mock_table_folder
    ):
        """Test handling of unexpected errors during merge operation.

        Args:
            mock_read_csv: Mock pandas read_csv function
            mock_generate_module: Mock generate_behavior_tables module
            runner: CLI test runner
            mock_table_folder: Mock table folder with files
        """
        # Arrange
        table_folder, files = mock_table_folder
        mock_read_csv.return_value = pd.DataFrame({"Behavior": ["test_behavior"]})
        mock_generate_module.merge_multiple_behavior_tables.side_effect = Exception(
            "Unexpected error"
        )

        cmd_args = [
            "merge-multiple-tables",
            "--table-folder",
            str(table_folder),
        ]

        # Act
        result = runner.invoke(app, cmd_args)

        # Assert
        assert result.exit_code == 1
        assert "Unexpected error: Unexpected error" in result.stdout

    @patch("jabs_postprocess.cli.main.generate_behavior_tables")
    @patch("pandas.read_csv")
    def test_merge_multiple_tables_output_messages(
        self, mock_read_csv, mock_generate_module, runner, mock_table_folder
    ):
        """Test that appropriate output messages are displayed.

        Args:
            mock_read_csv: Mock pandas read_csv function
            mock_generate_module: Mock generate_behavior_tables module
            runner: CLI test runner
            mock_table_folder: Mock table folder with files
        """
        # Arrange
        table_folder, files = mock_table_folder
        mock_read_csv.return_value = pd.DataFrame({"Behavior": ["test_behavior"]})

        mock_results = {
            "behavior_1": ("bout1.csv", "bin1.csv"),
            "behavior_2": ("bout2.csv", None),  # No bin file
        }
        mock_generate_module.merge_multiple_behavior_tables.return_value = mock_results

        cmd_args = [
            "merge-multiple-tables",
            "--table-folder",
            str(table_folder),
        ]

        # Act
        result = runner.invoke(app, cmd_args)

        # Assert
        assert result.exit_code == 0
        assert "Successfully merged tables for 2 behaviors:" in result.stdout
        assert "behavior_1:" in result.stdout
        assert "Bout table: bout1.csv" in result.stdout
        assert "Bin table: bin1.csv" in result.stdout
        assert "behavior_2:" in result.stdout
        assert "Bout table: bout2.csv" in result.stdout

    @patch("pandas.read_csv")
    def test_merge_multiple_tables_missing_behavior_column(
        self, mock_read_csv, runner, mock_table_folder
    ):
        """Test handling of CSV files without Behavior column.

        Args:
            mock_read_csv: Mock pandas read_csv function
            runner: CLI test runner
            mock_table_folder: Mock table folder with files
        """
        # Arrange
        table_folder, files = mock_table_folder
        mock_read_csv.return_value = pd.DataFrame(
            {"Animal": ["mouse1"], "Frame": [100]}
        )  # No Behavior column

        cmd_args = [
            "merge-multiple-tables",
            "--table-folder",
            str(table_folder),
        ]

        # Act
        result = runner.invoke(app, cmd_args)

        # Assert
        assert result.exit_code == 1
        assert "Warning: Could not read behavior from" in result.stdout
        assert "Error: No valid behavior tables found to merge" in result.stdout

    @patch("jabs_postprocess.cli.main.generate_behavior_tables")
    @patch("pandas.read_csv")
    def test_merge_multiple_tables_behavior_grouping(
        self, mock_read_csv, mock_generate_module, runner, mock_table_folder
    ):
        """Test that files are properly grouped by behavior name.

        Args:
            mock_read_csv: Mock pandas read_csv function
            mock_generate_module: Mock generate_behavior_tables module
            runner: CLI test runner
            mock_table_folder: Mock table folder with files
        """
        # Arrange
        table_folder, files = mock_table_folder

        # Create additional files with same behavior names
        extra_files = []
        for i in range(2):
            file_path = table_folder / f"extra_behavior_0_{i}.csv"
            file_path.write_text("Behavior,Animal,Frame\nbehavior_0,mouse1,100")
            extra_files.append(file_path)

        # Mock pandas to return behaviors based on filename
        def mock_csv_reader(file_path, nrows=None):
            file_name = Path(file_path).name
            if "behavior_0" in file_name:
                return pd.DataFrame({"Behavior": ["behavior_0"]})
            elif "behavior_1" in file_name:
                return pd.DataFrame({"Behavior": ["behavior_1"]})
            else:
                return pd.DataFrame({"Behavior": ["behavior_2"]})

        mock_read_csv.side_effect = mock_csv_reader

        mock_results = {
            "behavior_0": ("bout0.csv", "bin0.csv"),
            "behavior_1": ("bout1.csv", "bin1.csv"),
            "behavior_2": ("bout2.csv", "bin2.csv"),
        }
        mock_generate_module.merge_multiple_behavior_tables.return_value = mock_results

        cmd_args = [
            "merge-multiple-tables",
            "--table-folder",
            str(table_folder),
        ]

        # Act
        result = runner.invoke(app, cmd_args)

        # Assert
        assert result.exit_code == 0

        call_args = mock_generate_module.merge_multiple_behavior_tables.call_args
        table_groups = call_args.kwargs["table_groups"]

        # Verify that behavior_0 has multiple files (original + extras)
        assert len(table_groups["behavior_0"]) == 3  # 1 original + 2 extras
        assert len(table_groups["behavior_1"]) == 1
        assert len(table_groups["behavior_2"]) == 1
