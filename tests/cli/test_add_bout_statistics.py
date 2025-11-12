"""Unit tests for the add_bout_statistics CLI command.

This test module validates the functionality of the add_bout_statistics CLI command.
The command adds bout-level statistics to existing bout table files, including count,
duration statistics, and latency metrics.

Key functionality tested:
1. Basic statistics addition to single and multiple files
2. Output file handling (overwrite vs new files with suffix)
3. Input validation and error handling
4. File existence checks and error reporting
5. Success/failure counting and reporting
"""

from unittest.mock import MagicMock, patch
import pytest

from jabs_postprocess.cli.main import app


class TestAddBoutStatistics:
    """Test class for the add_bout_statistics CLI command."""

    @pytest.mark.parametrize("file_count", [1, 2, 5])
    @pytest.mark.parametrize("overwrite", [True, False])
    @patch("jabs_postprocess.cli.main.BoutTable")
    def test_add_bout_statistics_basic(
        self,
        mock_bout_table_class,
        runner,
        mock_bout_table_files,
        file_count,
        overwrite,
    ):
        """Test basic bout statistics addition functionality.

        Args:
            mock_bout_table_class: Mock BoutTable class
            runner: CLI test runner
            mock_bout_table_files: List of mock bout table files
            file_count: Number of files to process
            overwrite: Whether to overwrite original files
        """
        # Arrange
        input_files = mock_bout_table_files[:file_count]
        mock_bout_table = MagicMock()
        mock_bout_table_class.from_file.return_value = mock_bout_table

        cmd_args = ["add-bout-statistics"]
        for file_path in input_files:
            cmd_args.extend(["--input-tables", str(file_path)])

        if overwrite:
            cmd_args.append("--overwrite")

        # Act
        result = runner.invoke(app, cmd_args)

        # Assert
        assert result.exit_code == 0

        # Verify BoutTable operations
        assert mock_bout_table_class.from_file.call_count == file_count
        assert mock_bout_table.add_bout_statistics.call_count == file_count
        assert mock_bout_table.to_file.call_count == file_count

        # Verify output messages
        assert (
            f"Successfully processed {file_count} out of {file_count} tables"
            in result.stdout
        )
        assert "total_bout_count: Number of behavior bouts per animal" in result.stdout
        assert "avg_bout_duration: Average bout duration per animal" in result.stdout

        # Verify file operations for each input file
        for i, input_file in enumerate(input_files):
            # Check that from_file was called with correct path
            from_file_calls = [
                call[0][0] for call in mock_bout_table_class.from_file.call_args_list
            ]
            assert input_file in from_file_calls

            # Check output path logic
            to_file_calls = [
                call[0][0] for call in mock_bout_table.to_file.call_args_list
            ]
            if overwrite:
                assert input_file in to_file_calls
            else:
                # Should create new file with suffix
                expected_output = (
                    input_file.parent / f"{input_file.stem}_with_stats.csv"
                )
                assert expected_output in to_file_calls

    @pytest.mark.parametrize("output_suffix", ["_custom", "_stats_v2", "_enhanced"])
    @patch("jabs_postprocess.cli.main.BoutTable")
    def test_add_bout_statistics_custom_suffix(
        self, mock_bout_table_class, runner, mock_bout_table_files, output_suffix
    ):
        """Test bout statistics addition with custom output suffix.

        Args:
            mock_bout_table_class: Mock BoutTable class
            runner: CLI test runner
            mock_bout_table_files: List of mock bout table files
            output_suffix: Custom suffix for output files
        """
        # Arrange
        input_file = mock_bout_table_files[0]
        mock_bout_table = MagicMock()
        mock_bout_table_class.from_file.return_value = mock_bout_table

        cmd_args = [
            "add-bout-statistics",
            "--input-tables",
            str(input_file),
            "--output-suffix",
            output_suffix,
        ]

        # Act
        result = runner.invoke(app, cmd_args)

        # Assert
        assert result.exit_code == 0

        # Verify output file path uses custom suffix
        to_file_calls = mock_bout_table.to_file.call_args_list
        assert len(to_file_calls) == 1
        output_path = to_file_calls[0][0][0]
        expected_output = input_file.parent / f"{input_file.stem}{output_suffix}.csv"
        assert output_path == expected_output

    @pytest.mark.parametrize("missing_file_count", [1, 2])
    def test_add_bout_statistics_missing_files(
        self, runner, nonexistent_files, missing_file_count
    ):
        """Test error handling for missing input files.

        Args:
            runner: CLI test runner
            nonexistent_files: List of nonexistent file paths
            missing_file_count: Number of missing files to test
        """
        # Arrange
        missing_files = nonexistent_files[:missing_file_count]

        cmd_args = ["add-bout-statistics"]
        for file_path in missing_files:
            cmd_args.extend(["--input-tables", str(file_path)])

        # Act
        result = runner.invoke(app, cmd_args)

        # Assert
        assert result.exit_code == 1
        assert "Error: Input table not found" in result.stdout

    @patch("jabs_postprocess.cli.main.BoutTable")
    def test_add_bout_statistics_processing_errors(
        self, mock_bout_table_class, runner, mock_bout_table_files
    ):
        """Test error handling during bout table processing.

        Args:
            mock_bout_table_class: Mock BoutTable class
            runner: CLI test runner
            mock_bout_table_files: List of mock bout table files
        """
        # Arrange
        input_files = mock_bout_table_files[:2]

        # First file succeeds, second fails
        mock_bout_table_success = MagicMock()
        mock_bout_table_class.from_file.side_effect = [
            mock_bout_table_success,
            Exception("Processing error"),
        ]

        cmd_args = ["add-bout-statistics"]
        for file_path in input_files:
            cmd_args.extend(["--input-tables", str(file_path)])

        # Act
        result = runner.invoke(app, cmd_args)

        # Assert
        assert result.exit_code == 0  # Should not exit with error if some succeed
        assert "Added statistics to:" in result.stdout
        assert "Failed to process" in result.stdout
        assert "Successfully processed 1 out of 2 tables" in result.stdout

    @patch("jabs_postprocess.cli.main.BoutTable")
    def test_add_bout_statistics_all_processing_errors(
        self, mock_bout_table_class, runner, mock_bout_table_files
    ):
        """Test error handling when all processing fails.

        Args:
            mock_bout_table_class: Mock BoutTable class
            runner: CLI test runner
            mock_bout_table_files: List of mock bout table files
        """
        # Arrange
        input_file = mock_bout_table_files[0]
        mock_bout_table_class.from_file.side_effect = Exception("Processing error")

        cmd_args = [
            "add-bout-statistics",
            "--input-tables",
            str(input_file),
        ]

        # Act
        result = runner.invoke(app, cmd_args)

        # Assert
        assert result.exit_code == 1
        assert "Failed to process" in result.stdout
        assert "Error: No tables were successfully processed" in result.stdout

    def test_add_bout_statistics_no_input_tables(self, runner):
        """Test error handling when no input tables are provided.

        Args:
            runner: CLI test runner
        """
        # Arrange & Act
        result = runner.invoke(app, ["add-bout-statistics"])

        # Assert - Typer will error before reaching our code due to missing required option
        assert result.exit_code != 0
        assert "Missing option" in result.stderr

    @patch("jabs_postprocess.cli.main.BoutTable")
    def test_add_bout_statistics_output_messages(
        self, mock_bout_table_class, runner, mock_bout_table_files
    ):
        """Test that appropriate output messages are displayed.

        Args:
            mock_bout_table_class: Mock BoutTable class
            runner: CLI test runner
            mock_bout_table_files: List of mock bout table files
        """
        # Arrange
        input_files = mock_bout_table_files[:2]
        mock_bout_table = MagicMock()
        mock_bout_table_class.from_file.return_value = mock_bout_table

        cmd_args = ["add-bout-statistics"]
        for file_path in input_files:
            cmd_args.extend(["--input-tables", str(file_path)])

        # Act
        result = runner.invoke(app, cmd_args)

        # Assert
        assert result.exit_code == 0

        # Check for specific output messages
        for input_file in input_files:
            assert f"Added statistics to: {input_file}" in result.stdout
            assert (
                f"Output saved to: {input_file.parent / (input_file.stem + '_with_stats.csv')}"
                in result.stdout
            )

        # Check for statistics description
        expected_stats = [
            "total_bout_count: Number of behavior bouts per animal",
            "avg_bout_duration: Average bout duration per animal",
            "bout_duration_std: Standard deviation of bout durations",
            "bout_duration_var: Variance of bout durations",
            "latency_to_first_bout: Frame number of first behavior bout",
        ]

        for stat_desc in expected_stats:
            assert stat_desc in result.stdout

    @patch("jabs_postprocess.cli.main.BoutTable")
    def test_add_bout_statistics_overwrite_no_output_message(
        self, mock_bout_table_class, runner, mock_bout_table_files
    ):
        """Test that output path message is not shown when overwriting.

        Args:
            mock_bout_table_class: Mock BoutTable class
            runner: CLI test runner
            mock_bout_table_files: List of mock bout table files
        """
        # Arrange
        input_file = mock_bout_table_files[0]
        mock_bout_table = MagicMock()
        mock_bout_table_class.from_file.return_value = mock_bout_table

        cmd_args = [
            "add-bout-statistics",
            "--input-tables",
            str(input_file),
            "--overwrite",
        ]

        # Act
        result = runner.invoke(app, cmd_args)

        # Assert
        assert result.exit_code == 0
        assert f"Added statistics to: {input_file}" in result.stdout
        assert "Output saved to:" not in result.stdout

    @patch("jabs_postprocess.cli.main.BoutTable")
    def test_add_bout_statistics_mixed_success_failure(
        self, mock_bout_table_class, runner, mock_bout_table_files
    ):
        """Test handling of mixed success and failure scenarios.

        Args:
            mock_bout_table_class: Mock BoutTable class
            runner: CLI test runner
            mock_bout_table_files: List of mock bout table files
        """
        # Arrange
        input_files = mock_bout_table_files[:3]

        # Setup mixed success/failure pattern
        def side_effect(file_path):
            if "test_bout_1" in str(file_path):
                raise Exception("Middle file error")
            return MagicMock()

        mock_bout_table_class.from_file.side_effect = side_effect

        cmd_args = ["add-bout-statistics"]
        for file_path in input_files:
            cmd_args.extend(["--input-tables", str(file_path)])

        # Act
        result = runner.invoke(app, cmd_args)

        # Assert
        assert result.exit_code == 0
        assert "Successfully processed 2 out of 3 tables" in result.stdout
        assert "Added statistics to:" in result.stdout
        assert "Failed to process" in result.stdout
