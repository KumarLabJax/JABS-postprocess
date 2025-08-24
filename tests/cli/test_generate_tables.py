"""Unit tests for the generate_tables CLI command.

This test module validates the functionality of the generate_tables CLI command.
The command transforms behavior predictions from a JABS project into tabular format,
creating both bout-level and summary tables.

Key functionality tested:
1. Basic table generation without statistics
2. Table generation with bout statistics
3. Parameter validation and error handling
4. File output and overwrite behavior
5. Integration with underlying processing modules
"""

from unittest.mock import MagicMock, patch
import pytest

from jabs_postprocess.cli.main import app


class TestGenerateTables:
    """Test class for the generate_tables CLI command."""

    @pytest.mark.parametrize("behavior_count", [1, 2, 3])
    @pytest.mark.parametrize("add_statistics", [True, False])
    @patch("jabs_postprocess.cli.main.generate_behavior_tables")
    @patch("jabs_postprocess.cli.main.BoutTable")
    def test_generate_tables_basic(
        self,
        mock_bout_table_class,
        mock_generate_module,
        runner,
        mock_project_folder,
        behavior_count,
        add_statistics,
    ):
        """Test basic table generation functionality.

        Args:
            mock_bout_table_class: Mock BoutTable class
            mock_generate_module: Mock generate_behavior_tables module
            runner: CLI test runner
            mock_project_folder: Mock project directory
            behavior_count: Number of behaviors to test
            add_statistics: Whether to add bout statistics
        """
        # Arrange
        behaviors = [f"behavior{i}" for i in range(behavior_count)]
        out_prefix = "test_output"

        # Mock the process_multiple_behaviors return value
        mock_results = [
            (f"bout_{i}.csv", f"summary_{i}.csv") for i in range(behavior_count)
        ]
        mock_generate_module.process_multiple_behaviors.return_value = mock_results

        # Mock BoutTable for statistics addition
        mock_bout_table = MagicMock()
        mock_bout_table_class.from_file.return_value = mock_bout_table

        # Prepare CLI arguments
        cmd_args = [
            "generate-tables",
            "--project-folder",
            str(mock_project_folder),
            "--out-prefix",
            out_prefix,
        ]

        for behavior in behaviors:
            cmd_args.extend(["--behavior", behavior])

        if add_statistics:
            cmd_args.append("--add-statistics")

        # Act
        result = runner.invoke(app, cmd_args)

        # Assert
        assert result.exit_code == 0

        # Verify process_multiple_behaviors was called with correct parameters
        mock_generate_module.process_multiple_behaviors.assert_called_once()
        call_args = mock_generate_module.process_multiple_behaviors.call_args

        assert call_args.kwargs["project_folder"] == mock_project_folder
        assert call_args.kwargs["out_prefix"] == out_prefix
        assert len(call_args.kwargs["behaviors"]) == behavior_count

        for i, behavior_config in enumerate(call_args.kwargs["behaviors"]):
            assert behavior_config["behavior"] == f"behavior{i}"
            assert "interpolate_size" in behavior_config
            assert "stitch_gap" in behavior_config
            assert "min_bout_length" in behavior_config

        # Verify statistics handling
        if add_statistics:
            assert mock_bout_table_class.from_file.call_count == behavior_count
            assert mock_bout_table.add_bout_statistics.call_count == behavior_count
            assert mock_bout_table.to_file.call_count == behavior_count
        else:
            mock_bout_table_class.from_file.assert_not_called()

    @pytest.mark.parametrize(
        "interpolate_size,stitch_gap,min_bout_length,out_bin_size",
        [
            (None, None, None, 60),
            (30, 5, 10, 120),
            (50, None, 15, 30),
        ],
    )
    @patch("jabs_postprocess.cli.main.generate_behavior_tables")
    def test_generate_tables_with_parameters(
        self,
        mock_generate_module,
        runner,
        mock_project_folder,
        mock_feature_folder,
        interpolate_size,
        stitch_gap,
        min_bout_length,
        out_bin_size,
    ):
        """Test table generation with various parameter combinations.

        Args:
            mock_generate_module: Mock generate_behavior_tables module
            runner: CLI test runner
            mock_project_folder: Mock project directory
            mock_feature_folder: Mock feature directory
            interpolate_size: Interpolation size parameter
            stitch_gap: Stitch gap parameter
            min_bout_length: Minimum bout length parameter
            out_bin_size: Output bin size parameter
        """
        # Arrange
        behavior = "test_behavior"
        mock_generate_module.process_multiple_behaviors.return_value = [
            ("bout.csv", "summary.csv")
        ]

        cmd_args = [
            "generate-tables",
            "--project-folder",
            str(mock_project_folder),
            "--behavior",
            behavior,
            "--feature-folder",
            str(mock_feature_folder),
            "--out-bin-size",
            str(out_bin_size),
        ]

        if interpolate_size is not None:
            cmd_args.extend(["--interpolate-size", str(interpolate_size)])
        if stitch_gap is not None:
            cmd_args.extend(["--stitch-gap", str(stitch_gap)])
        if min_bout_length is not None:
            cmd_args.extend(["--min-bout-length", str(min_bout_length)])

        # Act
        result = runner.invoke(app, cmd_args)

        # Assert
        assert result.exit_code == 0

        call_args = mock_generate_module.process_multiple_behaviors.call_args
        assert call_args.kwargs["feature_folder"] == mock_feature_folder
        assert call_args.kwargs["out_bin_size"] == out_bin_size

        behavior_config = call_args.kwargs["behaviors"][0]
        assert behavior_config["interpolate_size"] == interpolate_size
        assert behavior_config["stitch_gap"] == stitch_gap
        assert behavior_config["min_bout_length"] == min_bout_length

    @pytest.mark.parametrize("overwrite", [True, False])
    @patch("jabs_postprocess.cli.main.generate_behavior_tables")
    def test_generate_tables_overwrite_option(
        self, mock_generate_module, runner, mock_project_folder, overwrite
    ):
        """Test table generation with overwrite option.

        Args:
            mock_generate_module: Mock generate_behavior_tables module
            runner: CLI test runner
            mock_project_folder: Mock project directory
            overwrite: Whether to enable overwrite
        """
        # Arrange
        behavior = "test_behavior"
        mock_generate_module.process_multiple_behaviors.return_value = [
            ("bout.csv", "summary.csv")
        ]

        cmd_args = [
            "generate-tables",
            "--project-folder",
            str(mock_project_folder),
            "--behavior",
            behavior,
        ]

        if overwrite:
            cmd_args.append("--overwrite")

        # Act
        result = runner.invoke(app, cmd_args)

        # Assert
        assert result.exit_code == 0

        call_args = mock_generate_module.process_multiple_behaviors.call_args
        assert call_args.kwargs["overwrite"] == overwrite

    @patch("jabs_postprocess.cli.main.generate_behavior_tables")
    @patch("jabs_postprocess.cli.main.BoutTable")
    def test_generate_tables_statistics_error_handling(
        self, mock_bout_table_class, mock_generate_module, runner, mock_project_folder
    ):
        """Test error handling when adding bout statistics fails.

        Args:
            mock_bout_table_class: Mock BoutTable class
            mock_generate_module: Mock generate_behavior_tables module
            runner: CLI test runner
            mock_project_folder: Mock project directory
        """
        # Arrange
        behavior = "test_behavior"
        mock_generate_module.process_multiple_behaviors.return_value = [
            ("bout.csv", "summary.csv")
        ]

        # Mock BoutTable to raise an exception
        mock_bout_table_class.from_file.side_effect = Exception("Test error")

        cmd_args = [
            "generate-tables",
            "--project-folder",
            str(mock_project_folder),
            "--behavior",
            behavior,
            "--add-statistics",
        ]

        # Act
        result = runner.invoke(app, cmd_args)

        # Assert
        assert result.exit_code == 0  # Command should still succeed
        assert "Warning: Failed to add statistics" in result.stdout

    @patch("jabs_postprocess.cli.main.generate_behavior_tables")
    @patch("jabs_postprocess.cli.main.BoutTable")
    def test_generate_tables_output_messages(
        self, mock_bout_table_class, mock_generate_module, runner, mock_project_folder
    ):
        """Test that appropriate output messages are displayed.

        Args:
            mock_bout_table_class: Mock BoutTable class
            mock_generate_module: Mock generate_behavior_tables module
            runner: CLI test runner
            mock_project_folder: Mock project directory
        """
        # Arrange
        behaviors = ["behavior1", "behavior2"]
        mock_generate_module.process_multiple_behaviors.return_value = [
            ("bout1.csv", "summary1.csv"),
            ("bout2.csv", "summary2.csv"),
        ]

        # Mock BoutTable for statistics addition
        mock_bout_table = MagicMock()
        mock_bout_table_class.from_file.return_value = mock_bout_table

        cmd_args = [
            "generate-tables",
            "--project-folder",
            str(mock_project_folder),
        ]

        for behavior in behaviors:
            cmd_args.extend(["--behavior", behavior])

        cmd_args.append("--add-statistics")

        # Act
        result = runner.invoke(app, cmd_args)

        # Assert
        assert result.exit_code == 0
        assert "Generated tables for behavior1:" in result.stdout
        assert "Generated tables for behavior2:" in result.stdout
        assert "Bout table: bout1.csv" in result.stdout
        assert "Summary table: summary1.csv" in result.stdout
        assert "Includes bout statistics" in result.stdout

    def test_generate_tables_missing_required_args(self, runner):
        """Test that missing required arguments cause appropriate errors.

        Args:
            runner: CLI test runner
        """
        # Arrange & Act
        result = runner.invoke(app, ["generate-tables"])

        # Assert
        assert result.exit_code != 0
        assert "Missing option" in result.stdout

    @patch("jabs_postprocess.cli.main.generate_behavior_tables")
    def test_generate_tables_no_behaviors(
        self, mock_generate_module, runner, mock_project_folder
    ):
        """Test behavior when no behaviors are specified.

        Args:
            mock_generate_module: Mock generate_behavior_tables module
            runner: CLI test runner
            mock_project_folder: Mock project directory
        """
        # Arrange
        cmd_args = [
            "generate-tables",
            "--project-folder",
            str(mock_project_folder),
        ]

        # Act
        result = runner.invoke(app, cmd_args)

        # Assert
        assert result.exit_code != 0
        assert "Missing option" in result.stdout
