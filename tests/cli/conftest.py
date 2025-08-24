"""Fixtures for the CLI tests."""

from typer.testing import CliRunner
import pytest


@pytest.fixture
def runner():
    """Create a Typer CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_project_folder(tmp_path):
    """Create a mock project folder for testing."""
    project_folder = tmp_path / "test_project"
    project_folder.mkdir()
    return project_folder


@pytest.fixture
def mock_feature_folder(tmp_path):
    """Create a mock feature folder for testing."""
    feature_folder = tmp_path / "features"
    feature_folder.mkdir()
    return feature_folder


@pytest.fixture
def mock_table_folder(tmp_path):
    """Create a mock table folder with CSV files for testing."""
    table_folder = tmp_path / "tables"
    table_folder.mkdir()

    # Create mock CSV files
    files = []
    for i in range(3):
        file_path = table_folder / f"behavior_{i}_bout.csv"
        file_path.write_text(f"Behavior,Animal,Frame\nbehavior_{i},mouse1,100")
        files.append(file_path)

    return table_folder, files


@pytest.fixture
def nonexistent_folder(tmp_path):
    """Create path to nonexistent folder for testing."""
    return tmp_path / "nonexistent"


@pytest.fixture
def mock_bout_table_files(tmp_path):
    """Create mock bout table files for testing."""
    files = []
    for i in range(6):  # Create enough files for all test cases
        file_path = tmp_path / f"test_bout_{i}.csv"
        file_path.write_text("mock,csv,content")
        files.append(file_path)
    return files


@pytest.fixture
def nonexistent_files(tmp_path):
    """Create paths to nonexistent files for testing."""
    return [tmp_path / "nonexistent1.csv", tmp_path / "nonexistent2.csv"]
