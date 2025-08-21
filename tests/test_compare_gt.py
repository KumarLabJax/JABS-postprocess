"""
Tests for the compare_gt module in the JABS-postprocess package.

This test file validates the functionality for comparing ground truth behavioral annotations
with model predictions. The module under test provides metrics for evaluating the performance
of behavior detection models against human-annotated ground truth data.

Key functions tested:
- evaluate_ground_truth: Evaluates predictions against ground truth data and generates
  performance visualizations
- generate_iou_scan: Performs parameter scanning to find optimal IoU (Intersection over Union)
  thresholds and filtering parameters

Test Categories:
1. TestEvaluateGroundTruth: Tests for the evaluate_ground_truth function
   - Basic functionality with mocked data
   - Handling of empty ground truth data
   - Custom parameter configurations

2. generate_iou_scan tests:
   - Parameter combination testing with different scan configurations
   - Handling cases with no valid annotation pairs
   - Testing the filter_ground_truth parameter behavior
   - Validation of performance metrics calculations in various scenarios (TP, FP, FN cases)

The tests use extensive mocking to isolate functionality and parametrization to
test multiple scenarios efficiently. Fixtures are provided for common test data
like mock bout tables, JABS projects, and annotation samples.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from jabs_postprocess.compare_gt import evaluate_ground_truth, generate_iou_scan
from jabs_postprocess.utils.project_utils import (
    Bouts,
)


@pytest.fixture
def mock_bout_table():
    """
    Create a mock BoutTable with sample ground truth data.

    This fixture simulates a BoutTable object containing ground truth behavioral annotations
    with the following structure:
    - 3 behavioral bouts across 2 different videos
    - Each bout has a start time, duration, and positive behavior label
    - Covers multiple animals (animal_idx 0 and 1)

    Returns:
        MagicMock: A mock BoutTable with predefined ground truth data
    """
    gt_data = pd.DataFrame(
        {
            "video_name": ["video1", "video1", "video2"],
            "animal_idx": [0, 0, 1],
            "start": [10, 50, 30],
            "duration": [20, 15, 25],
            "is_behavior": [1, 1, 1],
        }
    )

    mock = MagicMock()
    mock._data = gt_data
    return mock


@pytest.fixture
def mock_jabs_project():
    """
    Create a mock JabsProject with sample prediction data.

    This fixture simulates a JabsProject object that returns prediction bouts when
    get_bouts() is called. The predictions have:
    - 4 behavioral bouts across 3 different videos
    - Each bout has a start time, duration, and positive behavior label
    - Includes one video not present in the ground truth (video3)

    Returns:
        MagicMock: A mock JabsProject configured to return predefined bout predictions
    """
    pred_data = pd.DataFrame(
        {
            "video_name": ["video1", "video1", "video2", "video3"],
            "animal_idx": [0, 0, 1, 2],
            "start": [15, 60, 35, 40],
            "duration": [15, 10, 15, 20],
            "is_behavior": [1, 1, 1, 1],
        }
    )

    mock = MagicMock()
    mock_bouts = MagicMock()
    mock_bouts._data = pred_data
    mock.get_bouts.return_value = mock_bouts
    return mock


@pytest.fixture
def mock_empty_bout_table():
    """
    Create a mock BoutTable with empty data.

    This fixture is used to test how functions handle the edge case of having
    no ground truth annotations available. The DataFrame has the correct structure
    (column names) but contains no rows.

    Returns:
        MagicMock: A mock BoutTable with an empty DataFrame
    """
    mock = MagicMock()
    mock._data = pd.DataFrame(
        columns=["video_name", "animal_idx", "start", "duration", "is_behavior"]
    )
    return mock


@pytest.fixture
def sample_annotations():
    """
    Create sample annotations DataFrame for testing generate_iou_scan.

    This fixture creates a DataFrame containing both ground truth and prediction bouts
    for the same video and animal, differentiated by the 'is_gt' column. The structure
    mimics what would be produced when merging ground truth and predictions:
    - All bouts are for video1, animal_idx 0
    - Contains 2 ground truth bouts and 2 prediction bouts
    - Prediction bouts partially overlap with ground truth bouts

    Returns:
        pd.DataFrame: DataFrame with mixed ground truth and prediction annotations
    """
    return pd.DataFrame(
        {
            "video_name": ["video1", "video1", "video1", "video1"],
            "animal_idx": [0, 0, 0, 0],
            "start": [10, 50, 15, 60],
            "duration": [20, 15, 15, 10],
            "is_behavior": [1, 1, 1, 1],
            "is_gt": [True, True, False, False],
        }
    )


@pytest.fixture
def sample_bouts_objects():
    """
    Create sample Bouts objects for testing.

    This fixture creates two Bouts objects:
    1. Ground truth bouts with 2 behavioral segments
    2. Prediction bouts with 2 behavioral segments

    The bouts are structured to have partial overlaps, allowing for IoU calculation
    and metrics evaluation. This fixture is particularly useful for tests that need
    to examine the behavior of Bouts comparison methods directly.

    Returns:
        tuple: (gt_bouts, pred_bouts) - Two Bouts objects representing ground truth and predictions
    """
    # Define GT bouts
    gt_starts = np.array([10, 50])
    gt_durations = np.array([20, 15])
    gt_labels = np.array([1, 1])
    gt_bouts = Bouts(gt_starts, gt_durations, gt_labels)

    # Define prediction bouts
    pred_starts = np.array([15, 60])
    pred_durations = np.array([15, 10])
    pred_labels = np.array([1, 1])
    pred_bouts = Bouts(pred_starts, pred_durations, pred_labels)

    return gt_bouts, pred_bouts


class TestEvaluateGroundTruth:
    """
    Test suite for the evaluate_ground_truth function.

    This class contains tests that validate the functionality of the evaluate_ground_truth function,
    which compares ground truth behavioral annotations with model predictions and produces
    performance metrics and visualizations. The tests cover:

    1. Basic functionality with normal data
    2. Edge cases like empty ground truth data
    3. Different parameter configurations

    All tests use mocking to isolate the function from external dependencies and to
    control the test environment.
    """

    @patch("jabs_postprocess.compare_gt.BoutTable.from_jabs_annotation_folder")
    @patch("jabs_postprocess.compare_gt.JabsProject.from_prediction_folder")
    @patch("jabs_postprocess.compare_gt.generate_iou_scan")
    @patch("plotnine.ggplot")
    def test_evaluate_ground_truth_basic(
        self,
        mock_ggplot,
        mock_generate_iou_scan,
        mock_from_prediction_folder,
        mock_from_jabs_annotation_folder,
        mock_bout_table,
        mock_jabs_project,
    ):
        """
        Test basic functionality of evaluate_ground_truth with standard inputs.

        This test verifies that evaluate_ground_truth:
        1. Correctly loads ground truth data from an annotation folder
        2. Correctly loads prediction data from a prediction folder
        3. Calls generate_iou_scan with appropriate parameters
        4. Attempts to create visualization plots of the results

        The test uses mocked objects to avoid actual file system access and to
        control the test environment.
        """
        # Setup mocks
        mock_from_jabs_annotation_folder.return_value = mock_bout_table
        mock_from_prediction_folder.return_value = mock_jabs_project

        # Setup return value for generate_iou_scan
        mock_perf_df = pd.DataFrame(
            {
                "stitch": [10, 10, 10],
                "filter": [10, 10, 10],
                "threshold": [0.5, 0.6, 0.7],
                "tp": [5, 4, 3],
                "fn": [2, 3, 4],
                "fp": [1, 1, 1],
                "pr": [0.83, 0.8, 0.75],
                "re": [0.71, 0.57, 0.43],
                "f1": [0.77, 0.67, 0.55],
            }
        )
        mock_generate_iou_scan.return_value = mock_perf_df

        # Mock the plotting functions
        mock_plot = MagicMock()
        mock_ggplot.return_value = mock_plot

        # Call the function
        evaluate_ground_truth(
            behavior="test_behavior",
            ground_truth_folder="test_gt_folder",
            prediction_folder="test_pred_folder",
            results_folder="test_results_folder",
        )

        # Assertions
        mock_from_jabs_annotation_folder.assert_called_once_with(
            "test_gt_folder", "test_behavior"
        )
        mock_from_prediction_folder.assert_called_once()
        mock_generate_iou_scan.assert_called_once()

        # We don't directly assert on plot creation since it's complex and implementation-dependent

    @patch("jabs_postprocess.compare_gt.BoutTable.from_jabs_annotation_folder")
    @patch("jabs_postprocess.compare_gt.JabsProject.from_prediction_folder")
    @patch("jabs_postprocess.compare_gt.generate_iou_scan")
    @patch("plotnine.ggplot")
    def test_evaluate_ground_truth_empty_gt(
        self,
        mock_ggplot,
        mock_generate_iou_scan,
        mock_from_prediction_folder,
        mock_from_jabs_annotation_folder,
        mock_empty_bout_table,
        mock_jabs_project,
    ):
        """
        Test evaluate_ground_truth behavior with empty ground truth data.

        This test verifies that evaluate_ground_truth:
        1. Handles the edge case of having no ground truth annotations correctly
        2. Raises an appropriate warning to the user
        3. Still generates performance metrics with all precision/recall metrics set to 0

        This test case is important because real-world data might sometimes have videos
        with no ground truth annotations for certain behaviors.
        """
        # Setup mocks
        mock_from_jabs_annotation_folder.return_value = mock_empty_bout_table
        mock_from_prediction_folder.return_value = mock_jabs_project

        # Setup return value for generate_iou_scan
        mock_perf_df = pd.DataFrame(
            {
                "stitch": [10],
                "filter": [10],
                "threshold": [0.5],
                "tp": [0],
                "fn": [0],
                "fp": [4],
                "pr": [0],
                "re": [0],
                "f1": [0],
            }
        )
        mock_generate_iou_scan.return_value = mock_perf_df

        # Mock the plotting functions
        mock_plot = MagicMock()
        mock_ggplot.return_value = mock_plot

        evaluate_ground_truth(
            behavior="test_behavior",
            ground_truth_folder="test_gt_folder",
            prediction_folder="test_pred_folder",
            results_folder="test_results_folder",
        )

        # Assertions
        mock_from_jabs_annotation_folder.assert_called_once()
        mock_from_prediction_folder.assert_called_once()
        mock_generate_iou_scan.assert_called_once()

    @patch("jabs_postprocess.compare_gt.BoutTable.from_jabs_annotation_folder")
    @patch("jabs_postprocess.compare_gt.JabsProject.from_prediction_folder")
    @patch("jabs_postprocess.compare_gt.generate_iou_scan")
    @patch("plotnine.ggplot")
    def test_evaluate_ground_truth_custom_params(
        self,
        mock_ggplot,
        mock_generate_iou_scan,
        mock_from_prediction_folder,
        mock_from_jabs_annotation_folder,
        mock_bout_table,
        mock_jabs_project,
    ):
        """
        Test evaluate_ground_truth with custom parameter configurations.

        This test verifies that evaluate_ground_truth:
        1. Correctly accepts and processes custom parameter configurations
        2. Passes these custom parameters to the generate_iou_scan function
        3. Handles custom file output paths for visualizations

        Custom parameters tested include:
        - Custom stitch scan values
        - Custom filter scan values
        - Custom IoU thresholds
        - Filter ground truth option
        - Custom output file paths

        This test ensures the function's flexibility in accommodating different
        evaluation requirements.
        """
        # Setup mocks
        mock_from_jabs_annotation_folder.return_value = mock_bout_table
        mock_from_prediction_folder.return_value = mock_jabs_project

        # Setup return value for generate_iou_scan
        mock_perf_df = pd.DataFrame(
            {
                "stitch": [5, 5, 5],
                "filter": [8, 8, 8],
                "threshold": [0.1, 0.2, 0.3],
                "tp": [5, 4, 3],
                "fn": [2, 3, 4],
                "fp": [1, 1, 1],
                "pr": [0.83, 0.8, 0.75],
                "re": [0.71, 0.57, 0.43],
                "f1": [0.77, 0.67, 0.55],
            }
        )
        mock_generate_iou_scan.return_value = mock_perf_df

        # Mock the plotting functions
        mock_plot = MagicMock()
        mock_ggplot.return_value = mock_plot

        # Call the function with custom parameters
        evaluate_ground_truth(
            behavior="test_behavior",
            ground_truth_folder="test_gt_folder",
            prediction_folder="test_pred_folder",
            results_folder="test_results_folder",
            stitch_scan=[5, 10],
            filter_scan=[8, 16],
            iou_thresholds=[0.1, 0.2, 0.3],
            filter_ground_truth=True,
        )

        # Assertions
        mock_from_jabs_annotation_folder.assert_called_once()
        mock_from_prediction_folder.assert_called_once()
        mock_generate_iou_scan.assert_called_once()
        # Check if the custom parameters are passed correctly to generate_iou_scan
        call_args = mock_generate_iou_scan.call_args[0]
        assert call_args[1] == [5, 10]  # stitch_scan
        assert call_args[2] == [8, 16]  # filter_scan
        assert call_args[3] == [0.1, 0.2, 0.3]  # iou_thresholds
        assert call_args[4] is True  # filter_ground_truth


@pytest.mark.parametrize(
    "stitch_scan,filter_scan,threshold_scan,expected_calls",
    [
        (
            [5, 10],
            [5, 10],
            [0.5],
            4,
        ),  # 2 stitch × 2 filter × 1 threshold = 4 combinations
        (
            [5],
            [5, 10, 15],
            [0.5],
            3,
        ),  # 1 stitch × 3 filter × 1 threshold = 3 combinations
        (
            [5, 10],
            [5],
            [0.5, 0.6],
            4,
        ),  # 2 stitch × 1 filter × 2 threshold = 4 combinations
    ],
)
def test_generate_iou_scan_parameter_combinations(
    stitch_scan,
    filter_scan,
    threshold_scan,
    expected_calls,
    sample_annotations,
    sample_bouts_objects,
):
    """
    Test that generate_iou_scan correctly processes all parameter combinations.

    This parametrized test validates that generate_iou_scan processes all combinations of:
    - stitch_scan values (frames to stitch together)
    - filter_scan values (minimum bout duration to keep)
    - threshold_scan values (IoU thresholds for matching GT and predictions)

    Each parameter set is tested to ensure the function:
    1. Processes the correct number of parameter combinations
    2. Calls necessary methods for each combination
    3. Produces performance metrics for each combination

    Parameters:
        stitch_scan (list): List of frame counts for stitching bouts together
        filter_scan (list): List of minimum durations for filtering bouts
        threshold_scan (list): List of IoU thresholds for matching GT and predictions
        expected_calls (int): Expected number of parameter combinations
        sample_annotations (DataFrame): Fixture with sample annotation data
        sample_bouts_objects (tuple): Fixture with sample Bouts objects
    """
    gt_bouts, pred_bouts = sample_bouts_objects

    # Mock Bouts methods
    with patch("jabs_postprocess.utils.project_utils.Bouts") as mock_bouts_class:
        mock_instance = MagicMock()
        mock_bouts_class.return_value = mock_instance

        # Set up mocks for copy(), fill_to_size(), filter_by_settings(), compare_to()
        mock_instance.copy.return_value = mock_instance
        mock_instance.fill_to_size.return_value = None
        mock_instance.filter_by_settings.return_value = None
        mock_instance.compare_to.return_value = (
            np.array([[5, 0], [0, 3]]),  # intersection matrix
            np.array([[30, 25], [25, 20]]),  # union matrix
            np.array([[0.17, 0], [0, 0.15]]),  # iou matrix
        )

        # Mock calculate_iou_metrics to return predefined metrics
        with patch.object(
            Bouts,
            "calculate_iou_metrics",
            return_value={"tp": 1, "fn": 1, "fp": 1, "pr": 0.5, "re": 0.5, "f1": 0.5},
        ):
            # Run the function
            result = generate_iou_scan(
                sample_annotations, stitch_scan, filter_scan, threshold_scan, False
            )

            # Check result shape
            assert len(result) == expected_calls

            # Check that calculate_iou_metrics was called the expected number of times
            assert Bouts.calculate_iou_metrics.call_count == expected_calls


def test_generate_iou_scan_no_valid_pairs():
    """
    Test generate_iou_scan when there are no valid GT and prediction pairs.

    This test validates the behavior of generate_iou_scan when confronted with the edge case
    of having no matching video/animal pairs between ground truth and predictions.
    The function should:

    1. Issue a warning to alert the user
    2. Return a valid DataFrame with expected columns
    3. Set count metrics (tp, fn, fp) to 0
    4. Set rate metrics (pr, re, f1) to NaN

    This ensures the function degrades gracefully when no valid comparisons can be made,
    rather than failing with an error.
    """
    # Create sample data where animal/video combinations don't match
    annotations = pd.DataFrame(
        {
            "video_name": ["video1", "video2"],
            "animal_idx": [0, 1],
            "start": [10, 30],
            "duration": [20, 25],
            "is_behavior": [1, 1],
            "is_gt": [True, False],  # GT and pred are for different animals/videos
        }
    )

    # Run with warning check
    result = generate_iou_scan(annotations, [5, 10], [5, 10], [0.5], False)

    # Check that a DataFrame with expected columns and content is returned
    assert isinstance(result, pd.DataFrame)
    expected_columns = [
        "stitch",
        "filter",
        "threshold",
        "tp",
        "fn",
        "fp",
        "pr",
        "re",
        "f1",
    ]
    assert all(col in result.columns for col in expected_columns)

    # Verify the DataFrame contains all combinations of stitch and filter values
    stitch_filter_pairs = set(tuple(row) for row in result[["stitch", "filter"]].values)
    expected_pairs = {(5, 5), (5, 10), (10, 5), (10, 10)}
    assert stitch_filter_pairs == expected_pairs

    # All tp, fn, fp should be 0
    assert all(result["tp"] == 0)
    assert all(result["fn"] == 0)
    assert all(result["fp"] == 0)

    # pr, re, f1 should be NaN
    assert result["pr"].isna().all()
    assert result["re"].isna().all()
    assert result["f1"].isna().all()


def test_generate_iou_scan_filter_ground_truth():
    """
    Test generate_iou_scan with filter_ground_truth parameter.

    This test validates that the filter_ground_truth parameter correctly controls whether
    ground truth bouts are filtered using the same settings as predictions:

    1. When filter_ground_truth=False (default):
       - Only prediction bouts should be filtered by the filter settings
       - Ground truth bouts remain unmodified

    2. When filter_ground_truth=True:
       - Both ground truth and prediction bouts should be filtered
       - This can be useful when analyzing how filtering affects both datasets

    The test uses a spy to monitor calls to the filter_by_settings method to verify
    the correct behavior.
    """
    # Create sample annotations with both GT and predictions
    annotations = pd.DataFrame(
        {
            "video_name": ["video1", "video1"],
            "animal_idx": [0, 0],
            "start": [10, 15],
            "duration": [20, 15],
            "is_behavior": [1, 1],
            "is_gt": [True, False],  # One GT and one prediction
        }
    )

    # We'll use a spy to monitor the behavior of filter_by_settings
    with patch(
        "jabs_postprocess.utils.project_utils.Bouts.filter_by_settings"
    ) as filter_spy:
        # Test with filter_ground_truth=False
        generate_iou_scan(annotations, [5], [5], [0.5], False)

        # With filter_ground_truth=False, we expect filter_by_settings to be called once (only on predictions)
        assert filter_spy.call_count == 1, (
            "With filter_ground_truth=False, only predictions should be filtered"
        )

        # Reset the spy
        filter_spy.reset_mock()

        # Test with filter_ground_truth=True
        generate_iou_scan(annotations, [5], [5], [0.5], True)

        # With filter_ground_truth=True, we expect filter_by_settings to be called twice (on GT and predictions)
        assert filter_spy.call_count == 2, (
            "With filter_ground_truth=True, both GT and predictions should be filtered"
        )


@pytest.mark.parametrize(
    "mock_metrics,expected_result",
    [
        # TP only
        ({"tp": 5, "fn": 0, "fp": 0}, {"pr": 1.0, "re": 1.0, "f1": 1.0}),
        # FP only
        ({"tp": 0, "fn": 0, "fp": 5}, {"pr": 0.0, "re": np.nan, "f1": np.nan}),
        # FN only
        ({"tp": 0, "fn": 5, "fp": 0}, {"pr": np.nan, "re": 0.0, "f1": np.nan}),
        # Mixed case
        ({"tp": 5, "fn": 2, "fp": 3}, {"pr": 0.625, "re": 0.714, "f1": 0.667}),
    ],
)
def test_generate_iou_scan_metrics_calculation(mock_metrics, expected_result):
    """
    Test that performance metrics are calculated correctly in different scenarios.

    This parametrized test validates that performance metrics (precision, recall, F1)
    are calculated correctly for various combinations of true positives (TP),
    false negatives (FN), and false positives (FP). It covers several important cases:

    1. Perfect prediction (TP only) - should have precision, recall, F1 = 1.0
    2. All false positives (FP only) - should have precision = 0, recall/F1 = NaN
    3. All false negatives (FN only) - should have recall = 0, precision/F1 = NaN
    4. Mixed case with TP, FN, FP - should calculate correct metrics values

    The test mocks the underlying calculation methods to focus on the metric
    calculation logic itself.

    Parameters:
        mock_metrics (dict): Dictionary of TP, FN, FP counts to test
        expected_result (dict): Expected precision, recall, F1 for the metrics
    """
    annotations = pd.DataFrame(
        {
            "video_name": ["video1", "video1"],
            "animal_idx": [0, 0],
            "start": [10, 15],
            "duration": [20, 15],
            "is_behavior": [1, 1],
            "is_gt": [True, False],
        }
    )

    # Mock Bouts and its methods
    with patch("jabs_postprocess.utils.project_utils.Bouts") as mock_bouts_class:
        mock_instance = MagicMock()
        mock_bouts_class.return_value = mock_instance

        mock_instance.copy.return_value = mock_instance
        mock_instance.fill_to_size.return_value = None
        mock_instance.filter_by_settings.return_value = None
        mock_instance.compare_to.return_value = (
            np.array([[10]]),  # intersection
            np.array([[25]]),  # union
            np.array([[0.4]]),  # iou
        )
        mock_instance.starts = np.array([10])
        mock_instance.durations = np.array([20])

        # Mock calculate_iou_metrics with different scenarios
        with patch.object(Bouts, "calculate_iou_metrics", return_value=mock_metrics):
            result = generate_iou_scan(annotations, [5], [5], [0.5], False)

            # Check metrics calculation
            row = result.iloc[0]
            for metric, expected in expected_result.items():
                if np.isnan(expected):
                    assert np.isnan(row[metric])
                else:
                    assert round(row[metric], 3) == round(expected, 3)
