import numpy as np
import pytest

# Assuming Bouts is in jabs_utils.project_utils
# Adjust the import path if your project structure is different
from jabs_utils.project_utils import Bouts

def test_bouts_to_vector_empty_bouts_uses_min_frames():
    """
    Tests Bouts.to_vector when the Bouts object has no initial events.
    It should produce a vector of fill_state with length min_frames.
    This tests the fix for the ValueError with np.max on an empty array.
    """
    # Arrange
    empty_starts = np.array([])
    empty_durations = np.array([])
    empty_values = np.array([])
    bouts_obj = Bouts(starts=empty_starts, durations=empty_durations, values=empty_values)
    
    min_frames = 100
    fill_state = -1 # A common fill state
    
    # Act
    vector_output = bouts_obj.to_vector(min_frames=min_frames, fill_state=fill_state, shift_start=False)
    
    # Assert
    assert vector_output.shape == (min_frames,), f"Vector length should be {min_frames}"
    assert np.all(vector_output == fill_state), "All vector elements should be equal to fill_state"
    # Ensure original Bouts object is not modified by to_vector
    assert bouts_obj.starts.size == 0, "Original starts should remain empty"
    assert bouts_obj.durations.size == 0, "Original durations should remain empty"
    assert bouts_obj.values.size == 0, "Original values should remain empty"

def test_bouts_fill_to_size_with_no_initial_bouts_represents_not_behavior():
    """
    Tests Bouts.fill_to_size when the Bouts object is initialized with no events.
    This simulates a ground truth file where a behavior was not annotated (implying it didn't occur).
    It should result in a single bout representing the fill_state (e.g., not behavior) for max_frames.
    """
    # Arrange
    empty_starts = np.array([])
    empty_durations = np.array([])
    empty_values = np.array([])
    bouts_obj = Bouts(starts=empty_starts, durations=empty_durations, values=empty_values)
    
    max_frames_fill = 500
    # According to compare_gt.py, gt_obj.fill_to_size(full_duration, 0) is used.
    # So, fill_state 0 represents 'not behavior' in this context.
    fill_state_not_behavior = 0 
    
    # Act
    bouts_obj.fill_to_size(max_frames=max_frames_fill, fill_state=fill_state_not_behavior)
    
    # Assert
    # After fill_to_size, an empty Bouts object should now represent one continuous segment 
    # of fill_state_not_behavior for the duration of max_frames_fill.
    # The rle process within fill_to_size on a uniform vector (created by the fixed to_vector) 
    # should result in one segment.
    assert bouts_obj.starts.shape == (1,), "Should be one start entry after fill_to_size"
    assert bouts_obj.starts[0] == 0, "Start of the 'not behavior' segment should be 0"
    
    assert bouts_obj.durations.shape == (1,), "Should be one duration entry after fill_to_size"
    assert bouts_obj.durations[0] == max_frames_fill, f"Duration of 'not behavior' segment should be {max_frames_fill}"
    
    assert bouts_obj.values.shape == (1,), "Should be one value entry after fill_to_size"
    assert bouts_obj.values[0] == fill_state_not_behavior, f"Value of the segment should be {fill_state_not_behavior}"

@pytest.mark.parametrize(
    "initial_starts, initial_durations, initial_values, min_frames, fill_state, shift_start, expected_vector_override",
    [
        # Case 1: No initial bouts, min_frames determines length
        ([], [], [], 100, -1, False, None),
        # Case 2: Initial bouts end before min_frames
        ([10], [5], [1], 50, 0, False, None),
        # Case 3: Initial bouts end after min_frames
        ([10], [50], [1], 20, 0, False, None),
        # Case 4: Shift start, no initial bouts
        ([], [], [], 70, -1, True, None),
        # Case 5: Shift start, initial bouts, min_frames larger
        ([100, 120], [5, 5], [1, 0], 30, -1, True, None),
        # Case 6: Shift start, initial bouts, min_frames smaller
        ([100, 120], [5, 5], [1, 0], 10, -1, True, None),
    ]
)
def test_bouts_to_vector_various_scenarios(initial_starts, initial_durations, initial_values, min_frames, fill_state, shift_start, expected_vector_override):
    """
    Tests Bouts.to_vector with various configurations including empty and non-empty initial bouts,
    and different min_frames settings.
    """
    # Arrange
    # Ensure inputs are numpy arrays for consistent processing, especially for size checks
    np_initial_starts = np.array(initial_starts)
    np_initial_durations = np.array(initial_durations)
    np_initial_values = np.array(initial_values)

    bouts_obj = Bouts(starts=np_initial_starts, durations=np_initial_durations, values=np_initial_values)

    # Act
    vector_output = bouts_obj.to_vector(min_frames=min_frames, fill_state=fill_state, shift_start=shift_start)

    # Assert
    expected_length = 0
    if np_initial_starts.size == 0: # No initial bouts
        expected_length = min_frames
    elif shift_start:
        # Apply np.min only if array is not empty
        adj_starts = np_initial_starts - np.min(np_initial_starts) 
        max_event_end = np.max(adj_starts + np_initial_durations) 
        expected_length = np.max([max_event_end, min_frames])
    else: # No shift_start, but has initial bouts
        max_event_end = np.max(np_initial_starts + np_initial_durations)
        expected_length = np.max([max_event_end, min_frames])
        
    assert vector_output.shape == (expected_length,), f"Vector length mismatch. Expected {expected_length}, got {vector_output.shape[0]}"

    if expected_vector_override is not None:
        assert np.array_equal(vector_output, expected_vector_override), "Vector content mismatch with override."
    else:
        if np_initial_starts.size == 0:
            assert np.all(vector_output == fill_state), "Empty initial bouts should result in vector of fill_state"
