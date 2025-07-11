"""Tests for the Bouts class in jabs_postprocess.utils.project_utils.

This module contains tests for the Bouts class, which represents time segments 
of behavioral events with their durations and associated values. The tests focus
on the following key functionalities:

1. `to_vector`: Converts bout events to a flat vector representation
   - Handles empty bouts properly (using min_frames for vector length)
   - Processes different combinations of starts, durations, and values
   - Supports optional start time shifting for alignment

2. `fill_to_size`: Extends or truncates bouts to a specified frame count
   - Properly handles cases with no initial bouts
   - Creates appropriate representations for absent behaviors

These tests are particularly important because:
- They verify edge cases (empty arrays, different size configurations)
- They ensure consistent behavior with different parameter combinations
- They validate fixes for specific bugs (e.g., ValueError with np.max on empty arrays)

The test cases use parameterization to thoroughly verify behavior across
multiple scenarios with different inputs and expected outputs.
"""

import numpy as np
import pytest

# Assuming Bouts is in jabs_utils.project_utils
# Adjust the import path if your project structure is different
from jabs_postprocess.utils.project_utils import Bouts


def test_bouts_to_vector_empty_bouts_uses_min_frames():
    """
    Tests Bouts.to_vector when the Bouts object has no initial events.
    
    This test verifies a specific edge case fix: when the Bouts object has no
    initial events (empty arrays), the to_vector method should produce a vector
    filled with fill_state of length min_frames without raising exceptions.
    
    Prior to the fix, this would cause a ValueError when np.max was called on an
    empty array. This test ensures the fix works correctly by verifying:
    
    1. The output vector has the expected length (min_frames)
    2. All elements in the vector are set to fill_state
    3. The original Bouts object remains unmodified after the operation
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
    
    This test verifies that when fill_to_size is called on a Bouts object with no
    initial events, it correctly creates a single bout representing the absence of 
    the behavior for the entire duration specified by max_frames.
    
    This is a critical test for the behavior annotation workflow, where:
    - An empty Bouts object typically represents a behavior that was not present
    - After fill_to_size, it should be translated to an explicit representation
      of "not behavior" for the entire duration
    
    The test verifies:
    1. A single bout is created (not multiple segments)
    2. The bout starts at frame 0
    3. The bout spans the entire specified duration (max_frames)
    4. The bout has the correct fill_state value (typically 0 for "not behavior")
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
        # Tests when there are no behavior events - vector should be filled with fill_state
        ([], [], [], 100, -1, False, None),
        
        # Case 2: Initial bouts end before min_frames
        # Tests when behavior ends before the minimum frame count - vector should be extended with fill_state
        ([10], [5], [1], 50, 0, False, None),
        
        # Case 3: Initial bouts end after min_frames
        # Tests when behavior extends beyond min_frames - vector length should be determined by the behavior duration
        ([10], [50], [1], 20, 0, False, None),
        
        # Case 4: Shift start, no initial bouts
        # Tests shift_start=True with no initial bouts - should still produce a vector of min_frames length
        ([], [], [], 70, -1, True, None),
        
        # Case 5: Shift start, initial bouts, min_frames larger than adjusted bout range
        # Tests shift_start with initial bouts where min_frames extends beyond the adjusted bout range
        ([100, 120], [5, 5], [1, 0], 30, -1, True, None),
        
        # Case 6: Shift start, initial bouts, min_frames smaller than adjusted bout range
        # Tests shift_start with initial bouts where min_frames is less than the adjusted bout range
        ([100, 120], [5, 5], [1, 0], 10, -1, True, None),
    ]
)
def test_bouts_to_vector_various_scenarios(initial_starts, initial_durations, initial_values, min_frames, fill_state, shift_start, expected_vector_override):
    """
    Tests Bouts.to_vector with various configurations including empty and non-empty initial bouts,
    and different min_frames settings.
    
    This test comprehensively verifies the to_vector method behavior across different scenarios:
    
    Parameters:
        initial_starts: Starting frames for behaviors
        initial_durations: Duration (in frames) for each behavior
        initial_values: Value/label for each behavior segment
        min_frames: Minimum length of the output vector
        fill_state: Value to fill in non-behavior frames
        shift_start: Whether to shift all behaviors to start from frame 0
        expected_vector_override: Expected full vector (for specific assertion cases)
        
    Test Cases:
        1. Empty bouts: Verifies that min_frames is used when no behaviors exist
        2. Bouts ending before min_frames: Ensures vector is properly extended
        3. Bouts ending after min_frames: Verifies vector extends to cover all behaviors
        4. Shift start with empty bouts: Confirms behavior with no events but shift_start=True
        5. Shift start with min_frames > adjusted range: Tests padding behavior
        6. Shift start with min_frames < adjusted range: Tests when behaviors determine length
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
