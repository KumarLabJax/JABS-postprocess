"""Utilities for creating video snippets from JABS recordings."""

import os
import sys
from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np

from jabs_postprocess.analysis_utils.clip_utils import (
    read_pose_file,
    write_pose_clip,
    write_video_clip,
)


def get_time_in_frames(location: Union[float, int], unit: str, fps: int = 30) -> int:
    """Converts start and end in arbitrary units into frames.

    Args:
            location: Starting location
            unit: Units of start and end. Choices of frames, seconds, minutes, hours. Allows shortened versions of choices.
            fps: Frames per second used in calculation

    Returns:
            The requested time in frames
    """
    unit_char = unit[0]
    if unit_char == "f":
        return int(location)
    elif unit_char == "s":
        return int(location * fps)
    elif unit_char == "m":
        return int(location * fps * 60)
    elif unit_char == "h":
        return int(location * fps * 60 * 60)
    else:
        raise NotImplementedError(
            f"{unit} is unsupported. Pick from [frame, second, minute, hour]."
        )


def create_video_snippet(
    input_video: Union[str, Path],
    output_video: Union[str, Path],
    start: float = 0,
    end: Optional[float] = None,
    duration: Optional[float] = None,
    time_units: str = "s",
    pose_file: Optional[Union[str, Path]] = None,
    out_pose: Optional[Union[str, Path]] = None,
    render_pose: bool = False,
    behavior_file: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
) -> None:
    """Creates a video snippet from a JABS recording.

    Args:
            input_video: Path to input video for clipping
            output_video: Path to output clipped video
            start: Start time of the clip to produce
            end: End time of the clip to produce (mutually exclusive with duration)
            duration: Duration of the clip to produce (mutually exclusive with end)
            time_units: Units used when clipping. One of ['frame', 'second', 'minute', 'hour']
            pose_file: Optional path to input pose file
            out_pose: Optional path to output clipped pose file
            render_pose: Whether to render pose on the video clip
            behavior_file: Optional path to behavior predictions to render on the video
            overwrite: Whether to overwrite the output video if it already exists

    Returns:
            None

    Raises:
            FileNotFoundError: If input video doesn't exist
            FileNotFoundError: If behavior_file is not None and it doesn't exist.
            FileExistsError: If output video exists and overwrite is False
    """
    # Convert Path objects to strings
    input_video = str(input_video)
    output_video = str(output_video)
    pose_file = str(pose_file) if pose_file else None
    out_pose = str(out_pose) if out_pose else None
    behavior_file = str(behavior_file) if behavior_file else None

    # Check input exists
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Input video not found: {input_video}")

    # Check if output already exists
    if os.path.exists(output_video) and not overwrite:
        raise FileExistsError(
            f"{output_video} exists. Set overwrite=True if you wish to overwrite it."
        )

    # Convert time to frames
    start_frame = get_time_in_frames(start, time_units)

    if duration is None:
        if end is None:
            # If we want the whole video, we need to get a hint at how big it is
            end_frame = sys.maxsize
        else:
            end_frame = get_time_in_frames(end, time_units)
    else:
        end_frame = get_time_in_frames(start + duration, time_units)

    # Get behavior data if requested
    behavior_data = None
    if behavior_file:
        with h5py.File(behavior_file, "r") as f:
            max_frames = f["predictions/predicted_class"].shape[1]
            end_frame = np.clip(end_frame, 0, max_frames - 1)
            behavior_data = f["predictions/predicted_class"][:, start_frame:end_frame]
            # behavior data is stored as [animal, frame]
            behavior_data = np.transpose(behavior_data)

    # Get pose data if requested
    pose_data = None
    if pose_file:
        pose_data = read_pose_file(pose_file)
        max_frames = pose_data.shape[0] + 1
        end_frame = np.clip(end_frame, 0, max_frames - 1)
        pose_data = pose_data[start_frame:end_frame]

    # Create the frame range
    frame_range = range(start_frame, end_frame)

    # Generate the video
    pose_for_video = pose_data if render_pose else None
    write_video_clip(
        input_video, output_video, frame_range, behavior_data, pose_for_video
    )

    # Create the pose clip if requested
    if out_pose and pose_file is not None:
        write_pose_clip(pose_file, out_pose, frame_range)

    return frame_range
