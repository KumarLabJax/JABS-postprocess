from enum import Enum
from pathlib import Path
from typing import Annotated, List, Optional

import numpy as np
import typer

from jabs_postprocess import (
    bouts_to_bins,
    compare_gt,
    create_video_snippets,
    generate_behavior_tables,
    heuristic_classify,
)
from jabs_postprocess.utils.metadata import (
    DEFAULT_INTERPOLATE,
    DEFAULT_MIN_BOUT,
    DEFAULT_STITCH,
)

app = typer.Typer()

app.command()(bouts_to_bins.transform_bouts_to_bins)

class TimeUnit(str, Enum):
    """Time units for video snippet creation."""
    FRAME = "frame"
    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"

@app.command()
def create_snippet(
    input_video: Annotated[Path, typer.Option(help="Path to input video for clipping")],
    output_video: Annotated[Path, typer.Option(help="Path to output clipped video")],
    start: Annotated[float, typer.Option(help="Start time of the clip to produce")] = 0.0,
    end: Annotated[Optional[float], typer.Option(help="End time of the clip to produce (mutually exclusive with duration)")] = None,
    duration: Annotated[Optional[float], typer.Option(help="Duration of the clip to produce (mutually exclusive with end)")] = None,
    time_units: Annotated[TimeUnit, typer.Option(help="Units used when clipping")] = TimeUnit.SECOND,
    pose_file: Annotated[Optional[Path], typer.Option(help="Optional path to input pose file. Required to clip pose and render pose.")] = None,
    out_pose: Annotated[Optional[Path], typer.Option(help="Write the clipped pose file as well.")] = None,
    render_pose: Annotated[bool, typer.Option(help="Render the pose on the video clip.")] = False,
    behavior_file: Annotated[Optional[Path], typer.Option(help="Optional path to behavior predictions. If provided, will render predictions on the video.")] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite", "-o", help="Overwrite the output video if it already exists")] = False,
):
    """Create a video snippet from a JABS recording with optional behavior/pose rendering.
    
    This command allows you to extract portions of JABS videos and optionally overlay pose and behavior data.
    """
    # Validate parameters
    if end is not None and duration is not None:
        typer.echo("Error: Cannot specify both end and duration.")
        raise typer.Exit(1)
    
    # Map TimeUnit enum to string representation expected by the function
    unit_map = {
        TimeUnit.FRAME: "frame",
        TimeUnit.SECOND: "second",
        TimeUnit.MINUTE: "minute", 
        TimeUnit.HOUR: "hour"
    }
    
    try:
        frame_range = create_video_snippets.create_video_snippet(
            input_video=input_video,
            output_video=output_video,
            start=start,
            end=end,
            duration=duration,
            time_units=unit_map[time_units],
            pose_file=pose_file,
            out_pose=out_pose,
            render_pose=render_pose,
            behavior_file=behavior_file,
            overwrite=overwrite
        )
        
        typer.echo(f"Successfully created video snippet from {input_video} to {output_video}")
        typer.echo(f"Frame range: {frame_range.start} to {frame_range.stop-1}")
        
        if out_pose and pose_file:
            typer.echo(f"Pose data also saved to {out_pose}")
            
    except FileNotFoundError as e:
        typer.echo(f"Error: {str(e)}")
        raise typer.Exit(1)
    except FileExistsError as e:
        typer.echo(f"Error: {str(e)}")
        typer.echo("Use --overwrite to force overwrite.")
        raise typer.Exit(1)

@app.command()
def evaluate_ground_truth(
    behavior: str = typer.Option(..., help="Behavior to evaluate predictions"),
    ground_truth_folder: Path = typer.Option(..., help="Path to the JABS project which contains densely annotated ground truth data"),
    prediction_folder: Path = typer.Option(..., help="Path to the folder where behavior predictions were made"),
    stitch_scan: List[float] = typer.Option(
        np.arange(5, 46, 5).tolist(), 
        help="List of stitching (time gaps in frames to merge bouts together) values to test"
    ),
    filter_scan: List[float] = typer.Option(
        np.arange(5, 46, 5).tolist(), 
        help="List of filter (minimum duration in frames to consider real) values to test"
    ),
    iou_thresholds: List[float] = typer.Option(
        np.arange(0.05, 1.01, 0.05).tolist(), 
        help="List of intersection over union thresholds to scan"
    ),
    interpolation_size: int = typer.Option(0, help="Number of frames to interpolate missing data"),
    filter_ground_truth: bool = typer.Option(False, help="Apply filters to ground truth data (default is only to filter predictions)"),
    scan_output: Optional[Path] = typer.Option(None, help="Output file to save the filter scan performance plot"),
    bout_output: Optional[Path] = typer.Option(None, help="Output file to save the resulting bout performance plot"),
    trim_time: Optional[int] = typer.Option(None, help="Limit the duration in frames of videos for performance (e.g. only the first 2 minutes of a 10 minute video were densely annotated)"),
    ethogram_output: Optional[Path] = typer.Option(None, help="Output file to save the ethogram plot comparing GT and predictions"),
):
    """Evaluate classifier performance on densely annotated ground truth data."""
    
    # Validation
    if not ground_truth_folder.exists():
        raise typer.BadParameter(f"Ground truth folder does not exist: {ground_truth_folder}")
    if not prediction_folder.exists():
        raise typer.BadParameter(f"Prediction folder does not exist: {prediction_folder}")
    if not scan_output and not bout_output and not ethogram_output:
        typer.echo('Neither scan, bout, nor ethogram outputs were selected, nothing to do. Please use --scan_output, --bout_output, or --ethogram_output.')
        raise typer.Exit()
    
    # Convert Path objects to strings where needed
    ground_truth_folder_str = str(ground_truth_folder)
    prediction_folder_str = str(prediction_folder)
    scan_output_str = str(scan_output) if scan_output else None
    bout_output_str = str(bout_output) if bout_output else None
    ethogram_output_str = str(ethogram_output) if ethogram_output else None
    
    # Call the refactored function with individual parameters
    compare_gt.evaluate_ground_truth(
        behavior=behavior,
        ground_truth_folder=ground_truth_folder_str,
        prediction_folder=prediction_folder_str,
        stitch_scan=stitch_scan,
        filter_scan=filter_scan,
        iou_thresholds=iou_thresholds,
        interpolation_size=interpolation_size,
        filter_ground_truth=filter_ground_truth,
        scan_output=scan_output_str,
        bout_output=bout_output_str,
        trim_time=trim_time,
        ethogram_output=ethogram_output_str,
    )

@app.command()
def generate_tables(
    project_folder: Annotated[Path, typer.Option(help="Folder that contains the project with both pose files and behavior prediction files")],
    behavior: Annotated[List[str], typer.Option(help="Behavior(s) to produce table(s) for")],
    out_prefix: Annotated[str, typer.Option(help="File prefix to write output tables (prefix_bouts.csv and prefix_summaries.csv)")] = "behavior",
    out_bin_size: Annotated[int, typer.Option(help="Time duration used in binning the results")] = 60,
    feature_folder: Annotated[Optional[Path], typer.Option(help="If features were exported, include feature-based characteristics of bouts")] = None,
    interpolate_size: Annotated[Optional[int], typer.Option(help=f"Maximum number of frames in which missing data will be interpolated (default: {DEFAULT_INTERPOLATE})")] = None,
    stitch_gap: Annotated[Optional[int], typer.Option(help=f"Number of frames in which sequential behavior prediction bouts will be joined (default: {DEFAULT_STITCH})")] = None,
    min_bout_length: Annotated[Optional[int], typer.Option(help=f"Minimum number of frames in which a behavior prediction must be to be considered (default: {DEFAULT_MIN_BOUT})")] = None,
    overwrite: Annotated[bool, typer.Option(help="Overwrites output files")] = False,
):
    """Generate behavior tables from JABS predictions.
    
    This command transforms behavior predictions from a JABS project into tabular format,
    creating both bout-level and summary tables.
    """
    # Convert Path to string
    project_folder_str = str(project_folder)
    feature_folder_str = str(feature_folder) if feature_folder else None
    
    behaviors = []
    for behavior_name in behavior:
        behavior_config = {
            "behavior": behavior_name,
            "interpolate_size": interpolate_size,
            "stitch_gap": stitch_gap,
            "min_bout_length": min_bout_length
        }
        behaviors.append(behavior_config)
    
    results = generate_behavior_tables.process_multiple_behaviors(
        project_folder=project_folder_str,
        behaviors=behaviors,
        out_prefix=out_prefix,
        out_bin_size=out_bin_size,
        feature_folder=feature_folder_str,
        overwrite=overwrite
    )
    
    for behavior_name, (bout_file, summary_file) in zip(behavior, results, strict=True):
        typer.echo(f"Generated tables for {behavior_name}:")
        typer.echo(f"  Bout table: {bout_file}")
        typer.echo(f"  Summary table: {summary_file}")

@app.command()
def heuristic_classify(
    project_folder: Annotated[str, typer.Option(help="Folder that contains the project with both pose files and feature files")],
    behavior_config: Annotated[str, typer.Option(help="Configuration file for the heuristic definition")],
    feature_folder: Annotated[str, typer.Option(help="Folder where the features are present")] = "features",
    out_prefix: Annotated[str, typer.Option(help="File prefix to write output tables (prefix_bouts.csv and prefix_summaries.csv)")] = "behavior",
    out_bin_size: Annotated[int, typer.Option(help="Time duration used in binning the results")] = 60,
    overwrite: Annotated[bool, typer.Option(help="Overwrites output files")] = False,
    interpolate_size: Annotated[Optional[int], typer.Option(help=f"Maximum number of frames in which missing data will be interpolated (default: {DEFAULT_INTERPOLATE})")] = None,
    stitch_gap: Annotated[Optional[int], typer.Option(help=f"Number of frames in which frames sequential behavior prediction bouts will be joined (default: {DEFAULT_STITCH})")] = None,
    min_bout_length: Annotated[Optional[int], typer.Option(help=f"Minimum number of frames in which a behavior prediction must be to be considered (default: {DEFAULT_MIN_BOUT})")] = None,
) -> None:
    """Process heuristic classification for behavior analysis."""
    heuristic_classify.process_heuristic_classification(
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


if __name__ == "__main__":
    app()