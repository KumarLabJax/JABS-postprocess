from enum import Enum
from pathlib import Path
from typing import Annotated, List, Optional

import pandas as pd
import numpy as np
import typer

from jabs_postprocess import (
    bouts_to_bins,
    compare_gt,
    create_video_snippets,
    generate_behavior_tables,
    heuristic_classify as heuristic_classify_func,
)
from jabs_postprocess.utils.project_utils import BoutTable
from jabs_postprocess.utils.metadata import (
    DEFAULT_INTERPOLATE,
    DEFAULT_MIN_BOUT,
    DEFAULT_STITCH,
)
from jabs_postprocess.cli.utils import load_json

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
    start: Annotated[
        float, typer.Option(help="Start time of the clip to produce")
    ] = 0.0,
    end: Annotated[
        Optional[float],
        typer.Option(
            help="End time of the clip to produce (mutually exclusive with duration)"
        ),
    ] = None,
    duration: Annotated[
        Optional[float],
        typer.Option(
            help="Duration of the clip to produce (mutually exclusive with end)"
        ),
    ] = None,
    time_units: Annotated[
        TimeUnit, typer.Option(help="Units used when clipping")
    ] = TimeUnit.SECOND,
    pose_file: Annotated[
        Optional[Path],
        typer.Option(
            help="Optional path to input pose file. Required to clip pose and render pose."
        ),
    ] = None,
    out_pose: Annotated[
        Optional[Path], typer.Option(help="Write the clipped pose file as well.")
    ] = None,
    render_pose: Annotated[
        bool, typer.Option(help="Render the pose on the video clip.")
    ] = False,
    behavior_file: Annotated[
        Optional[Path],
        typer.Option(
            help="Optional path to behavior predictions. If provided, will render predictions on the video."
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite", "-o", help="Overwrite the output video if it already exists"
        ),
    ] = False,
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
        TimeUnit.HOUR: "hour",
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
            overwrite=overwrite,
        )

        typer.echo(
            f"Successfully created video snippet from {input_video} to {output_video}"
        )
        typer.echo(f"Frame range: {frame_range.start} to {frame_range.stop - 1}")

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
    ground_truth_folder: Path = typer.Option(
        ...,
        help="Path to the JABS project which contains densely annotated ground truth data",
    ),
    prediction_folder: Path = typer.Option(
        ..., help="Path to the folder where behavior predictions were made"
    ),
    stitch_scan: List[float] = typer.Option(
        np.arange(5, 46, 5).tolist(),
        help="List of stitching (time gaps in frames to merge bouts together) values to test",
    ),
    filter_scan: List[float] = typer.Option(
        np.arange(5, 46, 5).tolist(),
        help="List of filter (minimum duration in frames to consider real) values to test",
    ),
    iou_thresholds: List[float] = typer.Option(
        np.round(np.arange(0.05, 1.01, 0.05), 2).tolist(),
        help="List of intersection over union thresholds to scan (will be rounded to 2 decimal places).",
    ),
    filter_ground_truth: bool = typer.Option(
        False,
        help="Apply filters to ground truth data (default is only to filter predictions)",
    ),
    trim_time: Optional[int] = typer.Option(
        None,
        help="Limit the duration in frames of videos for performance (e.g. only the first 2 minutes of a 10 minute video were densely annotated)",
    ),
    results_folder: Path = typer.Option(
        Path.cwd() / "results",
        help="Output folder to save all the result plots and CSVs.",
    ),
):
    """Evaluate classifier performance on densely annotated ground truth data."""

    # Validation
    if not ground_truth_folder.exists():
        raise typer.BadParameter(
            f"Ground truth folder does not exist: {ground_truth_folder}"
        )
    if not prediction_folder.exists():
        raise typer.BadParameter(
            f"Prediction folder does not exist: {prediction_folder}"
        )

    # Call the refactored function with individual parameters
    compare_gt.evaluate_ground_truth(
        behavior=behavior,
        ground_truth_folder=ground_truth_folder,
        prediction_folder=prediction_folder,
        results_folder=results_folder,
        stitch_scan=stitch_scan,
        filter_scan=filter_scan,
        iou_thresholds=iou_thresholds,
        filter_ground_truth=filter_ground_truth,
        trim_time=trim_time,
    )


@app.command()
def generate_tables(
    project_folder: Annotated[
        Path,
        typer.Option(
            help="Folder that contains the project with both pose files and behavior prediction files"
        ),
    ],
    out_prefix: Annotated[
        str,
        typer.Option(
            help="File prefix to write output tables (prefix_bouts.csv and prefix_summaries.csv)"
        ),
    ] = "behavior",
    out_bin_size: Annotated[
        int, typer.Option(help="Time duration used in binning the results")
    ] = 60,
    feature_folder: Annotated[
        Optional[Path],
        typer.Option(
            help="If features were exported, include feature-based characteristics of bouts"
        ),
    ] = None,
    overwrite: Annotated[bool, typer.Option(help="Overwrites output files")] = False,
    add_statistics: Annotated[
        bool,
        typer.Option(
            help="Add bout statistics (count, duration stats, latency) to behavior tables",
        ),
    ] = True,
    behavior_config: Path | None = typer.Option(
        None, "--behavior-config", help="JSON file with behavior configurations"
    ),
    behaviors: List[str] | None = typer.Option(
        None, "--behavior", help="Simple behavior names (uses defaults)"
    ),
):
    """Generate behavior tables from JABS predictions.

    This command transforms behavior predictions from a JABS project into tabular format,
    creating both bout-level and summary tables.

    Example JSON for behavior_config argument:
    {
        "behaviors": [
            {"behavior": "Behavior_1_Name", "interpolate_size": 1},
            {"behavior": "Behavior_2_Name", "stitch_gap": 30, "min_bout_length": 150}
            {"behavior": "Behavior_2_Name",
             "stitch_gap": 30,
             "min_bout_length": 150,
             "interpolate_size": 1
            }
        ]
    }

    The --add-statistics option adds additional columns with bout-level statistics:
    - total_bout_count: Number of behavior bouts per animal
    - avg_bout_duration: Average bout duration per animal
    - bout_duration_std: Standard deviation of bout durations
    - bout_duration_var: Variance of bout durations
    - latency_to_first_bout: Frame number of first behavior bout
    """
    # Convert Path to string
    feature_folder = feature_folder if feature_folder else None

    behavior_args = []

    if behavior_config:
        try:
            config = load_json(behavior_config)
            if not isinstance(config, dict) or "behaviors" not in config:
                raise ValueError("Config must be a JSON object with 'behaviors' key")

            for b in config["behaviors"]:
                behavior_args.append(
                    {
                        "behavior": b["behavior"],
                        "interpolate_size": b.get(
                            "interpolate_size", DEFAULT_INTERPOLATE
                        ),
                        "stitch_gap": b.get("stitch_gap", DEFAULT_STITCH),
                        "min_bout_length": b.get("min_bout_length", DEFAULT_MIN_BOUT),
                    }
                )
        except (ValueError, TypeError, KeyError) as e:
            typer.echo(f"Error loading behavior config: {e}", err=True)
            raise typer.Exit(1)
    elif behaviors:
        for behavior_name in behaviors:
            behavior_args.append(
                {
                    "behavior": behavior_name,
                    "interpolate_size": DEFAULT_INTERPOLATE,
                    "stitch_gap": DEFAULT_STITCH,
                    "min_bout_length": DEFAULT_MIN_BOUT,
                }
            )
    else:
        typer.echo(
            "Error: Must provide either --behavior-config or --behavior options",
            err=True,
        )
        raise typer.Exit(1)

    results = generate_behavior_tables.process_multiple_behaviors(
        project_folder=project_folder,
        behaviors=behavior_args,
        out_prefix=out_prefix,
        out_bin_size=out_bin_size,
        feature_folder=feature_folder,
        overwrite=overwrite,
    )

    # Extract behavior names for output
    behavior_names = [b["behavior"] for b in behavior_args]

    # Add bout statistics if requested
    if add_statistics:
        typer.echo("Adding bout statistics to generated tables...")
        for behavior_name, (bout_file, summary_file) in zip(
            behavior_names, results, strict=True
        ):
            try:
                # Load bout table and add statistics
                bout_table = BoutTable.from_file(bout_file)
                bout_table.add_bout_statistics()
                bout_table.to_file(bout_file, overwrite=True)
                typer.echo(f"  Added statistics to {bout_file}")
            except Exception as e:
                typer.echo(
                    f"  Warning: Failed to add statistics to {bout_file}: {str(e)}"
                )

    for behavior_name, (bout_file, summary_file) in zip(
        behavior_names, results, strict=True
    ):
        typer.echo(f"Generated tables for {behavior_name}:")
        typer.echo(f"  Bout table: {bout_file}")
        typer.echo(f"  Summary table: {summary_file}")
        if add_statistics:
            typer.echo("    ✓ Includes bout statistics")


@app.command()
def heuristic_classify(
    project_folder: Annotated[
        str,
        typer.Option(
            help="Folder that contains the project with both pose files and feature files"
        ),
    ],
    behavior_config: Annotated[
        str, typer.Option(help="Configuration file for the heuristic definition")
    ],
    feature_folder: Annotated[
        str, typer.Option(help="Folder where the features are present")
    ] = "features",
    out_prefix: Annotated[
        str,
        typer.Option(
            help="File prefix to write output tables (prefix_bouts.csv and prefix_summaries.csv)"
        ),
    ] = "behavior",
    out_bin_size: Annotated[
        int, typer.Option(help="Time duration used in binning the results")
    ] = 60,
    overwrite: Annotated[bool, typer.Option(help="Overwrites output files")] = False,
    interpolate_size: Annotated[
        Optional[int],
        typer.Option(
            help=f"Maximum number of frames in which missing data will be interpolated (default: {DEFAULT_INTERPOLATE})"
        ),
    ] = None,
    stitch_gap: Annotated[
        Optional[int],
        typer.Option(
            help=f"Number of frames in which frames sequential behavior prediction bouts will be joined (default: {DEFAULT_STITCH})"
        ),
    ] = None,
    min_bout_length: Annotated[
        Optional[int],
        typer.Option(
            help=f"Minimum number of frames in which a behavior prediction must be to be considered (default: {DEFAULT_MIN_BOUT})"
        ),
    ] = None,
) -> None:
    """Process heuristic classification for behavior analysis."""
    heuristic_classify_func.process_heuristic_classification(
        project_folder=project_folder,
        behavior_config=behavior_config,
        feature_folder=feature_folder,
        out_prefix=out_prefix,
        out_bin_size=out_bin_size,
        overwrite=overwrite,
        interpolate_size=interpolate_size,
        stitch_gap=stitch_gap,
        min_bout_length=min_bout_length,
    )


@app.command()
def merge_tables(
    input_tables: Annotated[
        List[Path],
        typer.Option(
            help="Paths to behavior table files to merge (must be same behavior and table type)"
        ),
    ],
    output_prefix: Annotated[
        str,
        typer.Option(help="File prefix for merged output table"),
    ] = "merged_behavior",
    overwrite: Annotated[bool, typer.Option(help="Overwrites output files")] = False,
):
    """Merge multiple behavior tables of the same type and behavior.

    This command merges behavior tables that contain the same behavior data,
    combining them into a single consolidated table while preserving header information.
    """
    if not input_tables:
        typer.echo("Error: No input tables provided.")
        raise typer.Exit(1)

    # Validate all input files exist
    for table_path in input_tables:
        if not table_path.exists():
            typer.echo(f"Error: Input table not found: {table_path}")
            raise typer.Exit(1)

    try:
        output_file, _ = generate_behavior_tables.merge_behavior_tables(
            input_tables=input_tables,
            output_prefix=output_prefix,
            overwrite=overwrite,
        )

        typer.echo(f"Successfully merged {len(input_tables)} tables:")
        for table in input_tables:
            typer.echo(f"  - {table}")
        typer.echo(f"Output saved to: {output_file}")

    except FileExistsError as e:
        typer.echo(f"Error: {str(e)}")
        typer.echo("Use --overwrite to force overwrite.")
        raise typer.Exit(1)
    except ValueError as e:
        typer.echo(f"Error: {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Unexpected error: {str(e)}")
        raise typer.Exit(1)


@app.command()
def add_bout_statistics(
    input_tables: Annotated[
        List[Path],
        typer.Option(help="Paths to bout table files to add statistics to"),
    ],
    output_suffix: Annotated[
        str,
        typer.Option(help="Suffix to add to output filenames (before .csv)"),
    ] = "_with_stats",
    overwrite: Annotated[
        bool, typer.Option(help="Overwrites input files instead of creating new ones")
    ] = False,
):
    """Add bout statistics to existing behavior tables.

    This command adds bout-level statistics to existing bout table files:
    - total_bout_count: Number of behavior bouts per animal
    - avg_bout_duration: Average bout duration per animal
    - bout_duration_std: Standard deviation of bout durations
    - bout_duration_var: Variance of bout durations
    - latency_to_first_bout: Frame number of first behavior bout

    By default, creates new files with '_with_stats' suffix. Use --overwrite to modify files in-place.
    """
    if not input_tables:
        typer.echo("Error: No input tables provided.")
        raise typer.Exit(1)

    # Validate all input files exist
    for table_path in input_tables:
        if not table_path.exists():
            typer.echo(f"Error: Input table not found: {table_path}")
            raise typer.Exit(1)

    successful_count = 0
    for table_path in input_tables:
        try:
            # Load bout table and add statistics
            bout_table = BoutTable.from_file(table_path)
            bout_table.add_bout_statistics()

            # Determine output path
            if overwrite:
                output_path = table_path
            else:
                # Add suffix before .csv extension
                stem = table_path.stem
                output_path = table_path.parent / f"{stem}{output_suffix}.csv"

            # Save enhanced table
            bout_table.to_file(output_path, overwrite=True)

            typer.echo(f"✓ Added statistics to: {table_path}")
            if not overwrite:
                typer.echo(f"  Output saved to: {output_path}")

            successful_count += 1

        except Exception as e:
            typer.echo(f"✗ Failed to process {table_path}: {str(e)}")

    if successful_count > 0:
        typer.echo(
            f"\nSuccessfully processed {successful_count} out of {len(input_tables)} tables."
        )
        typer.echo("Added statistics columns:")
        typer.echo("  - total_bout_count: Number of behavior bouts per animal")
        typer.echo("  - avg_bout_duration: Average bout duration per animal")
        typer.echo("  - bout_duration_std: Standard deviation of bout durations")
        typer.echo("  - bout_duration_var: Variance of bout durations")
        typer.echo("  - latency_to_first_bout: Frame number of first behavior bout")
    else:
        typer.echo("Error: No tables were successfully processed.")
        raise typer.Exit(1)


@app.command()
def merge_multiple_tables(
    table_folder: Annotated[
        Path,
        typer.Option(
            help="Folder containing behavior table files to merge, grouped by behavior"
        ),
    ],
    behaviors: Annotated[
        Optional[List[str]],
        typer.Option(help="Specific behaviors to merge (default: auto-detect all)"),
    ] = None,
    table_pattern: Annotated[
        str,
        typer.Option(help="File pattern to match behavior tables"),
    ] = "*.csv",
    output_prefix: Annotated[
        str,
        typer.Option(help="File prefix for merged output tables"),
    ] = "merged_behavior",
    overwrite: Annotated[bool, typer.Option(help="Overwrites output files")] = False,
):
    """Merge multiple sets of behavior tables, automatically grouping by behavior.

    This command scans a folder for behavior table files, groups them by behavior name,
    and merges each group separately. Useful for combining results from multiple experiments.
    """
    if not table_folder.exists():
        typer.echo(f"Error: Table folder not found: {table_folder}")
        raise typer.Exit(1)

    # Find all table files matching the pattern
    table_files = list(table_folder.glob(table_pattern))
    if not table_files:
        typer.echo(
            f"Error: No table files found matching pattern '{table_pattern}' in {table_folder}"
        )
        raise typer.Exit(1)

    # Group tables by behavior (extract from filename or header)
    table_groups = {}
    for table_file in table_files:
        try:
            header_data = pd.read_csv(table_file, nrows=1)
            behavior_name = header_data["Behavior"][0]

            # Filter by requested behaviors if specified
            if behaviors and behavior_name not in behaviors:
                continue

            if behavior_name not in table_groups:
                table_groups[behavior_name] = []
            table_groups[behavior_name].append(table_file)

        except (KeyError, pd.errors.EmptyDataError, Exception):
            typer.echo(f"Warning: Could not read behavior from {table_file}, skipping.")
            continue

    if not table_groups:
        typer.echo("Error: No valid behavior tables found to merge.")
        raise typer.Exit(1)

    try:
        results = generate_behavior_tables.merge_multiple_behavior_tables(
            table_groups=table_groups,
            output_prefix=output_prefix,
            overwrite=overwrite,
        )

        typer.echo(f"Successfully merged tables for {len(results)} behaviors:")
        for behavior_name, (bout_file, bin_file) in results.items():
            typer.echo(f"  {behavior_name}:")
            if bout_file:
                typer.echo(f"    Bout table: {bout_file}")
            if bin_file:
                typer.echo(f"    Bin table: {bin_file}")

    except FileExistsError as e:
        typer.echo(f"Error: {str(e)}")
        typer.echo("Use --overwrite to force overwrite.")
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Unexpected error: {str(e)}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
