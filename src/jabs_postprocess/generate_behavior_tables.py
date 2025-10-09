"""Generates behavior tables from JABS predictions."""

from typing import Dict, List, Optional, Tuple

from pathlib import Path
import pandas as pd
from jabs_postprocess.utils.project_utils import (
    ClassifierSettings,
    JabsProject,
    BoutTable,
    BinTable,
)


def process_behavior_tables(
    project_folder: Path,
    behavior: str,
    out_prefix: str = "behavior",
    out_bin_size: int = 60,
    feature_folder: Optional[Path] = None,
    interpolate_size: Optional[int] = None,
    stitch_gap: Optional[int] = None,
    min_bout_length: Optional[int] = None,
    overwrite: bool = False,
) -> Tuple[str, str]:
    """Generates behavior tables for a specific behavior.

    Args:
            project_folder: Folder containing the JABS project with pose and prediction files
            behavior: Name of the behavior to process
            out_prefix: Prefix for output filenames
            out_bin_size: Time duration used in binning the results
            feature_folder: Optional folder containing feature files
            interpolate_size: Maximum frames for interpolation (None uses default)
            stitch_gap: Frames for stitching behavior bouts (None uses default)
            min_bout_length: Minimum bout length in frames (None uses default)
            overwrite: Whether to overwrite existing files

    Returns:
            Tuple[str, str]: (bout_table_path, bin_table_path) - Paths to the created files
    """
    behavior_settings = ClassifierSettings(
        behavior,
        interpolate_size,
        stitch_gap,
        min_bout_length,
    )

    project = JabsProject.from_prediction_folder(
        project_folder, behavior_settings, feature_folder
    )
    bout_table = project.get_bouts()
    bout_out_file = f"{out_prefix}_{behavior}_bouts.csv"
    bout_table.to_file(bout_out_file, overwrite)

    # Convert project into binned data
    bin_table = bout_table.to_summary_table(out_bin_size)
    bin_out_file = f"{out_prefix}_{behavior}_summaries.csv"
    bin_table.to_file(bin_out_file, overwrite)

    return bout_out_file, bin_out_file


def process_multiple_behaviors(
    project_folder: Path,
    behaviors: List[Dict],
    out_prefix: str = "behavior",
    out_bin_size: int = 60,
    feature_folder: Optional[Path] = None,
    overwrite: bool = False,
) -> List[Tuple[str, str]]:
    """Process multiple behaviors with different settings.

    Args:
            project_folder: Folder containing the JABS project
            behaviors: List of behavior settings dictionaries, each containing at least a 'behavior' key
            out_prefix: Prefix for output filenames
            out_bin_size: Time duration used in binning the results
            feature_folder: Optional folder containing feature files
            overwrite: Whether to overwrite existing files

    Returns:
            List of (bout_table_path, bin_table_path) tuples

    Raises:
            ValueError: If a specified behavior is not found in the project
            KeyError: If a behavior dict is missing the 'behavior' key
    """
    available_behaviors = JabsProject.find_behaviors(project_folder)
    try:
        behavior_names = [b["behavior"] for b in behaviors]
    except KeyError:
        raise KeyError(
            f"Behavior name required in behavior arguments, supplied {behaviors}."
        )

    # Validate behaviors exist
    for behavior in behavior_names:
        if behavior not in available_behaviors:
            raise ValueError(
                f"{behavior} not in experiment folder. Available behaviors: {', '.join(available_behaviors)}."
            )

    results = []
    for behavior_args in behaviors:
        bout_path, bin_path = process_behavior_tables(
            project_folder=project_folder,
            behavior=behavior_args["behavior"],
            out_prefix=out_prefix,
            out_bin_size=out_bin_size,
            feature_folder=feature_folder,
            interpolate_size=behavior_args.get("interpolate_size"),
            stitch_gap=behavior_args.get("stitch_gap"),
            min_bout_length=behavior_args.get("min_bout_length"),
            overwrite=overwrite,
        )
        results.append((bout_path, bin_path))

    return results


def merge_behavior_tables(
    input_tables: List[Path],
    output_prefix: str | None = None,
    overwrite: bool = False,
) -> Tuple[str, str]:
    """Merge multiple behavior tables for the same behavior.

    Args:
            input_tables: List of paths to behavior table files to merge
            output_prefix: Optional prefix for output filenames
            overwrite: Whether to overwrite existing files

    Returns:
            Tuple[str, str]: (merged_bout_table_path, merged_bin_table_path) - Paths to the created files

    Raises:
            FileNotFoundError: If any input table file doesn't exist
            ValueError: If tables have different behaviors or incompatible headers
            FileExistsError: If output files exist and overwrite is False
    """
    if not input_tables:
        raise ValueError("No input tables provided")

    # Validate all input files exist
    for table_path in input_tables:
        if not Path(table_path).exists():
            raise FileNotFoundError(f"Input table not found: {table_path}")

    # Read the first table to determine if it's a bout or bin table and get behavior info
    first_table = BoutTable.from_file(input_tables[0])
    behavior_name = first_table.settings.behavior
    table_type = "bout"

    # Try to determine table type by checking columns
    if "bout_behavior" in first_table.data.columns:
        # This is likely a bin table
        first_table = BinTable.from_file(input_tables[0])
        table_type = "bin"

    # Load all tables and validate they're compatible
    tables = []
    for table_path in input_tables:
        if table_type == "bout":
            table = BoutTable.from_file(table_path)
        else:
            table = BinTable.from_file(table_path)

        # Validate same behavior
        if table.settings.behavior != behavior_name:
            raise ValueError(
                f"Incompatible behaviors: {behavior_name} vs {table.settings.behavior} in {table_path}"
            )

        tables.append(table)

    output_file = behavior_name.replace(" ", "_").lower()
    if output_prefix:
        output_file = f"{output_prefix}_{output_file}"

    # Merge the tables using the existing combine_data method
    if table_type == "bout":
        merged_table = BoutTable.combine_data(tables)
        output_file = f"{output_file}_bouts_merged.csv"
    else:
        merged_table = BinTable.combine_data(tables)
        output_file = f"{output_file}_summaries_merged.csv"

    # Write the merged table
    merged_table.to_file(output_file, overwrite)

    return (
        output_file,
        output_file,
    )  # Return same file for both since we only merged one type


def merge_multiple_behavior_tables(
    table_groups: Dict[str, List[Path]],
    output_prefix: str | None = None,
    overwrite: bool = False,
) -> Dict[str, Tuple[str, str]]:
    """Merge multiple sets of behavior tables grouped by behavior.

    Args:
            table_groups: Dictionary mapping behavior names to lists of table file paths
            output_prefix: Optional prefix for output filenames
            overwrite: Whether to overwrite existing files

    Returns:
            Dictionary mapping behavior names to (bout_table_path, bin_table_path) tuples

    Raises:
            ValueError: If any behavior group is empty
            FileNotFoundError: If any input table file doesn't exist
            FileExistsError: If output files exist and overwrite is False
    """
    results = {}

    for behavior_name, table_paths in table_groups.items():
        if not table_paths:
            raise ValueError(f"No tables provided for behavior: {behavior_name}")

        # Group tables by type (bout vs bin) for this behavior
        bout_tables = []
        bin_tables = []

        for table_path in table_paths:
            data_sample = pd.read_csv(table_path, skiprows=2, nrows=1)

            if data_sample.empty:
                continue  # Skip empty tables

            # Check if it's a bin table (has bout_behavior column) or bout table
            full_data = pd.read_csv(table_path, skiprows=2)
            if "bout_behavior" in full_data.columns:
                bin_tables.append(table_path)
            else:
                bout_tables.append(table_path)

        # Merge bout tables if any exist
        bout_output = None
        if bout_tables:
            bout_output, _ = merge_behavior_tables(
                bout_tables, output_prefix, overwrite
            )

        # Merge bin tables if any exist
        bin_output = None
        if bin_tables:
            bin_output, _ = merge_behavior_tables(
                bin_tables, output_prefix, overwrite
            )

        results[behavior_name] = (bout_output, bin_output)

    return results
