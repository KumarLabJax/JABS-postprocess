"""Generates behavior tables from JABS predictions."""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, Tuple

from jabs_postprocess.utils.project_utils import JabsProject, ClassifierSettings
from jabs_postprocess.utils.metadata import DEFAULT_INTERPOLATE, DEFAULT_STITCH, DEFAULT_MIN_BOUT


def process_behavior_tables(
	project_folder: str,
	behavior: str,
	out_prefix: str = 'behavior',
	out_bin_size: int = 60,
	feature_folder: Optional[str] = None,
	interpolate_size: Optional[int] = None,
	stitch_gap: Optional[int] = None,
	min_bout_length: Optional[int] = None,
	overwrite: bool = False
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
	
	project = JabsProject.from_prediction_folder(project_folder, behavior_settings, feature_folder)
	bout_table = project.get_bouts()
	bout_out_file = f'{out_prefix}_{behavior}_bouts.csv'
	bout_table.to_file(bout_out_file, overwrite)
	
	# Convert project into binned data
	bin_table = bout_table.to_summary_table(out_bin_size)
	bin_out_file = f'{out_prefix}_{behavior}_summaries.csv'
	bin_table.to_file(bin_out_file, overwrite)
	
	return bout_out_file, bin_out_file


def process_multiple_behaviors(
	project_folder: str,
	behaviors: List[Dict],
	out_prefix: str = 'behavior',
	out_bin_size: int = 60,
	feature_folder: Optional[str] = None,
	overwrite: bool = False
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
		behavior_names = [b['behavior'] for b in behaviors]
	except KeyError:
		raise KeyError(f'Behavior name required in behavior arguments, supplied {behaviors}.')
	
	# Validate behaviors exist
	for behavior in behavior_names:
		if behavior not in available_behaviors:
			raise ValueError(f'{behavior} not in experiment folder. Available behaviors: {", ".join(available_behaviors)}.')
	
	results = []
	for behavior_args in behaviors:
		
		bout_path, bin_path = process_behavior_tables(
			project_folder=project_folder,
			behavior=behavior_args['behavior'],
			out_prefix=out_prefix,
			out_bin_size=out_bin_size,
			feature_folder=feature_folder,
			interpolate_size=behavior_args.get('interpolate_size'),
			stitch_gap=behavior_args.get('stitch_gap'),
			min_bout_length=behavior_args.get('min_bout_length'),
			overwrite=overwrite
		)
		results.append((bout_path, bin_path))
	
	return results
