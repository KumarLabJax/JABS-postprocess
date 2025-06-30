from typing import Optional

from jabs_postprocess.utils.project_utils import FeatureSettings, JabsProject


def process_heuristic_classification(
	project_folder: str,
	behavior_config: str,
	feature_folder: str = 'features',
	out_prefix: str = 'behavior',
	out_bin_size: int = 60,
	overwrite: bool = False,
	interpolate_size: Optional[int] = None,
	stitch_gap: Optional[int] = None,
	min_bout_length: Optional[int] = None
) -> None:
	"""Process heuristic classification for behavior analysis.
	
	Args:
		project_folder: Folder that contains the project with both pose files and feature files
		behavior_config: Configuration file for the heuristic definition
		feature_folder: Folder where the features are present
		out_prefix: File prefix to write output tables
		out_bin_size: Time duration used in binning the results
		overwrite: Whether to overwrite output files
		interpolate_size: Maximum number of frames in which missing data will be interpolated
		stitch_gap: Number of frames in which frames sequential behavior prediction bouts will be joined
		min_bout_length: Minimum number of frames in which a behavior prediction must be to be considered
	"""
	# Note that defaults here are None, but will be used by constructor if not set in the configuration file.
	f_settings = FeatureSettings(behavior_config, interpolate_size, stitch_gap, min_bout_length)
	project = JabsProject.from_feature_folder(project_folder, f_settings, feature_folder)

	bout_table = project.get_bouts()
	bout_out_file = f'{out_prefix}_{f_settings.behavior}_bouts.csv'
	bout_table.to_file(bout_out_file, overwrite)

	bin_table = bout_table.to_summary_table(out_bin_size)
	bin_out_file = f'{out_prefix}_{f_settings.behavior}_summaries.csv'
	bin_table.to_file(bin_out_file, overwrite)
