import pandas as pd
import os


def write_experiment_header(out_file: os.path, project_args, behavior_args):
	"""Writes the output file header.

	Args:
		out_file: File to write the header to
		project_args: Project arguments. Namespace that must include 'project_folder', 'out_bin_size', and 'overwrite' fields
		behavior_args: Behavior arguments. Namespace that must include 'behavior', 'interpolate_size', 'stitch_gap', and 'min_bout_length' fields.
	"""
	header_df = pd.DataFrame({'Project Folder': [project_args.project_folder], 'Behavior': [behavior_args.behavior], 'Interpolate Size': [behavior_args.interpolate_size], 'Stitch Gap': [behavior_args.stitch_gap], 'Min Bout Length': [behavior_args.min_bout_length], 'Out Bin Size': [project_args.out_bin_size]})
	if os.path.exists(out_file) and not project_args.overwrite:
		raise FileExistsError('Out_file ' + str(out_file) + ' exists. Please use --overwrite if you wish to overwrite data.')
	else:
		with open(out_file, 'w') as f:
			header_df.to_csv(f, header=True, index=False)


def write_experiment_data(project_args, behavior_args, data_table, suffix=None):
	"""Writes the experiment data to file with a header.

	Args:
		project_args: Project arguments. Namespace that must include 'out_prefix', 'project_folder', 'out_bin_size', and 'overwrite' fields
		behavior_args: Behavior arguments. Namespace that must include 'behavior', 'interpolate_size', 'stitch_gap', and 'min_bout_length' fields.
		data_table: Pandas table containing the behavioral data to write out
		suffix: Suffix to include with naming the file. Typically used to add either '_bouts' or '_summaries'
	"""
	if suffix is None:
		suffix = ''
	out_file = project_args.out_prefix + suffix + '.csv'
	try:
		# Write the header
		write_experiment_header(out_file, project_args, behavior_args)
		# Write the data
		with open(out_file, 'a') as f:
			data_table.to_csv(f, header=True, index=False)
	except FileExistsError as e:
		print(e)
