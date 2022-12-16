import pandas as pd
import os

# Writes the header of filers used in this script to file
def write_experiment_header(out_file: os.path, args, behavior):
	header_df = pd.DataFrame({'Project Folder': [args.project_folder], 'Behavior': [behavior], 'Interpolate Size':[args.interpolate_size], 'Stitch Gap': [args.stitch_gap], 'Min Bout Length': [args.min_bout_length], 'Out Bin Size': [args.out_bin_size]})
	if os.path.exists(out_file) and not args.overwrite:
		raise FileExistsError('Out_file ' + str(out_file) + ' exists. Please use --overwrite if you wish to overwrite data.')
	else:
		with open(out_file, 'w') as f:
			header_df.to_csv(f, header=True, index=False)

# Writes the experiment data to file with a header
def write_experiment_data(args, behavior, bout_table, suffix=None):
	if suffix is None:
		suffix = ''
	out_file = args.out_prefix + suffix + '.csv'
	try:
		# Write the header
		write_experiment_header(out_file, args, behavior)
		# Write the data
		with open(out_file, 'a') as f:
			bout_table.to_csv(f, header=True, index=False)
	except FileExistsError as e:
		print(e)
