import pandas as pd
import numpy as np
import sys, os
import argparse
from types import SimpleNamespace

from jabs_utils.bout_utils import filter_bouts_by_bouts
from jabs_utils.bin_utils import generate_binned_results
from analysis_utils.parse_table import read_postprocess_table
from jabs_utils.write_utils import write_experiment_data

# Returns a filtered behavior table from file paths
# mode must be 'before', 'after', or 'overlap'
# gap_tolerance is the maximum allowable time gap
# overlap_tolerance is the minimum required time overlap
# discard_bouts controls whether the filter should remove bouts that meet the criteria (True) or preserve bouts meet the criteria and discard ones that do not (False)
def filter_behavior_tables(input_table_file: os.path, filter_table_file: os.path, out_prefix: str, mode: str, gap_tolerance: int, overlap_tolerance: int, discard_bouts: bool=False):
	# Read in the data
	header_in, df_in = read_postprocess_table(input_table_file)
	header_filter, df_filter = read_postprocess_table(filter_table_file)
	if not ('video_name' in list(df_in.keys()) or 'video_name' in list(df_filter.keys())):
		print('At least one input table did not contain the video_name field. Please ensure both ' + input_table_file + ' and ' + filter_table_file + ' are compliant bout tables.')
		exit(1)
	# Filter the data
	filtered_dfs = []
	for video_name, longterm_idx in df_in.groupby(['video_name','longterm_idx']).groups.keys():
		to_filter_bouts_df = df_in[np.logical_and(df_in['video_name']==video_name, df_in['longterm_idx']==longterm_idx)].reset_index(drop=True)
		filter_by_bouts_df = df_filter[np.logical_and(df_filter['video_name']==video_name, df_filter['longterm_idx']==longterm_idx)].reset_index(drop=True)
		# Extract the RLE data
		to_filter_bouts = to_filter_bouts_df[['start','duration','is_behavior']].values.T
		filter_by_bouts = filter_by_bouts_df[['start','duration','is_behavior']].values.T
		# Calculate the filtered results for these
		if mode == 'before':
			starts, durations, states = filter_bouts_by_bouts(to_filter_bouts, filter_by_bouts, inverse_discard = discard_bouts, before_tolerance = gap_tolerance)
		elif mode == 'after':
			starts, durations, states = filter_bouts_by_bouts(to_filter_bouts, filter_by_bouts, inverse_discard = discard_bouts, after_tolerance = gap_tolerance)
		elif mode == 'overlap':
			starts, durations, states = filter_bouts_by_bouts(to_filter_bouts, filter_by_bouts, inverse_discard = discard_bouts, intersect = overlap_tolerance)
		else:
			raise NotImplementedError('Mode ' + mode + ' not supported. Must be before, after, or overlap')
		# Convert back into a df for storage
		filtered_dfs.append(pd.DataFrame({'animal_idx':to_filter_bouts_df['animal_idx'][0], 'start':starts, 'duration':durations, 'is_behavior':states, 'time':to_filter_bouts_df['time'][0], 'exp_prefix':to_filter_bouts_df['exp_prefix'][0], 'video_name':to_filter_bouts_df['video_name'][0], 'longterm_idx':to_filter_bouts_df['longterm_idx'][0]}))
	experiment_bout_data = pd.concat(filtered_dfs)
	# For output, it looks in args for header info, so we can pull them from the headers above
	output_args = SimpleNamespace(out_prefix = out_prefix, project_folder = header_in['Project Folder'][0], interpolate_size = header_in['Interpolate Size'][0], stitch_gap = header_in['Stitch Gap'][0], min_bout_length = header_in['Min Bout Length'][0], out_bin_size = header_in['Out Bin Size'][0])
	behavior = header_in['Behavior'][0] + ' filtered by ' + header_filter['Behavior'][0]
	# Write project bout output
	write_experiment_data(output_args, behavior, experiment_bout_data, suffix='_' + header_in['Behavior'][0] + '_bouts')
	# Convert project into binned data
	experiment_bin_data = generate_binned_results(experiment_bout_data, output_args.out_bin_size)
	# Remove empty data
	experiment_bin_data = experiment_bin_data[~np.all(experiment_bin_data[['time_no_pred','time_not_behavior','time_behavior']]==0, axis=1)]
	# Write binned project output
	write_experiment_data(output_args, behavior, experiment_bin_data, suffix='_' + header_in['Behavior'][0] + '_summaries')

def main(argv):
	parser = argparse.ArgumentParser(description='Script that filters one bouts behavior table with another.')
	parser.add_argument('--in_table', help='Bout table that you wish to filter (*_bouts.csv created from generate_behavior_tables.py)', required=True)
	parser.add_argument('--by_table', help='Bout table that you are comparing to (*_bouts.csv created from generate_behavior_tables.py)', required=True)
	parser.add_argument('--out_prefix', help='File prefix to write output tables (prefix_bouts.csv and prefix_summaries.csv), default=behavior', default='behavior', type=str)
	parser.add_argument('--mode', help='Mode of the filtering. Options: [before, after, overlap]', type=str, choices=['before','after','overlap'], default='overlap')
	parser.add_argument('--gap_tolerance', help='Maximum time gap between bouts when filtering before/after, default=15', type=int, default=15)
	parser.add_argument('--overlap_tolerance', help='Minimum time overlap of bouts when filtering overlaps, default=5', type=int, default=5)
	parser.add_argument('--discard', help='Default behavior is to only keep bouts that meet the criteria. This reverses the logic to discard bouts.', default=False, action='store_true')
	#
	args = parser.parse_args()
	filter_behavior_tables(args.in_table, args.by_table, args.out_prefix, args.mode, args.gap_tolerance, args.overlap_tolerance, args.discard)

if __name__  == '__main__':
	main(sys.argv[1:])
