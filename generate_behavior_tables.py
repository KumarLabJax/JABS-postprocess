import pandas as pd
import numpy as np
import sys
import argparse

from jabs_utils.project_utils import get_behaviors_in_folder, get_predictions_in_folder
from jabs_utils.read_utils import read_experiment_folder, read_activity_folder
from jabs_utils.bin_utils import generate_binned_results
from jabs_utils.write_utils import write_experiment_data

# Generates the 2 prediction tables given an experiment
def generate_behavior_tables(args, behavior: str, linking_dicts: dict={}, experiment_dist_dicts: dict={}):
	# Detect all the experiments in a folder
	exp_folders = get_predictions_in_folder(args.project_folder)
	# Read in all the experiments (RLE format)
	experiment_bout_data = []
	for cur_experiment in exp_folders:
		if cur_experiment in experiment_dist_dicts.keys():
			distance_dict = experiment_dist_dicts[cur_experiment]
		else:
			distance_dict = {}
		if cur_experiment in linking_dicts.keys():
			experiment_data, _ = read_experiment_folder(cur_experiment, behavior, interpolate_size = args.interpolate_size, stitch_bouts = args.stitch_gap, filter_bouts = args.min_bout_length, linking_dict = linking_dicts[cur_experiment], activity_dict = distance_dict)
		else:
			experiment_data, linking_dict = read_experiment_folder(cur_experiment, behavior, interpolate_size = args.interpolate_size, stitch_bouts = args.stitch_gap, filter_bouts = args.min_bout_length, activity_dict = distance_dict)
			linking_dicts[cur_experiment] = linking_dict
		experiment_bout_data.append(experiment_data)
	# Merge experiments into a single project (RLE format)
	experiment_bout_data = pd.concat(experiment_bout_data)
	# Write project bout output
	write_experiment_data(args, behavior, experiment_bout_data, suffix='_' + behavior + '_bouts')
	# Convert project into binned data
	experiment_bin_data = generate_binned_results(experiment_bout_data, args.out_bin_size)
	# Remove empty data
	experiment_bin_data = experiment_bin_data[~np.all(experiment_bin_data[['time_no_pred','time_not_behavior','time_behavior']]==0, axis=1)]
	# Write binned project output
	write_experiment_data(args, behavior, experiment_bin_data, suffix='_' + behavior + '_summaries')
	return experiment_bout_data, linking_dicts

# Generates 3 tables for a given experiment
def generate_activity_tables(args, cm_per_frame_threshold: float, linking_dicts: dict={}, experiment_dist_dicts: dict={}, activity_idx: int=0):
	# Detect all the experiments in a folder
	exp_folders = get_predictions_in_folder(args.project_folder)
	# Read in the activity data for the experiments
	experiment_activity_data = []
	for cur_experiment in exp_folders:
		if cur_experiment in linking_dicts.keys() and cur_experiment in experiment_dist_dicts.keys():
			experiment_data, _, experiment_dists = read_activity_folder(cur_experiment, activity_threshold = cm_per_frame_threshold, interpolate_size = args.interpolate_size, stitch_bouts = args.stitch_gap, filter_bouts = args.min_bout_length, smooth = args.activity_smooth, linking_dict = linking_dicts[cur_experiment], activity_dict = experiment_dist_dicts[cur_experiment])
		# Although probably impossible, still check if we only have distances
		elif cur_experiment in experiment_dist_dicts.keys():
			experiment_data, linking_dict, experiment_dists = read_activity_folder(cur_experiment, activity_threshold = cm_per_frame_threshold, interpolate_size = args.interpolate_size, stitch_bouts = args.stitch_gap, filter_bouts = args.min_bout_length, smooth = args.activity_smooth, activity_dict = experiment_dist_dicts[cur_experiment])
			linking_dicts[cur_experiment] = linking_dict
		else:
			experiment_data, linking_dict, experiment_dists = read_activity_folder(cur_experiment, activity_threshold = cm_per_frame_threshold, interpolate_size = args.interpolate_size, stitch_bouts = args.stitch_gap, filter_bouts = args.min_bout_length, smooth = args.activity_smooth)
			experiment_dist_dicts[cur_experiment] = experiment_dists
			linking_dicts[cur_experiment] = linking_dict
		experiment_activity_data.append(experiment_data)
		experiment_dist_dicts[cur_experiment] = experiment_dists
	# Merge experiments into a single project (RLE format)
	experiment_activity_data = pd.concat(experiment_activity_data)
	# Write project bout output
	if not args.exclude_activity:
		behavior_str = 'Activity > ' + str(np.round(cm_per_frame_threshold*30,3)) + 'cm/s'
		write_experiment_data(args, behavior_str, experiment_activity_data, suffix='_Activity_' + str(activity_idx) + '_bouts')
		# Convert project into binned data
		experiment_bin_data = generate_binned_results(experiment_activity_data, args.out_bin_size)
		# Write binned project output
		write_experiment_data(args, behavior_str, experiment_bin_data, suffix='_Activity_' + str(activity_idx) + '_summaries')
	return experiment_activity_data, linking_dicts, experiment_dist_dicts

def main(argv):
	parser = argparse.ArgumentParser(description='Script that transforms JABS behavior predictions for a project folder into an easier to work with set of files.')
	parser.add_argument('--project_folder', help='Folder that contains the project with both pose files and behavior prediction files', required=True)
	parser.add_argument('--interpolate_size', help='Maximum number of frames in which missing data will be interpolated, default=5', default=5, type=int)
	parser.add_argument('--stitch_gap', help='Number of frames in which frames sequential behavior prediction bouts will be joined, default=5', default=5, type=int)
	parser.add_argument('--min_bout_length', help='Minimum number of frames in which a behavior prediction must be to be considered, default=5', default=5, type=int)
	parser.add_argument('--out_bin_size', help='Time duration used in binning the results, default=60', default=60, type=int)
	parser.add_argument('--behavior', help='Behavior to produce a table for, default=all behaviors in project folder', default=None, type=str, nargs='+')
	parser.add_argument('--exclude_activity', help='Disable output of "Activity" behavior', default=False, action='store_true')
	parser.add_argument('--activity_thresholds', help='Value for active vs inactive behavior bouts in cm/s (default=2.5cm/s)', default=[2.5], type=float, nargs='+')
	parser.add_argument('--activity_smooth', help='Smoothing value for motion calculation in frames (default=5)', default=5, type=int)
	parser.add_argument('--out_prefix', help='File prefix to write output tables (prefix_bouts.csv and prefix_summaries.csv), default=behavior', default='behavior', type=str)
	parser.add_argument('--overwrite', help='Overwrites output files, default=False', default=False, action='store_true')
	#
	args = parser.parse_args()
	# Detect or select which behavior we want tables for
	if args.behavior is None:
		behaviors = get_behaviors_in_folder(args.project_folder)
	else:
		behaviors = list(args.behavior)
	# Calculate activity for use later
	# This also calculates a full linking dict that we can reuse later
	activity_dicts = {}
	linking_dicts = {}
	if not args.exclude_activity:
		print('Calculating activity for ' + ', '.join([str(x) for x in args.activity_thresholds]) + 'cm/s breakpoints')
		for activity_idx, cur_threshold in enumerate(args.activity_thresholds):
			threshold_cm_per_frame = cur_threshold/30
			activity_table, linking_dicts, activity_dicts = generate_activity_tables(args, threshold_cm_per_frame, linking_dicts, activity_dicts, activity_idx = activity_idx)
	# Loop through all behaviors:
	print('Generating behavior tables for behaviors: ' + ', '.join(behaviors) + ' in ' + args.project_folder + '...')
	for behavior in behaviors:
		cur_table, linking_dicts = generate_behavior_tables(args, behavior, linking_dicts, activity_dicts)

if __name__  == '__main__':
	main(sys.argv[1:])
