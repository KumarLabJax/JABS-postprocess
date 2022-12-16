import pandas as pd
import sys
import argparse

from jabs_utils.project_utils import get_behaviors_in_folder, get_predictions_in_folder
from jabs_utils.read_utils import read_experiment_folder
from jabs_utils.bin_utils import generate_binned_results
from jabs_utils.write_utils import write_experiment_data

# Generates the 2 prediction tables given an experiment
def generate_behavior_tables(args, behavior: str):
	# Detect all the experiments in a folder
	exp_folders = get_predictions_in_folder(args.project_folder)
	# Read in all the experiments (RLE format)
	experiment_bout_data = []
	for cur_experiment in exp_folders:
		experiment_data = read_experiment_folder(cur_experiment, behavior, interpolate_size = args.interpolate_size, stitch_bouts = args.stitch_gap, filter_bouts = args.min_bout_length)
		experiment_bout_data.append(experiment_data)
	# Merge experiments into a single project (RLE format)
	experiment_bout_data = pd.concat(experiment_bout_data)
	# Write project bout output
	write_experiment_data(args, behavior, experiment_bout_data, suffix='_' + behavior + '_bouts')
	# Convert project into binned data
	experiment_bin_data = generate_binned_results(experiment_bout_data, args.out_bin_size)
	# Write binned project output
	write_experiment_data(args, behavior, experiment_bin_data, suffix='_' + behavior + '_summaries')
	return experiment_bout_data

def main(argv):
	parser = argparse.ArgumentParser(description='Script that transforms JABS behavior predictions for a project folder into an easier to work with set of files.')
	parser.add_argument('--project_folder', help='Folder that contains the project with both pose files and behavior prediction files', required=True)
	parser.add_argument('--interpolate_size', help='Maximum number of frames in which missing data will be interpolated, default=5', default=5, type=int)
	parser.add_argument('--stitch_gap', help='Number of frames in which frames sequential behavior prediction bouts will be joined, default=5', default=5, type=int)
	parser.add_argument('--min_bout_length', help='Minimum number of frames in which a behavior prediction must be to be considered, default=5', default=5, type=int)
	parser.add_argument('--out_bin_size', help='Time duration used in binning the results, default=60', default=60, type=int)
	parser.add_argument('--behavior', help='Behavior to produce a table for, default=all behaviors in project folder', default=None, type=str)
	parser.add_argument('--out_prefix', help='File prefix to write output tables (prefix_bouts.csv and prefix_summaries.csv), default=behavior', default=None, type=str, required=True)
	parser.add_argument('--overwrite', help='Overwrites output files, default=False', default=False, action='store_true')
	#
	args = parser.parse_args()
	# Detect or select which behavior we want tables for
	if args.behavior is None:
		behaviors = get_behaviors_in_folder(args.project_folder)
	else:
		behaviors = list(args.behavior)
	# Loop through all behaviors:
	print('Generating behavior tables for behaviors: ' + ', '.join(behaviors) + ' in ' + args.project_folder + '...')
	for behavior in behaviors:
		cur_table = generate_behavior_tables(args, behavior)

if __name__  == '__main__':
	main(sys.argv[1:])
