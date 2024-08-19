import pandas as pd
import numpy as np
import sys
import argparse

from jabs_utils.project_utils import JabsProject, ClassifierSettings


def by_sets(iterator, start):
	"""Iterator splitter for parsing arguments.

	Args:
		iterator: iterator, typically args.split()
		start: parameter to split arguments by

	Yields:
		argument groups split by 'start'
	"""
	cur_set = []
	for val in iterator:
		if cur_set and val == start:
			if cur_set[0] == start:
				yield cur_set
			cur_set = [val]
		else:
			cur_set.append(val)
	if cur_set[0] == start:
		yield cur_set


def generate_behavior_tables(args, behavior_args: {}):
	"""Generates the 2 prediction tables given an experimental folder.

	Args:
		args: Namespace of parsed arguments. See `main`.
		behavior_args: Namespace of behavior specific arguments. See `main`.

	Raises:
		KeyError if behavior_args does not contain 'behavior' key.
	"""
	if 'behavior' not in behavior_args.keys():
		raise KeyError(f'Behavior name non optional in behavior arguments, supplied {behavior_args}.')
	behavior_settings = ClassifierSettings(
		behavior_args['behavior'],
		behavior_args.get('interpolate_size', None),
		behavior_args.get('stitch_gap', None),
		behavior_args.get('min_bout_length', None),
	)
	project = JabsProject.from_prediction_folder(args.project_folder, behavior_settings, args.feature_folder)
	bout_table = project.get_bouts()
	bout_out_file = f'{args.out_prefix}_{behavior_args["behavior"]}_bouts.csv'
	bout_table.to_file(bout_out_file, True)
	# Convert project into binned data
	bin_table = bout_table.to_summary_table(args.out_bin_size)
	bin_out_file = f'{args.out_prefix}_{behavior_args["behavior"]}_summaries.csv'
	bin_table.to_file(bin_out_file, True)


def main(argv):
	parser = argparse.ArgumentParser(description='Script that transforms JABS behavior predictions for a project folder into an easier to work with set of files.', epilog='Example:\n\tpython3 generate_behavior_tables.py --project_folder /path/to/project/ --behavior Behavior_1_Name --interpolate_size 1 --behavior Behavior_2_Name --stitch_gap 30 --min_bout_length 150\nwill produce 2 pairs of tables: Behavior_1_Name with interpolate_size=1, and Behavior_2_Name with stitch_gap=30 and min_bout_length=150.', formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument('--project_folder', help='Folder that contains the project with both pose files and behavior prediction files', required=True)
	parser.add_argument('--feature_folder', help='If features were exported at the same time, include feature-based characteristics of bouts. Features are expected to be structured [video_name]/[animal_index]/features.h5 within the folder supplied.', default=None)
	parser.add_argument('--out_prefix', help='File prefix to write output tables (prefix_bouts.csv and prefix_summaries.csv), default=behavior', default='behavior', type=str)
	parser.add_argument('--out_bin_size', help='Time duration used in binning the results, default=60', default=60, type=int)
	parser.add_argument('--overwrite', help='Overwrites output files, default=False', default=False, action='store_true')
	parser.add_argument('--behavior', help='Behaviors to produce a table for. This argument can be used multiple times for different behaviors and is used in combination with interpolate_size, stitch_gap, and min_bout_length (e.g. each behavior will get different values).', nargs='+')
	parser.add_argument('--interpolate_size', help='Maximum number of frames in which missing data will be interpolated, default=5', default=argparse.SUPPRESS)
	parser.add_argument('--stitch_gap', help='Number of frames in which frames sequential behavior prediction bouts will be joined, default=5', default=argparse.SUPPRESS)
	parser.add_argument('--min_bout_length', help='Minimum number of frames in which a behavior prediction must be to be considered, default=5', default=argparse.SUPPRESS)
	#
	behavior_parser = argparse.ArgumentParser(description='Behavior sub-parser.')
	behavior_parser.add_argument('--behavior', help='Behaviors to produce a table for. This argument can be used multiple times for different behaviors and is used in combination with interpolate_size, stitch_gap, and min_bout_length (e.g. each behavior will get different values).', type=str)
	behavior_parser.add_argument('--interpolate_size', help='Maximum number of frames in which missing data will be interpolated, default=5', default=5, type=int)
	behavior_parser.add_argument('--stitch_gap', help='Number of frames in which frames sequential behavior prediction bouts will be joined, default=5', default=5, type=int)
	behavior_parser.add_argument('--min_bout_length', help='Minimum number of frames in which a behavior prediction must be to be considered, default=5', default=5, type=int)

	args = parser.parse_args()
	behavior_args = []
	for behavior_set in by_sets(argv, '--behavior'):
		new_args = behavior_parser.parse_args(behavior_set)
		behavior_args.append(vars(new_args))

	available_behaviors = JabsProject.find_behaviors(args.project_folder)
	behaviors = [x['behavior'] for x in behavior_args]
	for cur_behavior in behaviors:
		if cur_behavior not in available_behaviors:
			raise ValueError(f'{cur_behavior} not in experiment folder. Available behaviors: {", ".join(available_behaviors)}.')

	# Loop through all behaviors:
	print('Generating behavior tables for behaviors: ' + ', '.join(behaviors) + ' in ' + args.project_folder + '...')
	for cur_behavior_args in behavior_args:
		generate_behavior_tables(args, cur_behavior_args)


if __name__ == '__main__':
	main(sys.argv[1:])
