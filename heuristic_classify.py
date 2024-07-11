import sys
import argparse

from jabs_utils.project_utils import FeatureRule, FeatureSettings, Relation, JabsProject


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


def main(argv):
	parser = argparse.ArgumentParser(description='Script that classifies a behavior based on rules.', epilog='Example:\n\tpython3 heuristic_classify.py --project_folder /path/to/project/ --behavior "Corner Facing" --feature "" --relation "less than" --threshold 5 --feature ', formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument('--project_folder', help='Folder that contains the project with both pose files and feature files', required=True)
	parser.add_argument('--feature_folder', help='Folder where the features are present', default='features')
	parser.add_argument('--out_prefix', help='File prefix to write output tables (prefix_bouts.csv and prefix_summaries.csv), default=behavior', default='behavior', type=str)
	parser.add_argument('--out_bin_size', help='Time duration used in binning the results, default=60', default=60, type=int)
	parser.add_argument('--overwrite', help='Overwrites output files, default=False', default=False, action='store_true')
	parser.add_argument('--behavior', help='Name of behavior', required=True, type=str)
	parser.add_argument('--interpolate_size', help='Maximum number of frames in which missing data will be interpolated, default=5', default=5, type=int)
	parser.add_argument('--stitch_gap', help='Number of frames in which frames sequential behavior prediction bouts will be joined, default=5', default=5, type=int)
	parser.add_argument('--min_bout_length', help='Minimum number of frames in which a behavior prediction must be to be considered, default=5', default=5, type=int)
	parser.add_argument('--feature_key', help='Feature key', default=argparse.SUPPRESS)
	parser.add_argument('--relation', help='Relationship for thresholding values on feature', default=argparse.SUPPRESS)
	parser.add_argument('--threshold', help='Threshold for feature', default=argparse.SUPPRESS)
	#
	rule_parser = argparse.ArgumentParser(description='Feature rule sub-parser.')
	rule_parser.add_argument('--feature_key', help='Feature key. This argument can be used multiple times for different feature filters and is used in combination with relation and threshold. Multiple rules will be interpreted as an "and" operation.', type=str)
	rule_parser.add_argument('--relation', help='Relationship for thresholding values on features', type=str, required=True, choices=Relation.get_all_options())
	rule_parser.add_argument('--threshold', help='Threshold for feature', type=float, required=True)

	args = parser.parse_args()
	rules = []
	for feature_rule in by_sets(argv, '--feature_key'):
		new_args = rule_parser.parse_args(feature_rule)
		new_rule = FeatureRule(new_args.feature_key, new_args.threshold, new_args.relation)
		rules.append(new_rule)

	f_settings = FeatureSettings(args.behavior, rules, args.interpolate_size, args.stitch_gap, args.min_bout_length)
	project = JabsProject.from_feature_folder(args.project_folder, f_settings, args.feature_folder)

	bout_table = project.get_bouts()
	bout_out_file = f'{args.out_prefix}_bouts.csv'
	bout_table.to_file(bout_out_file, True)

	bin_table = bout_table.to_summary_table()
	bin_out_file = f'{args.out_prefix}_summaries.csv'
	bin_table.to_file(bin_out_file, True)


if __name__ == '__main__':
	main(sys.argv[1:])
