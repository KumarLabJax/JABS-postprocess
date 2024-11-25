import sys
import argparse

from jabs_utils.project_utils import FeatureSettings, JabsProject


def main(argv):
	parser = argparse.ArgumentParser(description='Script that classifies a behavior based on rules.', epilog='Example:\n\tpython3 heuristic_classify.py --project_folder /path/to/project/ --behavior_config "heuristic_classifiers.corner.yaml"', formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument('--project_folder', help='Folder that contains the project with both pose files and feature files', required=True)
	parser.add_argument('--feature_folder', help='Folder where the features are present', default='features')
	parser.add_argument('--behavior_config', help='Configuration file for the heuristic definition', required=True)
	parser.add_argument('--out_prefix', help='File prefix to write output tables (prefix_bouts.csv and prefix_summaries.csv), default=behavior', default='behavior', type=str)
	parser.add_argument('--out_bin_size', help='Time duration used in binning the results, default=60', default=60, type=int)
	parser.add_argument('--overwrite', help='Overwrites output files, default=False', default=False, action='store_true')
	parser.add_argument('--interpolate_size', help='Maximum number of frames in which missing data will be interpolated, default=5', default=5, type=int)
	parser.add_argument('--stitch_gap', help='Number of frames in which frames sequential behavior prediction bouts will be joined, default=5', default=5, type=int)
	parser.add_argument('--min_bout_length', help='Minimum number of frames in which a behavior prediction must be to be considered, default=5', default=5, type=int)
	#

	args = parser.parse_args()

	f_settings = FeatureSettings(args.behavior_config, args.interpolate_size, args.stitch_gap, args.min_bout_length)
	project = JabsProject.from_feature_folder(args.project_folder, f_settings, args.feature_folder)

	bout_table = project.get_bouts()
	bout_out_file = f'{args.out_prefix}_{f_settings.behavior}_bouts.csv'
	bout_table.to_file(bout_out_file, True)

	bin_table = bout_table.to_summary_table(args.out_bin_size)
	bin_out_file = f'{args.out_prefix}_{f_settings.behavior}_summaries.csv'
	bin_table.to_file(bin_out_file, True)


if __name__ == '__main__':
	main(sys.argv[1:])
