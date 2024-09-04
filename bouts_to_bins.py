import sys
import re
import argparse

from jabs_utils.project_utils import BoutTable


def main(argv):
	parser = argparse.ArgumentParser(description='Script that runs a bout -> summary transformation.')
	parser.add_argument('--input_file', help='Bout file to transform', required=True)
	#
	args = parser.parse_args()
	bout_file = args.input_file
	experiment_bout_data = BoutTable.from_file(bout_file)
	# Convert project into binned data
	experiment_bin_data = experiment_bout_data.to_summary_table(60)
	out_fname = re.sub('_bouts', '_summaries', bout_file)
	experiment_bin_data.to_file(out_fname)


if __name__ == '__main__':
	main(sys.argv[1:])
