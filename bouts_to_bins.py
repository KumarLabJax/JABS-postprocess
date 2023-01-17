import pandas as pd
import numpy as np
import sys, re
import argparse

from jabs_utils.bin_utils import generate_binned_results

def main(argv):
	parser = argparse.ArgumentParser(description='Script that runs a bout -> summary transformation.')
	parser.add_argument('--input_file', help='Bout file to transform', required=True)
	#
	args = parser.parse_args()
	bout_file = args.input_file
	experiment_bout_data = pd.read_csv(bout_file, skiprows=2)
	# Convert project into binned data
	experiment_bin_data = generate_binned_results(experiment_bout_data, 60)
	# Write binned project output
	header = pd.read_csv(bout_file, nrows=1)
	out_fname = re.sub('_bouts','_summaries', bout_file)
	with open(out_fname, 'w') as f:
		header.to_csv(f, header=True, index=False)
	# Write the data
	with open(out_fname, 'a') as f:
		experiment_bin_data.to_csv(f, header=True, index=False)

if __name__  == '__main__':
	main(sys.argv[1:])
