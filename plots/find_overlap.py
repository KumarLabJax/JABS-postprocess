'''
Script to find overlapping bouts for a given JABS behavior. Used upstream of plot_overlap.py. This code will produce an output csv file used to generate plots.
'''
import pandas as pd
import plotnine as p9
import re
import numpy as np
import mizani
import os
import scipy
import sys
import argparse
from types import SimpleNamespace

from analysis_utils.parse_table import read_ltm_summary_table, filter_experiment_time
from analysis_utils.plots import generate_time_vs_feature_plot
from itertools import chain
from datetime import datetime 

# Run this line before starting up the interactive python session for accessing libraries
# export PYTHONPATH=/JABS-postprocess/
# Alternatively, now we can use the singularity image at /projects/kumar-lab/JABS/JABS-Postprocessing-2023-02-07.sif

def read_ltm_bouts_table(bouts_file, jmcrs_data):
	'''
	Function used to read in long-term monitoring bouts table. Adaptred from read_ltm_summary_table.

	Inputs: bouts_file
		Path to bouts file generated from generate_behavior_tables
	jmcrs_data
		Path to metadata file from JCMS
	'''
	df = pd.read_csv(bouts_file, skiprows=2)
	df['Unique_animal'] = df['longterm_idx'].astype(str) + df['exp_prefix']
	# time to datetime
	df['time'] = pd.to_datetime(df['time'])
	# group experiment to get first timestamp
	experiment_start_time = df.groupby('exp_prefix')['time'].min().to_dict()
	experiment_start_time = df.groupby('exp_prefix')['time'].min().to_dict()
	experiment_start_time_keys = [str(x) for x in experiment_start_time.keys()]
	experiment_start_time_values = [str(x) for x in experiment_start_time.values()]
	experiment_start_time_strings = {}	
	for key in experiment_start_time_keys:
		for value in experiment_start_time_values:
			experiment_start_time_strings[key] = value
			experiment_start_time_values.remove(value)
			break
	df = df[df['is_behavior'] == 1]
	df = df[df['longterm_idx'] != -1]
	df['time'] = df['time'].astype(str)
	df_specific_bout = df[['start','duration','time', 'longterm_idx', 'Unique_animal','exp_prefix','video_name']].copy()
	df_specific_bout['offset'] = df_specific_bout.apply(lambda row: time_to_frame(row['time'], experiment_start_time_strings[str(row['exp_prefix'])], 30), axis=1)
	df_specific_bout['start'] = df_specific_bout.apply(lambda row: row['offset'] + row['start'],axis=1)
	if jmcrs_data is not None:
		meta_df = pd.read_excel(jmcrs_data)
		meta_df = meta_df[['ExptNumber','Sex','Strain','Location']].drop_duplicates()
		meta_df['Room'] = [x.split(' ')[0] if isinstance(x,str) else ''  for x in meta_df['Location']]
		meta_df['Computer'] = [re.sub('.*(NV[0-9]+).*','\\1',x) if isinstance(x,str) else ''  for x in meta_df['Location']]
		df_specific_bout = pd.merge(df_specific_bout, meta_df, left_on='exp_prefix', right_on='ExptNumber', how='left')
	return df_specific_bout


def time_to_frame(t: str, rel_t: str, fps: float):
	"""
	Converts the time to an equivalent frame index relative to rel_t.
	"""
	delta = datetime.strptime(t,'%Y-%m-%d %H:%M:%S')-datetime.strptime(rel_t,'%Y-%m-%d %H:%M:%S')
	return np.int64(delta.total_seconds()*fps)


def find_overlapping_bouts(args):
	'''
    Main function for finding social overlapping behaviors.

    Args:
        args: Namespace of arguments. See `main` for arguments and descriptions.
    '''
	# Read in the summary results
	results_file = args.summary_file
	bouts_file = re.sub('_summaries', '_bouts', results_file)

	jmcrs_data = args.metadata
	header_data, df = read_ltm_summary_table(results_file, jmcrs_metadata=jmcrs_data)
	df_bouts_behavior = read_ltm_bouts_table(bouts_file, jmcrs_data)

	# Experiments to be removed from the dataset
	remove_experiments = ['MDB0003','MDX0008','MDX0017','MDX0093']
	df = df[~np.isin(df['ExptNumber'], remove_experiments)]

	# Delete out bins where no data exists
	no_data = np.all(df[['time_no_pred','time_not_behavior','time_behavior']]==0, axis=1)
	df = df[~no_data].reset_index()

	# Get the average bout length per hour in frames
	df['avg_bout_length'] = df['time_behavior']/df['bout_behavior']

	# Get the absolute start times instead of the relative ones

	# Define columns for the light/dark cycles 
	df['LightCycle'] = df['zt_time_hour'].apply(lambda x: 'Light' if 0 <= x < 12 else 'Dark')
	time_eat_dark = df[df['LightCycle'] == 'Dark']['time_behavior'].sum(numeric_only=True)
	time_eat_light = df[df['LightCycle'] == 'Light']['time_behavior'].sum(numeric_only=True)
	percent_eat_dark = time_eat_dark/(time_eat_light + time_eat_dark)
	df['LightCycle'] = df['zt_time_hour'].apply(lambda x: 'Light' if 0 <= x < 12 else 'Dark')
	df_bouts_behavior['end'] = df_bouts_behavior['start'] + df_bouts_behavior['duration']

	# Literature tells us that approximately 70% of food intake happens in the dark period
	print(f"Percent of time eating during the dark: {percent_eat_dark}")
	#print(df_bouts_behavior)

	two_mice_overlap = pd.DataFrame(columns=['ExptNumber', 'MiceIDs', 'Strain', 'Sex', 'overlap_len', 'bout1_duration', 'bout2_duration'])
	print(f"Total amount of experiments: {len(np.unique(df_bouts_behavior['ExptNumber']))}")
	for cur_exp, exp_df in df_bouts_behavior.groupby('ExptNumber'):
		print(f"Working on experiment {cur_exp}")
		exp_df = exp_df.sort_values('start')
		exp_df.reset_index()
		for cur_mouse, mouse_df in exp_df.groupby('longterm_idx'):
			mouse_df.reset_index()
			if cur_mouse == -1:
				continue
			for i, cur_mouse_bout in mouse_df.iterrows():
				for index, bout in exp_df.iterrows():
					if bout['start'] > cur_mouse_bout['end']: 
						break
					if bout['longterm_idx'] == cur_mouse:
						continue
					elif bout['longterm_idx'] == -1:
						continue
					else:
						if bout['start'] > cur_mouse_bout['start'] and bout['start'] <= cur_mouse_bout['end']:
							mouse_id = bout['longterm_idx']
							if cur_mouse + mouse_id == 1:
								mice = "0 and 1"
							elif cur_mouse + mouse_id == 2:
								mice = "0 and 2"
							elif cur_mouse + mouse_id == 3:
								mice = "1 and 2" 
							else:
								mice = "Error"

							overlap_len = min(bout['end'], cur_mouse_bout['end']) - max(bout['start'], cur_mouse_bout['start'])
							
							to_add = {'ExptNumber': cur_exp, 'MiceIDs': mice, 'Strain': bout['Strain'], 'Sex': bout['Sex'], 'overlap_len': overlap_len, 'bout1_duration': cur_mouse_bout['duration'], 'bout2_duration': bout['duration']}
							tmp_df = pd.DataFrame([to_add])
							two_mice_overlap = pd.concat([two_mice_overlap, tmp_df], ignore_index=True)
							two_mice_overlap.reset_index()
	two_mice_overlap.to_csv(str(args.outpath) + str(args.behavior) + '_overlap.csv')


def main(argv):
	'''Main function that parses arguments and runs minor checks
	
	Args:
		argv: Command-line arguments
	'''
	parser = argparse.ArgumentParser(description='Finds overlaps for behaviors within an arena. Produces csv used to plot overlaps.')
	parser.add_argument('--summary_file', help="Path to the input summary csv file. Bout file should exist in the same folder with same prefix.", type=str)
	parser.add_argument('--outpath', help="Full path to place outfile csv", type=str)
	parser.add_argument('--metadata', help="Path to JCMS metadata sheet. Usually named something like '2023-09-07 TOM_TotalQueryForConfluence.xlsx'")
	parser.add_argument('--behavior', help="Name of behavior being evaluated", type=str, default=None)
	args = parser.parse_args()

	assert os.path.exists(args.summary_file)
	assert os.path.exists(args.outpath)

	find_overlapping_bouts(args)


if __name__ == "__main__":
	main(sys.argv[1:])