import pandas as pd
import numpy as np
import os
import re

# Helper function to just read in a generic behavior table
def read_postprocess_table(behavior_table: os.path):
	header_data = pd.read_csv(behavior_table, nrows=1)
	df = pd.read_csv(behavior_table, skiprows=2)
	return header_data, df

# Helper function to read in longterm 
def read_ltm_summary_table(behavior_table: os.path, jmcrs_metadata: os.path=None, light_cycle: list[int]=[6,18], timezone: str='America/New_York'):
	# Read in the behavior table data
	header_data, df = read_postprocess_table(behavior_table)
	# Format a bunch of the time data into a more meaningful format
	# str -> datetime object (handling daylight savings)
	df['time'] = pd.to_datetime(df['time']).dt.tz_localize(tz=timezone)
	# Normalize experiment time to be relative to experiment starts
	exp_starts = df.groupby('exp_prefix').agg({'time':min}).reset_index()
	exp_starts.columns = ['exp_prefix','exp_start_time']
	exp_starts['start_date'] = exp_starts['exp_start_time'].dt.normalize()
	df = pd.merge(df, exp_starts, on='exp_prefix', how='left')
	# Actually produce the useful column for plotting
	df['relative_exp_time'] = df['time']-df['start_date']
	# Read in the metadata to add into the main table
	if jmcrs_metadata is not None:
		meta_df = pd.read_excel(jmcrs_metadata)
		meta_df = meta_df[['ExptNumber','Sex','Strain','Location']].drop_duplicates()
		meta_df['Room'] = [x.split(' ')[0] if isinstance(x,str) else ''  for x in meta_df['Location']]
		meta_df['Computer'] = [re.sub('.*(NV[0-9]+).*','\\1',x) if isinstance(x,str) else ''  for x in meta_df['Location']]
		# Note: If you want to drop rows that don't have metadata, change how='inner'
		df = pd.merge(df, meta_df, left_on='exp_prefix', right_on='ExptNumber', how='left')
	# Since the data coming in has "-1" as predictions not assigned an identity, we should drop them
	df = df[df['longterm_idx']!=-1].reset_index(drop=True)
	# Add a light column
	df['lights_on'] = False
	# Lights are on for all MDX/B experiments 6am to 6pm
	df.loc[[x in range(light_cycle[0], light_cycle[1]) for x in df['time'].dt.hour],'lights_on'] = True
	df['zt_time'] = df['time']-pd.Timedelta(str(light_cycle[0]) + ' hour')
	df['zt_exp_time'] = df['relative_exp_time'] - pd.Timedelta(str(light_cycle[0]) + ' hour')
	df['zt_time_hour'] = df['zt_time'].dt.hour
	# Since we want to account for missing data, we should calculate the relative time spent per hour in behavior relative to not behavior
	df['rel_time_behavior'] = df['time_behavior']/(df['time_behavior'] + df['time_not_behavior'])
	# Some cleanup
	# Delete out bins where no data exists
	no_data = np.all(df[['time_no_pred','time_not_behavior','time_behavior']]==0, axis=1)
	df = df[~no_data]
	# Add behavior column
	df['Behavior'] = header_data['Behavior'][0]
	return header_data, df

# Creates a sub-table filtering by time
# Default values remove the first day of the experiment
# df: Dataframe to filter (using time_field field)
# time_field: Column in dataframe to filter using. Must be a relative time field (timedelta64)
# num_hours: Time in hours to assign to the filter cut (relative to start or end)
# filter_from_start: Align time cut to the start (True) or end (False) of experiment
# filter_out_start: Remove time data before (True) or after (False) the specified hours
def filter_experiment_time(df: pd.DataFrame, time_field: str='zt_exp_time', num_hours: int=24, filter_from_start: bool=True, filter_out_start: bool=True):
	# Confirm the datatype of the time field
	col_types = df.dtypes
	assert pd.api.types.is_timedelta64_dtype(col_types[time_field]) or pd.api.types.is_timedelta64_ns_dtype(col_types[time_field])
	# Build the filter based on args
	if filter_from_start:
		filter_field = df[time_field]
	else:
		filter_field = np.max(df[time_field]) - df[time_field]
	if filter_out_start:
		data_to_keep = filter_field >= pd.Timedelta(str(num_hours) + ' hour')
	else:
		data_to_keep = filter_field <= pd.Timedelta(str(num_hours) + ' hour')
	return df[data_to_keep].reset_index(drop=True)

