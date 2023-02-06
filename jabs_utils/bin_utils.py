import numpy as np
import pandas as pd
from datetime import datetime

# Transforms raw data per-experiment into binned results
# Helper function that runs on an entire experiment dataframe
def generate_binned_results(df: pd.DataFrame, bin_size_minutes: int=60):
	grouped_df = df.groupby(['exp_prefix','longterm_idx'])
	all_results = []
	for cur_group, cur_data in grouped_df:
		binned_results = get_animal_results(cur_data, bin_size_minutes)
		binned_results['exp_prefix'], binned_results['longterm_idx'] = cur_group
		all_results.append(binned_results)
	all_results = pd.concat(all_results)
	return all_results

# Function that transforms event data into binned time format summaries
def get_animal_results(event_df: pd.DataFrame, bin_size_minutes: int=60, fps=30):
	# Get the range that the experiment spans
	try:
		# TODO: Add support for different sized experiment blocks (re-use block below to make an end time that is adjusted per-video)
		start_time = round_hour(datetime.strptime(min(event_df['time']),'%Y-%m-%d %H:%M:%S'))
		end_time = round_hour(datetime.strptime(max(event_df['time']),'%Y-%m-%d %H:%M:%S'), up=True)
	# Timestamp doesn't exist. Make up some. This assumes only 1 video exists and just makes up timestamps based on the available bout data.
	except:
		start_time_str = '1970-01-01 00:00:00'
		start_time = round_hour(datetime.strptime(start_time_str,'%Y-%m-%d %H:%M:%S'))
		num_hours = np.max(event_df['start'] + event_df['duration'])/fps/60//60 + 1
		end_time_str = start_time_str
		for _ in np.arange(num_hours):
			end_time = round_hour(datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S'), up=True)
			end_time_str = str(end_time)
		# Add one last hour (can be discarded later)
		end_time = round_hour(datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S'), up=True)
		# Also adjust the df to contain valid times
		event_df['time'] = start_time_str
	# Calculate the framewise time bins that we need to summarize
	time_idx = pd.date_range(start=start_time, end=end_time, freq=str(bin_size_minutes) + 'min')
	event_df['adjusted_start'] = [time_to_frame(x['time'],str(start_time),30) + x['start'] for row_idx, x in event_df.iterrows()]
	event_df['adjusted_end'] = event_df['adjusted_start']+event_df['duration']
	event_df['percent_bout'] = 1.0
	# Summarize each time bin
	results_df_list = []
	for t1, t2 in zip(time_idx[:-1], time_idx[1:]):
		start_frame = time_to_frame(str(t1), str(start_time), fps)
		end_frame = time_to_frame(str(t2), str(start_time), fps)
		bins_to_summarize = event_df[np.logical_and(event_df['adjusted_start']>=start_frame, event_df['adjusted_end']<=end_frame)]
		cut_start = pd.DataFrame.copy(event_df[np.logical_and(event_df['adjusted_start']<start_frame, event_df['adjusted_end']>=start_frame)])
		if len(cut_start)>0:
			new_duration = cut_start['duration'] - (cut_start['adjusted_start'] - start_frame)
			cut_start['percent_bout'] = new_duration/cut_start['duration']
			cut_start['duration'] = new_duration
			bins_to_summarize = pd.concat([bins_to_summarize,cut_start])
		cut_end = pd.DataFrame.copy(event_df[np.logical_and(event_df['adjusted_start']<end_frame, event_df['adjusted_end']>=end_frame)])
		if len(cut_end)>0:
			new_duration = cut_end['duration'] - (end_frame - cut_end['adjusted_start'])
			cut_end['percent_bout'] = new_duration/cut_end['duration']
			cut_end['duration'] = new_duration
			bins_to_summarize = pd.concat([bins_to_summarize,cut_end])
		# With bins_to_summarize as needed
		# This operation throws a warning which can be ignored, so mute it before throwing...
		pd.options.mode.chained_assignment = None
		if 'distance' in bins_to_summarize.keys():
			bins_to_summarize['calc_dist'] = bins_to_summarize['distance']*bins_to_summarize['percent_bout']
		else:
			pass
		pd.options.mode.chained_assignment = 'warn'
		results = {}
		results['time'] = [str(t1)]
		results['time_no_pred'] = bins_to_summarize.loc[bins_to_summarize['is_behavior']==-1,'duration'].sum()
		results['time_not_behavior'] = bins_to_summarize.loc[bins_to_summarize['is_behavior']==0,'duration'].sum()
		results['time_behavior'] = bins_to_summarize.loc[bins_to_summarize['is_behavior']==1,'duration'].sum()
		results['bout_behavior'] = len(bins_to_summarize.loc[bins_to_summarize['is_behavior']==1])
		if 'distance' in bins_to_summarize.keys():
			results['not_behavior_dist'] = bins_to_summarize.loc[bins_to_summarize['is_behavior']==0,'calc_dist'].sum()
			results['behavior_dist'] = bins_to_summarize.loc[bins_to_summarize['is_behavior']==1,'calc_dist'].sum()
		results_df_list.append(pd.DataFrame(results))
	return pd.concat(results_df_list)

# Rounds a time object down to the nearest hour
# Default is to round down
def round_hour(t, up: bool=False):
	hour_to_round_to = t.hour
	# If we want to round up, just increment the hour
	if up:
		hour_to_round_to += 1
	# Do some remainder/modulo operations to handle day wrapping
	return (t.replace(day=t.day + (hour_to_round_to)//24, second=0, microsecond=0, minute=0, hour=hour_to_round_to%24))

# Converts a time to an equivalent frame index relative to rel_t
def time_to_frame(t: str, rel_t: str, fps: float):
	delta = datetime.strptime(t,'%Y-%m-%d %H:%M:%S')-datetime.strptime(rel_t,'%Y-%m-%d %H:%M:%S')
	return np.int64(delta.total_seconds()*fps)
