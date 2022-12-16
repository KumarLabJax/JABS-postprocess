import numpy as np
import pandas as pd
from datetime import datetime

# Transforms raw data per-experiment into binned results
def generate_binned_results(df: pd.DataFrame, bin_size_minutes: int=60):
	grouped_df = df.groupby(['exp_prefix','longterm_idx'])
	all_results = []
	for cur_group, cur_data in grouped_df:
		time_data = to_vector(cur_data)
		binned_results = time_data.groupby(pd.Grouper(freq=str(bin_size_minutes) + 'T'), group_keys=True).apply(get_results).reset_index().drop(columns=['level_1'])
		binned_results = binned_results.rename(columns={'level_0':'time'})
		binned_results['exp_prefix'], binned_results['longterm_idx'] = cur_group
		all_results.append(binned_results)
	all_results = pd.concat(all_results)
	return all_results

# Modifies a block of time to contain the predictions
def get_results(x):
	y = pd.DataFrame()
	y['time_no_pred'] = [np.sum(x['behavior']==-1)]
	y['time_not_behavior'] = [np.sum(x['behavior']==0)]
	y['time_behavior'] = [np.sum(x['behavior']==1)]
	y['bout_behavior'] = [np.sum(x['bout'])]
	return y

# Moves clock to the next hour
def add_hour(t):
	# Rounds to nearest hour by adding a timedelta hour if minute >= 30
	return (t.replace(day=t.day+(t.hour+1)//24, second=0, microsecond=0, minute=0, hour=(t.hour+1)%24))

# Converts an event dataframe into a vector
# This function should only be run on a single animal (eg 1 prediction per-frame)
def to_vector(event_df: pd.DataFrame):
	vid_blocks = event_df.groupby('time')
	dfs = []
	for cur_block, cur_group in vid_blocks:
		end_time = datetime.strftime(add_hour(datetime.strptime(cur_block,'%Y-%m-%d %H:%M:%S')), '%Y-%m-%d %H:%M:%S')
		time_idx = pd.date_range(start=cur_block, end=end_time, freq='33ms')
		time_vector = np.zeros(len(time_idx), dtype=np.int8)-1
		bout_vector = np.zeros(len(time_idx), dtype=np.float32)
		for _, cur_row in cur_group.iterrows():
			time_vector[cur_row['start']:cur_row['start']+cur_row['duration']] = cur_row['is_behavior']
			if cur_row['is_behavior'] == 1:
				bout_vector[cur_row['start']:cur_row['start']+cur_row['duration']] = 1/cur_row['duration']
		dfs.append(pd.DataFrame({'behavior':time_vector, 'bout':bout_vector}, index=time_idx))
	df = pd.concat(dfs)
	return df
