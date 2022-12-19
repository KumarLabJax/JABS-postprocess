import pandas as pd
import re
import numpy as np

from analysis_utils.parse_table import read_ltm_summary_table, filter_experiment_time
from analysis_utils.plots import generate_time_vs_feature_plot

#-------------------------------
# Example multi-day summary plot
#-------------------------------
# Read in the summary results
results_file = '/media/bgeuther/Storage/TempStorage/test-behavior-project/test_results_Approach_summaries.csv'
# results_file = 'ProjectResults_2022-12-15_Nose_nose_summaries.csv'
# results_file = 'ProjectResults_2022-12-15_Nose_genital_summaries.csv'

jmcrs_data = '~/Downloads/2022-11-16 TOM_TotalQueryForConfluence.xlsx'
header_data, df = read_ltm_summary_table(results_file, jmcrs_metadata=jmcrs_data)

# Now we have a dataframe with a bunch of temporal columns and a bunch of feature columns
df.keys()

# Plot relative_exp_time vs rel_time_behavior
generate_time_vs_feature_plot(df, 'relative_exp_time', 'rel_time_behavior').draw().show()
# Collapse the days into one 24-hr cycle and plot number of bouts
generate_time_vs_feature_plot(df, 'zt_time_hour', 'bout_behavior').draw().show()

# Since the first day normally has acclimation, do the same previous plot removing data before midnight on the first night
generate_time_vs_feature_plot(filter_experiment_time(df), 'zt_time_hour', 'bout_behavior').draw().show()

# Check the mean of data after first day (midnight)
filter_experiment_time(df).groupby('Strain').agg({'rel_time_behavior':np.mean})

#----------------------
# Example Ethogram plot
#----------------------
# Read in the bout results
results_file_bouts = re.sub('_summaries', '_bouts', results_file)
df_bouts = pd.read_csv(results_file_bouts, skiprows=2)
df_bouts['time'] = pd.to_datetime(df_bouts['time'])
# Since this is hourly data, we can transform start and duration fields to actual timestamps
frame_to_s = 30
df_bouts['time_start'] = df_bouts['time'] + pd.to_timedelta(df_bouts['start']/frame_to_s, unit='s')
df_bouts['time_end'] = df_bouts['time'] + pd.to_timedelta(df_bouts['start']/frame_to_s, unit='s') + pd.to_timedelta(df_bouts['duration']/frame_to_s, unit='s')

# Plot the bouts (behavior, not behavior, and missing data)
(
	p9.ggplot(df_bouts)+
	p9.geom_rect(p9.aes(xmin='time_start', xmax='time_end', ymin='longterm_idx - 0.5', ymax='longterm_idx + 0.5', fill='factor(is_behavior)'))+
	p9.scale_x_datetime(breaks=mizani.breaks.date_breaks('6 hour'), labels=mizani.formatters.date_format('%Y-%m-%d %H-%M-%S'))+
	p9.scale_fill_discrete(labels=['No Mouse', 'Not Behavior', 'Behavior'])+
	p9.facet_wrap('exp_prefix', scales='free_x')+
	p9.theme_bw()+
	p9.theme(axis_text_x=p9.element_text(rotation=90, hjust=0.5))+
	p9.labs(title=header_data['Behavior'][0])
).draw().show()
