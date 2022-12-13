import pandas as pd
import plotnine as p9
import mizani
import re

# Read in the summary results
results_file = 'results_2022-12-13_Approach_summaries.csv'
header_data = pd.read_csv(results_file, nrows=1)
df = pd.read_csv(results_file, skiprows=2)
df['time'] = pd.to_datetime(df['level_0'])

# Plot the number of bouts as points across time for each experiment
(
	p9.ggplot(df)+
	p9.geom_point(p9.aes(x='time', y='bout_behavior', color='factor(longterm_idx)'))+
	p9.scale_x_datetime(breaks=mizani.breaks.date_breaks('6 hour'), labels=mizani.formatters.date_format('%Y-%m-%d %H-%M-%S'))+
	p9.facet_wrap('exp_prefix', scales='free_x')+
	p9.theme_bw()+
	p9.theme(axis_text_x=p9.element_text(rotation=90, hjust=0.5))+
	p9.labs(title=header_data['Behavior'][0], color='Identity')
).draw().show()

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
