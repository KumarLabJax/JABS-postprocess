import pandas as pd
import plotnine as p9
import re
import numpy as np
import mizani
from matplotlib import gridspec

from analysis_utils.parse_table import read_ltm_summary_table, filter_experiment_time
from analysis_utils.plots import generate_time_vs_feature_plot

#-------------------------------
# Example multi-day summary plot
#-------------------------------
# Read in the summary results
results_file = '/media/bgeuther/Storage/TempStorage/SocialPaper/Play/analysis-2023-07-20/Results_2023-09-28_Approach_summaries.csv'

jmcrs_data = '~/Downloads/2023-08-04 TOM_TotalQueryForConfluence.xlsx'

header_data, df = read_ltm_summary_table(results_file, jmcrs_data)
# Now we have a dataframe with a bunch of temporal columns and a bunch of feature columns
df.keys()

# Plot relative_exp_time vs rel_time_behavior
generate_time_vs_feature_plot(df, 'relative_exp_time', 'rel_time_behavior', title=header_data['Behavior'][0]).draw().show()
# Collapse the days into one 24-hr cycle and plot number of bouts
generate_time_vs_feature_plot(df, 'zt_time_hour', 'bout_behavior', title=header_data['Behavior'][0]).draw().show()

# Since the first day normally has acclimation, do the same previous plot removing data before midnight on the first night
generate_time_vs_feature_plot(filter_experiment_time(df), 'zt_time_hour', 'bout_behavior', title=header_data['Behavior'][0]).draw().show()

# Plot the data by Strain+Room
df['Strain+Room'] = df['Strain'] + ' ' + df['Room']
generate_time_vs_feature_plot(filter_experiment_time(df), 'zt_time_hour', 'bout_behavior', 'Strain+Room', title=header_data['Behavior'][0]).draw().show()

# Check the mean of data after first day (midnight)
filter_experiment_time(df).groupby('Strain').agg({'rel_time_behavior':np.mean})

# Read in more behaviors to make an aggregate graph
# This is stretching the limits of the function capability. Facets don't work too well. Nonetheless, plot room by behavior predictions
# Earlier social paper scans w/ heuristics and classifiers
# folder = '/media/bgeuther/Storage/TempStorage/B6-BTBR/results/'
# results_files = [folder + x for x in ['Results_2023-01-13_Nose_nose_summaries.csv', 'Results_2023-01-13_Nose_genital_summaries.csv', 'Results_2023-01-13_Approach_summaries.csv', 'Results_2023-01-13_Chase_summaries.csv', 'Results_2023-01-13_Leave_summaries.csv', 'Results_2023-01-13_Activity_1_summaries.csv', 'Results_2023-02-07_heuristic_approach_summaries.csv', 'Results_2023-02-07_heuristic_close_summaries.csv', 'Results_2023-02-07_heuristic_contact_summaries.csv', 'Results_2023-02-07_heuristic_nose_ear_summaries.csv', 'Results_2023-02-07_heuristic_nose_genital_summaries.csv', 'Results_2023-02-07_heuristic_nose_nose_summaries.csv']]

# Filtering tests
# results_files = [
# 	'results/Results_2023-02-07_heuristic_nose_nose_summaries.csv',
# 	'filtered_results/test_2023-02-13_highact_heuristic_nose_nose_summaries.csv',
# 	'filtered_results/test_2023-02-13_medact_heuristic_nose_nose_summaries.csv',
# 	'filtered_results/test_2023-02-13_heuristic_nose_nose_summaries.csv',
# 	'filtered_results/test_2023-02-13_medact_invover_heuristic_nose_nose_summaries.csv',
# ]

# Updated play results:
# Located in 
folder = '/media/bgeuther/Storage/TempStorage/SocialPaper/Play/analysis-2023-07-20/'
results_files = [folder + x for x in [
	'Results_2023-09-28_Approach_summaries.csv',
	'Results_2023-09-28_Chase_summaries.csv',
	'Results_2023-09-28_Leave_summaries.csv',
	'Results_2023-09-28_Jerk_summaries.csv',
	'Results_2023-09-28_Play_Chase_summaries.csv',
	'Results_2023-09-28_Activity_1_summaries.csv']]

# This inline loop reads in all the data from results_files.
# Note that header data is discarded
# header_data, df = read_ltm_summary_table(results_file, jmcrs_data)
df = pd.concat([read_ltm_summary_table(x, jmcrs_data)[1] for x in results_files])
# 8 weeks are adults
df['Adult'] = np.where(df['Age'].dt.days > 8*7, 'Adult', 'Juvenile')

df['group_to_color'] = df['Strain'] + ' ' + df['Adult'] + ' ' + df['sex']
# To allow activity to be plotted against bouts...
# df['feature_val'] = df['bout_behavior'].astype(float)
df['feature_val'] = df['time_behavior'].astype(float)
df.loc[df['Behavior']=='Activity > 5.0cm/s', 'feature_val'] = df.loc[df['Behavior']=='Activity > 5.0cm/s', 'behavior_dist']
# Change default sorting
df.loc[:,'Behavior'] = pd.Categorical(df['Behavior'], categories=['Approach', 'Chase', 'Leave', 'Jerk', 'Play_Chase', 'Activity > 5.0cm/s'])


adult_male_df = df[np.logical_and(df['sex'] == 'M', df['Adult'] == 'Adult')].reset_index(drop=True)

(generate_time_vs_feature_plot(filter_experiment_time(adult_male_df), 'zt_time_hour', 'feature_val', 'group_to_color')+p9.facet_grid('Behavior~Room', scales='free_y')).draw().show()
(generate_time_vs_feature_plot(filter_experiment_time(df), 'zt_time_hour', 'feature_val', 'group_to_color')+p9.facet_grid('Behavior~sex', scales='free_y')).draw().show()

# Trimmed down version of interesting areas
play_behaviors = ['Jerk', 'Play_Chase', 'Activity > 5.0cm/s']
(generate_time_vs_feature_plot(filter_experiment_time(df[np.isin(df['Behavior'], play_behaviors)]), 'zt_time_hour', 'feature_val', 'Adult')+p9.facet_grid('Behavior~Strain+sex', scales='free_y')).draw().show()

not_play_behaviors = ['Approach', 'Chase', 'Leave', 'Activity > 5.0cm/s']
new_df = adult_male_df[np.isin(adult_male_df['Behavior'], not_play_behaviors)]
new_df.loc[:,'Behavior'] = pd.Categorical(new_df['Behavior'], categories=not_play_behaviors)
(generate_time_vs_feature_plot(filter_experiment_time(new_df), 'zt_time_hour', 'feature_val', 'group_to_color')+p9.facet_grid('Behavior~Room', scales='free_y')).draw().show()

# Day/Night summaries
df['exp_day'] = df['relative_exp_time'].dt.days
(p9.ggplot(adult_male_df[adult_male_df['Behavior']=='Chase'])+p9.stat_summary(p9.aes(x='exp_day', y='rel_time_behavior', fill='Strain'), fun_data='mean_sdl', geom='bar', position='dodge')+p9.facet_grid('lights_on~.')+p9.scale_fill_brewer(type='qual',palette='Set1')+p9.theme_bw()).draw().show()

# Plots for Vivek (re-generating Gautam's) for social paper/presentation
# bars are SE
# data is remove for habituation, but we can just ignore day1 for this...
behaviors = ['Approach', 'Chase', 'Leave', 'Jerk', 'Play_Chase']
bout_df = filter_experiment_time(df[df['Behavior'].isin(behaviors)])
activity_df = filter_experiment_time(df[df['Behavior']=='Activity > 5.0cm/s'])
# Lazy to put the features into the same column
bout_df.loc[:,'feature_vals'] = bout_df['bout_behavior']
activity_df.loc[:,'feature_vals'] = activity_df['behavior_dist']
full_df = pd.concat([bout_df, activity_df])
# Only use adult males for this plot
adult_male_df = full_df[np.logical_and(full_df['sex'] == 'M', full_df['Adult'] == 'Adult')].reset_index(drop=True)
# Base plot that doesn't scale everything correctly...
# (generate_time_vs_feature_plot(filter_experiment_time(full_df), 'zt_time_hour', 'feature_vals', 'Strain')+p9.facet_grid('Behavior~Room', scales='free_y')).draw().show()

# Room comparison plotting
all_panels = []
panel_strs = behaviors + ['Activity > 5.0cm/s']
rooms = {'B2B': 'B2B', 'CBAX2B': 'CBAX2', 'B6': 'RAF-B6'}
for idx in range(len(panel_strs)):
	panel_df = adult_male_df[adult_male_df['Behavior'] == panel_strs[idx]]
	panel_light = panel_df.groupby(['zt_time_hour','Strain'])[['feature_vals','lights_on']].mean().reset_index().groupby('zt_time_hour')[['feature_vals','lights_on']].max().reset_index()
	panel_light['lights_val'] = (1-panel_light['lights_on'])*1.2*np.max(panel_light['feature_vals'])
	for key, val in rooms.items():
		sub_df = panel_df[panel_df['Room']==key]
		panel = (
			p9.ggplot(sub_df) +
			p9.geom_bar(p9.aes(x='zt_time_hour', y='lights_val'), panel_light, width=1, stat='identity', fill='lightgrey') + 
			p9.stat_summary(p9.aes(x='zt_time_hour', y='feature_vals', color='Strain + sex + Adult', fill='Strain + sex + Adult'), fun_ymin=lambda x: np.mean(x)-np.std(x)/np.sqrt(len(x)), fun_ymax=lambda x: np.mean(x)+np.std(x)/np.sqrt(len(x)), fun_y=np.mean, geom=p9.geom_smooth) + 
			p9.stat_summary(p9.aes(x='zt_time_hour', y='feature_vals', color='Strain + sex + Adult', fill='Strain + sex + Adult'), fun_y=np.mean, geom=p9.geom_point) + 
			# p9.facet_grid('.~Room') + 
			p9.theme_bw() + 
			p9.scale_color_brewer(type='qual', palette='Set1') + 
			p9.scale_fill_brewer(type='qual', palette='Set1', guide=False)
		)
		if panel_strs[idx] == 'Activity > 5.0cm/s':
			panel = panel + p9.labs(title=val, color='Strain', x='ZT Time', y='Distance Traveled (cm)')
		else:
			panel = panel + p9.labs(title='', color='Strain', x='ZT Time', y=panel_strs[idx] + ' Counts')
		all_panels.append(panel)

fig = (p9.ggplot()+p9.geom_blank(data=adult_male_df)+p9.theme_void()).draw()
gs = gridspec.GridSpec(int(len(all_panels)/len(rooms)),len(rooms))
for idx in range(len(all_panels)):
	cur_ax = fig.add_subplot(gs[idx//len(rooms),idx%len(rooms)])
	_ = all_panels[idx]._draw_using_figure(fig, [cur_ax])

fig.show()

# Play behavior plotting
all_panels = []
panel_strs = behaviors + ['Activity > 5.0cm/s']
for idx in range(len(panel_strs)):
	panel_df = full_df[full_df['Behavior'] == panel_strs[idx]]
	panel_light = panel_df.groupby(['zt_time_hour','Strain'])[['feature_vals','lights_on']].mean().reset_index().groupby('zt_time_hour')[['feature_vals','lights_on']].max().reset_index()
	panel_light['lights_val'] = (1-panel_light['lights_on'])*1.2*np.max(panel_light['feature_vals'])
	panel = (
		p9.ggplot(panel_df) +
		p9.geom_bar(p9.aes(x='zt_time_hour', y='lights_val'), panel_light, width=1, stat='identity', fill='lightgrey') + 
		p9.stat_summary(p9.aes(x='zt_time_hour', y='feature_vals', color='Strain + sex + Adult', fill='Strain + sex + Adult'), fun_ymin=lambda x: np.mean(x)-np.std(x)/np.sqrt(len(x)), fun_ymax=lambda x: np.mean(x)+np.std(x)/np.sqrt(len(x)), fun_y=np.mean, geom=p9.geom_smooth) + 
		p9.stat_summary(p9.aes(x='zt_time_hour', y='feature_vals', color='Strain + sex + Adult', fill='Strain + sex + Adult'), fun_y=np.mean, geom=p9.geom_point) + 
		# p9.facet_grid('.~Room') + 
		p9.theme_bw() + 
		p9.scale_color_brewer(type='qual', palette='Set1') + 
		p9.scale_fill_brewer(type='qual', palette='Set1', guide=False)
	)
	if panel_strs[idx] == 'Activity > 5.0cm/s':
		panel = panel + p9.labs(title=val, color='Strain', x='ZT Time', y='Distance Traveled (cm)')
	else:
		panel = panel + p9.labs(title='', color='Strain', x='ZT Time', y=panel_strs[idx] + ' Counts')
	all_panels.append(panel)

fig = (p9.ggplot()+p9.geom_blank(data=full_df)+p9.theme_void()).draw()
gs = gridspec.GridSpec(int(len(all_panels)),1)
for idx in range(len(all_panels)):
	cur_ax = fig.add_subplot(gs[idx,0])
	_ = all_panels[idx]._draw_using_figure(fig, [cur_ax])

fig.show()


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
experiments = np.unique(df_bouts['exp_prefix'])


# Plot the bouts (behavior, not behavior, and missing data)
(
	p9.ggplot(df_bouts[df_bouts['exp_prefix']==experiments[1]])+
	p9.geom_rect(p9.aes(xmin='time_start', xmax='time_end', ymin='longterm_idx - 0.5', ymax='longterm_idx + 0.5', fill='factor(is_behavior)'))+
	p9.scale_x_datetime(breaks=mizani.breaks.date_breaks('6 hour'), labels=mizani.formatters.date_format('%Y-%m-%d %H-%M-%S'))+
	p9.scale_fill_discrete(labels=['No Mouse', 'Not Behavior', 'Behavior'])+
	p9.facet_wrap('exp_prefix', scales='free_x')+
	p9.theme_bw()+
	p9.theme(axis_text_x=p9.element_text(rotation=90, hjust=0.5))+
	p9.labs(title=header_data['Behavior'][0])
).draw().show()

# Plot only behavior events
(
	p9.ggplot(df_bouts[np.logical_and(df_bouts['exp_prefix']==experiments[1], df_bouts['is_behavior']==1)])
	+ p9.geom_point(p9.aes(x='time_start', y='factor(longterm_idx)', color='factor(longterm_idx)'), shape='|')
	+ p9.scale_x_datetime(breaks=mizani.breaks.date_breaks('6 hour'), labels=mizani.formatters.date_format('%Y-%m-%d %H-%M-%S'))
	+ p9.theme_bw()
	+ p9.theme(axis_text_x=p9.element_text(rotation=90, hjust=0.5))
).draw().show()

# Plot hourly summary
(
	p9.ggplot(df[np.logical_and(df['ExptNumber']==experiments[1], df['Behavior']==header_data['Behavior'][0])])
	+ p9.geom_line(p9.aes(x='time', y='bout_behavior', color='factor(longterm_idx)'))
	+ p9.scale_x_datetime(breaks=mizani.breaks.date_breaks('6 hour'), labels=mizani.formatters.date_format('%Y-%m-%d %H-%M-%S'))
	+ p9.theme_bw()
	+ p9.theme(axis_text_x=p9.element_text(rotation=90, hjust=0.5))
).draw().show()

# Plot hours over day
(
	p9.ggplot(df[np.logical_and(df['ExptNumber']==experiments[1], df['Behavior']==header_data['Behavior'][0])])
	+ p9.stat_summary(p9.aes(x='zt_time_hour', y='bout_behavior', color='factor(longterm_idx)', fill='factor(longterm_idx)'), fun_data='mean_sdl', geom='smooth')
	+ p9.theme_bw()
	+ p9.theme(axis_text_x=p9.element_text(rotation=90, hjust=0.5))
).draw().show()

# Remove day 1 from previous
(
	p9.ggplot(filter_experiment_time(df[np.logical_and(df['ExptNumber']==experiments[1], df['Behavior']==header_data['Behavior'][0])]))
	+ p9.stat_summary(p9.aes(x='zt_time_hour', y='bout_behavior', color='factor(longterm_idx)', fill='factor(longterm_idx)'), fun_data='mean_sdl', geom='smooth')
	+ p9.theme_bw()
	+ p9.theme(axis_text_x=p9.element_text(rotation=90, hjust=0.5))
).draw().show()
