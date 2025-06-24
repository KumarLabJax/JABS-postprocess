'''
File for plotting generic plots for multi-day behavior

# Run this line before starting up the interactive python session for accessing libraries
# export PYTHONPATH=/JABS-postprocess/
# Alternatively, now we can use the singularity image at /projects/kumar-lab/JABS/JABS-Postprocessing-2023-02-07.sif
'''

import pandas as pd
import plotnine as p9
import re
import numpy as np
import mizani
from mizani import formatters, breaks
import os
import scipy
import itertools
from parse_table import read_ltm_summary_table,filter_experiment_time
# DONT FORGET TO change the above line to be:
#from analysis_utils.parse_table import read_ltm_summary_table,filter_experiment_time


# Name of the behavior based on what you are analyzing
plot_title = "Drinking Behavior"
behavior = "Drinking"

# Read in the summary results
results_file = '/projects/kumar-lab/b6-btbr-social-poses-update-id-2024-03-11/behavior_Drinking_summaries.csv'
# results_file = '/projects/kumar-lab/choij/b6-btbr-plots/Results_2024-01-10_sumner1_Drinking_Adults_summaries.csv'
# results_file = '/projects/kumar-lab/choij/pose-files-btbr-b6j-f/Results_2023-06-27_f_Drinking_summaries.csv'
# Read in JCMS metadata file
jmcrs_data = '/projects/kumar-lab/choij/lepr_poses/2023-09-07 TOM_TotalQueryForConfluence.xlsx'
header_data, df = read_ltm_summary_table(results_file, jmcrs_metadata=jmcrs_data)

# Extract experiment number from exp_prefix (remove underscore)
df['ExptNumber'] = df['exp_prefix'].str.rstrip('_')

# Experiments to be removed from the dataset
remove_experiments = ['MDB0003','MDX0008','MDX0017','MDX0093']
df = df[~np.isin(df['ExptNumber'], remove_experiments)]

# Delete out bins where no data exists
no_data = np.all(df[['time_no_pred','time_not_behavior','time_behavior']]==0, axis=1)
df = df[~no_data].reset_index()
# Get the average bout length per hour in frames
df['avg_bout_length'] = df['time_behavior']/df['bout_behavior']

# Read in bout information from bout file
results_file_bouts = re.sub('_summaries', '_bouts', results_file)
df_bouts = pd.read_csv(results_file_bouts, skiprows=2)
df_bouts['time'] = pd.to_datetime(df_bouts['time'])
# Since this is hourly data, we can transform start and duration fields to actual timestamps
frame_to_s = 30
df_bouts['time_start'] = df_bouts['time'] + pd.to_timedelta(df_bouts['start']/frame_to_s, unit='s')

df_bouts['time_end'] = df_bouts['time'] + pd.to_timedelta(df_bouts['start']/frame_to_s, unit='s') + pd.to_timedelta(df_bouts['duration']/frame_to_s, unit='s')


df_bouts_behavior = df_bouts[df_bouts['is_behavior'] == 1]
if jmcrs_data is not None:
	meta_df = pd.read_excel(jmcrs_data)
	meta_df = meta_df[['ExptNumber','Sex','Strain','Location']].drop_duplicates()
	meta_df['Room'] = [x.split(' ')[0] if isinstance(x,str) else ''  for x in meta_df['Location']]
	meta_df['Computer'] = [re.sub('.*(NV[0-9]+).*','\\1',x) if isinstance(x,str) else ''  for x in meta_df['Location']]
	# Merge metadata into main df
	df = pd.merge(df, meta_df, left_on='ExptNumber', right_on='ExptNumber', how='left')
	# Clean up duplicate columns from merge
	meta_cols = ['Sex', 'Strain', 'Location', 'Room', 'Computer']
	for col in meta_cols:
		if f"{col}_x" in df.columns:
			df.drop(columns=[f"{col}_x"], inplace=True)
		if f"{col}_y" in df.columns:
			df.rename(columns={f"{col}_y": col}, inplace=True)
	# Note: If you want to drop rows that don't have metadata, change how='inner'
	df_bouts_behavior = pd.merge(df_bouts_behavior, meta_df, left_on='exp_prefix', right_on='ExptNumber', how='left')

# Diagnostic: Check for experiments in results but missing from metadata
results_expts = set(df['ExptNumber'].unique())
meta_expts = set(meta_df['ExptNumber'].unique())
missing_expts = results_expts - meta_expts
print("Experiments in results but missing from metadata:", missing_expts)
print(df[df['ExptNumber'].isin(missing_expts)][['ExptNumber', 'Strain']].drop_duplicates())

# Calculate time_alive and prop_time_alive
# (Make sure this comes before filtering on time_alive)
df['time_alive'] = df['time_not_behavior'] + df['time_behavior']
df['prop_time_alive'] = df['time_alive'] / (df['time_alive'] + df['time_no_pred'])

# Only keep rows with valid Strain info
df = df[~df['Strain'].isna()]

# Filter out rows where time_alive <= 1 and print an error message for each
invalid_rows = df[df['time_alive'] <= 1]
for idx, row in invalid_rows.iterrows():
    print(f"[ERROR] Mouse {row['ExptNumber']} at time bin {row['zt_time_hour']} did not have valid predictions (only tracked for {row['time_alive']} frame(s) in that hour bin).")


df = df[df['time_alive'] > 1]

# Proportion of time where there are valid predictions need to be over 0.5 for them to remain in the plotting.
df = df[df['prop_time_alive'] > .5]

# df = df[df['Sex']=='F']

# This function has been modified from 
# from analysis_utils.plots import generate_time_vs_feature_plot
# Plotnine plot objects, which are returned here can be modified, plotted (blocking), plotted (non-blocking), or saved directly to disk as a figure (png or svg)
# New layers can be added to an existing plot through the + operator
# Plots can be plotted either using print(plot) [blocking] or plot.draw().show() [non-blocking]
# Plots can be saved to disk using plot.save('filename.ext', ...) see help(plot.save) for options

# Plotting time vs feature with groupings
# Generates a plotnine figure (which can be modified after returned)
# Handles the formatting under the hood
# If you want to remove the data plotted in favor of something else (eg points), pass draw_data=False
def generate_time_vs_feature_plot(y_axis, x_axis, outfile, df: pd.DataFrame, time: str='zt_time_hour', feature: str='rel_time_behavior', factor: str='Strain', draw_data: bool=True, title: str=None):
	# Detect the time datatype
	col_types = df.dtypes
	df_copy = pd.DataFrame.copy(df)
	if not pd.api.types.is_categorical_dtype(col_types[factor]):
		df_copy[factor] = df_copy[factor].astype('category')
	# Make a custom df for the lights block
	light_df = df.groupby([time,factor])[[feature,'lights_on']].mean().reset_index()
	# Max across the factor
	light_df = light_df.groupby(time)[[feature,'lights_on']].max().reset_index()
	light_df['lights_val'] = (1-light_df['lights_on'])*1.1*np.max(light_df[feature])
	if pd.api.types.is_timedelta64_dtype(col_types[time]) or pd.api.types.is_timedelta64_ns_dtype(col_types[time]):
		light_width = 60*60*10**9
	else:
		light_width = 1
	# Start building the plot
	plot = p9.ggplot(df)
	# Add in the line + background
	if draw_data:
		# Plot the background light rectangles first
		plot = plot + p9.geom_bar(p9.aes(x=time, y='lights_val'), light_df, width=light_width, stat='identity', fill='lightgrey')
		plot = plot + p9.stat_summary(p9.aes(x=time, y=feature, color=factor, fill=factor), fun_ymin=lambda x: np.mean(x)-np.std(x)/np.sqrt(len(x)), fun_ymax=lambda x: np.mean(x)+np.std(x)/np.sqrt(len(x)), fun_y=np.mean, geom=p9.geom_smooth)
	# Clean up some formatting
	plot = plot + p9.theme_bw()
	# Try to handle the different types of times
	# With full datetime, rotate
	if pd.api.types.is_datetime64_any_dtype(col_types[time]):
		plot = plot + p9.theme(axis_text_x=p9.element_text(rotation=90, hjust=0.5))
	# Timedelta, rotate and force breaks to hour format
	elif pd.api.types.is_timedelta64_dtype(col_types[time]) or pd.api.types.is_timedelta64_ns_dtype(col_types[time]):
		plot = plot + p9.theme(axis_text_x=p9.element_text(rotation=90, hjust=0.5)) + p9.scale_x_timedelta(labels=formatters.timedelta_format('h'))
		# breaks=breaks.timedelta_breaks(n_breaks)
	# 
	if title is not None:
		plot = plot + p9.labs(title=plot_title, color=factor, y=y_axis, x=x_axis)
	else:
		plot = plot + p9.labs(color=factor, y=feature)
	plot = plot + p9.scale_color_brewer(type='qual', palette='Set1')
	plot = plot + p9.scale_fill_brewer(type='qual', palette='Set1', guide=False) + p9.theme(axis_text_y=p9.element_text(size=18), axis_text_x=p9.element_text(size=18))
	plot.save(f'{outfile}.png')
	plot.save(f'{outfile}.svg')
	return plot


# Generate Relative Experiment Time Plots
proportion_fig = generate_time_vs_feature_plot(f"Proportion of Time Spent {behavior}", "Relative Experiment Time", "prop_rel", df, 'relative_exp_time', 'rel_time_behavior', title=header_data['Behavior'][0])
bout_num_fig = generate_time_vs_feature_plot("Average Number of Bouts", "Relative Experiment Time", "numbout_rel", df, 'relative_exp_time', 'bout_behavior', title=header_data['Behavior'][0])

df['avg_bout_length_sec'] = df['avg_bout_length']/30
# Generate ZT Experiment Time Plots
zt_proportion_fig = generate_time_vs_feature_plot(f"Proportion of Time Spent {behavior}", "ZT hour", "prop_zt", filter_experiment_time(df,num_hours=12), 'zt_time_hour', 'rel_time_behavior', title=header_data['Behavior'][0])
zt_bout_num_fig = generate_time_vs_feature_plot("Average Number of Bouts", "ZT hour", "numbout_zt", filter_experiment_time(df,num_hours=12), 'zt_time_hour', 'bout_behavior', title=header_data['Behavior'][0])
zt_bout_length_fig = generate_time_vs_feature_plot("Average Bout Length", "ZT hour", "boutlen_zt", filter_experiment_time(df,num_hours=12),'zt_time_hour', 'avg_bout_length_sec', title=header_data['Behavior'][0])


# Generate Room Comparison Line Plot
# The room comparison was only in males
df['LightCycle'] = df['zt_time_hour'].apply(lambda x: 'Light' if 0 <= x < 12 else 'Dark')
df_males = df[df['Sex']=='M']
df_males['Strain+Room'] = df_males['Strain'] + ' ' + df_males['Room']
room = generate_time_vs_feature_plot("Average Bout Length", "ZT Experiment Time", "delete_me", filter_experiment_time(df_males,num_hours=12), 'zt_time_hour', 'bout_behavior', 'Room', title=header_data['Behavior'][0],draw_data=False) + p9.facet_wrap('Strain')
room.save('room_comp_numbouts.svg')
room.save('room_comp_numbouts.png')


# Generate Room Comparison Box Plot

def generate_room_comp_box_plot(df, behavior_col, strain_col, room_col, lightcycle_col):
    plot = p9.ggplot(df, p9.aes(x=strain_col, y=behavior_col, fill=room_col)) + p9.geom_boxplot() + p9.facet_wrap(lightcycle_col) + p9.ggtitle('Boxplot of Behavior by Strain, Room, and Light Cycle') + p9.labs(y = "Average number of bouts") + p9.coord_cartesian(ylim=(0,20)) + p9.theme(axis_text_y=p9.element_text(size=18), axis_text_x=p9.element_text(size=18))
    plot.save('room_comp_box.svg')
    plot.save('room_comp_box.png')

filtered_df = filter_experiment_time(df_males,num_hours=12)
generate_room_comp_box_plot(filtered_df, 'bout_behavior', 'Strain', 'Room', 'LightCycle')


# Generate Strain Comparison Violin Plot of average bout lengths across light cycle
def generate_strain_comp_box_plot(df, behavior_col, strain_col, lightcycle_col):
    plot = p9.ggplot(df, p9.aes(x=strain_col, y=behavior_col, fill=lightcycle_col)) + p9.coord_cartesian(ylim=(0,30)) + p9.geom_violin(width=0.3) + p9.theme_bw() + p9.facet_wrap('~' + strain_col, scales='free') + p9.theme(axis_text_x=p9.element_text(rotation=90, hjust=1), axis_ticks=p9.element_blank()) + p9.ggtitle('Violin Plot of Behavior by Strain and Light Cycle') + p9.labs(y = "Average Bout Length")
    plot.save('violinplot_light_dark_bout_length_compare.png')
    plot.save('violinplot_light_dark_bout_length_compare.svg')

filtered_df = filter_experiment_time(df,num_hours=12)
generate_strain_comp_box_plot(filtered_df, 'avg_bout_length_sec', 'Strain', 'LightCycle')


df['Unique_animal'] = df['longterm_idx'].astype(str) + df['exp_prefix']
# Generate plots with every individual as a line
prop_fig = generate_time_vs_feature_plot(f"Proportion of Time Spent {behavior}", "Relative Experiment Time", "delete", df, 'relative_exp_time', 'rel_time_behavior',factor='Unique_animal', draw_data=False, title=header_data['Behavior'][0])
plot_prop_fig = prop_fig + p9.geom_line(p9.aes(x='relative_exp_time', y='rel_time_behavior',group='Unique_animal', color='Strain'), alpha=0.5)
plot_prop_fig.save("individual_prop_rel.png")
plot_prop_fig.save("individual_prop_rel.svg")


num_bout = generate_time_vs_feature_plot("Average Number of Bouts", "Relative Experiment Time", "delete", df, 'relative_exp_time', 'bout_behavior', draw_data=False, title=header_data['Behavior'][0])
plot_prop_fig = num_bout + p9.geom_line(p9.aes(x='relative_exp_time', y='bout_behavior',group='Unique_animal', color='Strain'), alpha=0.5)
plot_prop_fig.save("individual_numbout_rel.png")
plot_prop_fig.save("individual_numbout_rel.svg")


df['avg_bout_length_sec'] = df['avg_bout_length']/30

# Generate ZT Experiment Time Plots
zt_prop_fig = generate_time_vs_feature_plot(f"Proportion of Time Spent {behavior}", "ZT hour", "delete", filter_experiment_time(df,num_hours=12), 'zt_time_hour', 'rel_time_behavior', draw_data=False, title=header_data['Behavior'][0])
plot_zt_prop_fig = zt_prop_fig + p9.stat_summary(p9.aes(x='zt_time_hour', y='rel_time_behavior',group='Unique_animal', color='Strain'), geom=p9.geom_line, fun_y=np.mean, alpha=0.5)
plot_zt_prop_fig.save("individual_prop_zt.png")
plot_zt_prop_fig.save("individual_prop_zt.svg")

zt_bout_num_fig = generate_time_vs_feature_plot("Average Number of Bouts", "ZT hour", "delete", filter_experiment_time(df,num_hours=12), 'zt_time_hour', 'bout_behavior', draw_data=False, title=header_data['Behavior'][0])
plot_zt_bout_num_fig = zt_bout_num_fig + p9.stat_summary(p9.aes(x='zt_time_hour', y='bout_behavior',group='Unique_animal', color='Strain'), geom=p9.geom_line, fun_y=np.mean, alpha=0.5)
plot_zt_bout_num_fig.save("individual_numbout_zt.png")
plot_zt_bout_num_fig.save("individual_numbout_zt.svg")

zt_bout_length_fig = generate_time_vs_feature_plot("Average Bout Length", "ZT hour", "delete", filter_experiment_time(df,num_hours=12),'zt_time_hour', 'avg_bout_length_sec', draw_data=False, title=header_data['Behavior'][0])
plot_zt_bout_length_fig = zt_bout_length_fig + p9.stat_summary(p9.aes(x='zt_time_hour', y='avg_bout_length_sec',group='Unique_animal', color='Strain'), geom=p9.geom_line, fun_y=np.mean, alpha=0.5)
plot_zt_bout_length_fig.save("individual_boutlen_zt.png")
plot_zt_bout_length_fig.save("individual_boutlen_zt.svg")
