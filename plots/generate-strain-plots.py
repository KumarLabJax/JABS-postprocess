'''
File for plotting generic plots for multi-day behavior
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
# from parse_table import read_ltm_summary_table,filter_experiment_time
# DONT FORGET TO change the above line to be:
from analysis_utils.parse_table import read_ltm_summary_table,filter_experiment_time


def generate_behavior_plots(behavior, results_file, jmcrs_data, remove_experiments, output_dir="."):
	# Create output directory if it doesn't exist
	os.makedirs(output_dir, exist_ok=True)

	# Custom theme for consistent styling
	custom_theme = p9.theme_bw() + \
		p9.theme(
			axis_text_y=p9.element_text(size=12),
			axis_text_x=p9.element_text(size=12),
			axis_title=p9.element_text(size=14, face='bold'),
			plot_title=p9.element_text(size=16, face='bold'),
			legend_title=p9.element_text(size=12, face='bold'),
			legend_text=p9.element_text(size=10),
			panel_grid_minor=p9.element_blank()
		)

	print(f"Generating plots for {behavior} behavior...")

	plot_title = f"{behavior} Behavior"

	# Read in the summary results
	header_data, df = read_ltm_summary_table(results_file, jmcrs_metadata=jmcrs_data)

	# Check if data exists before plotting
	if len(df) == 0:
		print(f"Warning: No data available for {behavior} behavior")
		return

	# Extract experiment number from exp_prefix (remove underscore)
	df['ExptNumber'] = df['exp_prefix'].str.rstrip('_')

	# Experiments to be removed from the dataset
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

	# Get sample size information
	n_arenas = len(df['ExptNumber'].unique())
	print(f"Number of arenas in analysis: {n_arenas}")

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
	def generate_time_vs_feature_plot(y_axis, x_axis, outfile, df: pd.DataFrame, time: str='zt_time_hour', feature: str='rel_time_behavior', factor: str='Strain', draw_data: bool=True, title: str=None, save_files: bool=True):
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
		plot = plot + custom_theme
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
			plot = plot + p9.labs(title=f"{title} (n={n_arenas} arenas)", color=factor, y=y_axis, x=x_axis)
		else:
			plot = plot + p9.labs(color=factor, y=feature)
		plot = plot + p9.scale_color_brewer(type='qual', palette='Set1')
		plot = plot + p9.scale_fill_brewer(type='qual', palette='Set1', guide=False)
		if save_files:
			try:
				plot.save(os.path.join(output_dir, f'{outfile}_{behavior}.svg'))
			except Exception as e:
				print(f"Error saving {outfile}_{behavior}.svg: {e}")
		return plot


	# Generate Relative Experiment Time Plots
	proportion_fig = generate_time_vs_feature_plot(f"Proportion of Time Spent {behavior}", "Relative Experiment Time", "prop_rel", df, 'relative_exp_time', 'rel_time_behavior', title=header_data['Behavior'][0])
	bout_num_fig = generate_time_vs_feature_plot("Average Number of Bouts", "Relative Experiment Time", "numbout_rel", df, 'relative_exp_time', 'bout_behavior', title=header_data['Behavior'][0])

	df['avg_bout_length_sec'] = df['avg_bout_length']/30
	# Generate ZT Experiment Time Plots
	zt_proportion_fig = generate_time_vs_feature_plot(f"Proportion of Time Spent {behavior}", "ZT hour", "prop_zt", filter_experiment_time(df,num_hours=12), 'zt_time_hour', 'rel_time_behavior', title=header_data['Behavior'][0])
	zt_bout_num_fig = generate_time_vs_feature_plot("Average Number of Bouts", "ZT hour", "numbout_zt", filter_experiment_time(df,num_hours=12), 'zt_time_hour', 'bout_behavior', title=header_data['Behavior'][0])
	zt_bout_length_fig = generate_time_vs_feature_plot("Average Bout Length", "ZT hour", "boutlen_zt", filter_experiment_time(df,num_hours=12),'zt_time_hour', 'avg_bout_length_sec', title=header_data['Behavior'][0])


	# Generate Room Comparison Line Plot (males only)
	# The room comparison was only in males (in terms of experiment design)
	df['LightCycle'] = df['zt_time_hour'].apply(lambda x: 'Light' if 0 <= x < 12 else 'Dark')
	df_males = df[df['Sex']=='M']
	
	# Create room comparison line plot
	room_data = filter_experiment_time(df_males, num_hours=12)
	
	# Create light/dark background data
	light_df = room_data.groupby(['zt_time_hour', 'Room'])['bout_behavior'].mean().reset_index()
	light_df = light_df.groupby('zt_time_hour')['bout_behavior'].max().reset_index()
	light_df['lights_on'] = room_data.groupby('zt_time_hour')['lights_on'].first().values
	light_df['lights_val'] = (1-light_df['lights_on'])*1.1*np.max(light_df['bout_behavior'])
	
	room_plot = p9.ggplot(room_data) + \
		p9.geom_bar(p9.aes(x='zt_time_hour', y='lights_val'), light_df, width=1, stat='identity', fill='lightgrey') + \
		p9.stat_summary(p9.aes(x='zt_time_hour', y='bout_behavior', color='Room', fill='Room'), 
						fun_ymin=lambda x: np.mean(x)-np.std(x)/np.sqrt(len(x)), 
						fun_ymax=lambda x: np.mean(x)+np.std(x)/np.sqrt(len(x)), 
						fun_y=np.mean, geom=p9.geom_smooth) + \
		p9.facet_wrap('Strain') + \
		custom_theme + \
		p9.labs(title=f'{behavior} Behavior by Room and Strain (Males Only, n={len(df_males["ExptNumber"].unique())} arenas)', 
				x='Zeitgeber Time (hours)', y='Average Number of Bouts') + \
		p9.scale_color_brewer(type='qual', palette='Set1') + \
		p9.scale_fill_brewer(type='qual', palette='Set1', guide=False)
	try:
		room_plot.save(os.path.join(output_dir, f'room_comp_numbouts_{behavior}.svg'))
	except Exception as e:
		print(f"Error saving room_comp_numbouts_{behavior}.svg: {e}")


	# Generate Room Comparison Box Plot

	def generate_room_comp_box_plot(df, behavior_col, strain_col, room_col, lightcycle_col):
		plot = p9.ggplot(df, p9.aes(x=strain_col, y=behavior_col, fill=room_col)) + \
			p9.geom_boxplot(alpha=0.7) + \
			p9.facet_wrap(lightcycle_col) + \
			custom_theme + \
			p9.ggtitle(f'Boxplot of {behavior} Behavior by Strain, Room, and Light Cycle') + \
			p9.labs(y = "Average number of bouts", x="Strain") + \
			p9.coord_cartesian(ylim=(0,20)) + \
			p9.scale_fill_brewer(type='qual', palette='Set1')
		try:
			plot.save(os.path.join(output_dir, f'room_comp_box_{behavior}.svg'))
		except Exception as e:
			print(f"Error saving room_comp_box_{behavior}.svg: {e}")

	filtered_df = filter_experiment_time(df_males,num_hours=12)
	generate_room_comp_box_plot(filtered_df, 'bout_behavior', 'Strain', 'Room', 'LightCycle')


	# Generate Strain Comparison Violin Plot of average bout lengths across light cycle
	def generate_strain_comp_box_plot(df, behavior_col, strain_col, lightcycle_col):
		plot = p9.ggplot(df, p9.aes(x=strain_col, y=behavior_col, fill=lightcycle_col)) + \
			p9.coord_cartesian(ylim=(0,30)) + \
			p9.geom_violin(width=0.3, alpha=0.7) + \
			p9.geom_boxplot(width=0.2, fill='white', alpha=0.7) + \
			custom_theme + \
			p9.facet_wrap('~' + strain_col, scales='free') + \
			p9.theme(axis_text_x=p9.element_text(rotation=90, hjust=1), axis_ticks=p9.element_blank()) + \
			p9.ggtitle(f'Violin Plot of {behavior} Behavior by Strain and Light Cycle') + \
			p9.labs(y = "Average Bout Length (seconds)", x="Light Cycle") + \
			p9.scale_fill_brewer(type='qual', palette='Set1')
		try:
			plot.save(os.path.join(output_dir, f'violinplot_light_dark_bout_length_compare_{behavior}.svg'))
		except Exception as e:
			print(f"Error saving violinplot_light_dark_bout_length_compare_{behavior}.svg: {e}")

	filtered_df = filter_experiment_time(df,num_hours=12)
	generate_strain_comp_box_plot(filtered_df, 'avg_bout_length_sec', 'Strain', 'LightCycle')


	df['Unique_animal'] = df['longterm_idx'].astype(str) + df['exp_prefix']
	# Generate plots with every individual as a line
	prop_fig = generate_time_vs_feature_plot(f"Proportion of Time Spent {behavior}", "Relative Experiment Time", "delete", df, 'relative_exp_time', 'rel_time_behavior',factor='Unique_animal', draw_data=False, title=header_data['Behavior'][0], save_files=False)
	plot_prop_fig = prop_fig + p9.geom_line(p9.aes(x='relative_exp_time', y='rel_time_behavior',group='Unique_animal', color='Strain'), alpha=0.3) + \
		custom_theme + p9.labs(title=f"Individual {behavior} Behavior (n={n_arenas} arenas)")
	try:
		plot_prop_fig.save(os.path.join(output_dir, f"individual_prop_rel_{behavior}.svg"))
	except Exception as e:
		print(f"Error saving individual_prop_rel_{behavior}.svg: {e}")


	num_bout = generate_time_vs_feature_plot("Average Number of Bouts", "Relative Experiment Time", "delete", df, 'relative_exp_time', 'bout_behavior', draw_data=False, title=header_data['Behavior'][0], save_files=False)
	plot_prop_fig = num_bout + p9.geom_line(p9.aes(x='relative_exp_time', y='bout_behavior',group='Unique_animal', color='Strain'), alpha=0.3) + \
		custom_theme + p9.labs(title=f"Individual Bout Numbers (n={n_arenas} arenas)")
	try:
		plot_prop_fig.save(os.path.join(output_dir, f"individual_numbout_rel_{behavior}.svg"))
	except Exception as e:
		print(f"Error saving individual_numbout_rel_{behavior}.svg: {e}")


	df['avg_bout_length_sec'] = df['avg_bout_length']/30

	# Generate ZT Experiment Time Plots
	zt_prop_fig = generate_time_vs_feature_plot(f"Proportion of Time Spent {behavior}", "ZT hour", "delete", filter_experiment_time(df,num_hours=12), 'zt_time_hour', 'rel_time_behavior', draw_data=False, title=header_data['Behavior'][0], save_files=False)
	plot_zt_prop_fig = zt_prop_fig + p9.stat_summary(p9.aes(x='zt_time_hour', y='rel_time_behavior',group='Unique_animal', color='Strain'), geom=p9.geom_line, fun_y=np.mean, alpha=0.3) + \
		p9.geom_vline(xintercept=12, linetype='dashed', alpha=0.5, color='red') + \
		custom_theme + p9.labs(title=f"Individual {behavior} Behavior by ZT (n={n_arenas} arenas)")
	try:
		plot_zt_prop_fig.save(os.path.join(output_dir, f"individual_prop_zt_{behavior}.svg"))
	except Exception as e:
		print(f"Error saving individual_prop_zt_{behavior}.svg: {e}")

	zt_bout_num_fig = generate_time_vs_feature_plot("Average Number of Bouts", "ZT hour", "delete", filter_experiment_time(df,num_hours=12), 'zt_time_hour', 'bout_behavior', draw_data=False, title=header_data['Behavior'][0], save_files=False)
	plot_zt_bout_num_fig = zt_bout_num_fig + p9.stat_summary(p9.aes(x='zt_time_hour', y='bout_behavior',group='Unique_animal', color='Strain'), geom=p9.geom_line, fun_y=np.mean, alpha=0.3) + \
		p9.geom_vline(xintercept=12, linetype='dashed', alpha=0.5, color='red') + \
		custom_theme + p9.labs(title=f"Individual Bout Numbers by ZT (n={n_arenas} arenas)")
	try:
		plot_zt_bout_num_fig.save(os.path.join(output_dir, f"individual_numbout_zt_{behavior}.svg"))
	except Exception as e:
		print(f"Error saving individual_numbout_zt_{behavior}.svg: {e}")

	zt_bout_length_fig = generate_time_vs_feature_plot("Average Bout Length", "ZT hour", "delete", filter_experiment_time(df,num_hours=12),'zt_time_hour', 'avg_bout_length_sec', draw_data=False, title=header_data['Behavior'][0], save_files=False)
	plot_zt_bout_length_fig = zt_bout_length_fig + p9.stat_summary(p9.aes(x='zt_time_hour', y='avg_bout_length_sec',group='Unique_animal', color='Strain'), geom=p9.geom_line, fun_y=np.mean, alpha=0.3) + \
		p9.geom_vline(xintercept=12, linetype='dashed', alpha=0.5, color='red') + \
		custom_theme + p9.labs(title=f"Individual Bout Lengths by ZT (n={n_arenas} arenas)")
	try:
		plot_zt_bout_length_fig.save(os.path.join(output_dir, f"individual_boutlen_zt_{behavior}.svg"))
	except Exception as e:
		print(f"Error saving individual_boutlen_zt_{behavior}.svg: {e}")


def main(argv):
	import argparse
	parser = argparse.ArgumentParser(description="Generate behavior plots for multi-day behavior analysis.")
	parser.add_argument('--behavior', type=str, required=True, help='Name of the behavior (e.g., Drinking)')
	parser.add_argument('--results_file', type=str, required=True, help='Path to the summary results CSV file')
	parser.add_argument('--jmcrs_data', type=str, required=False, default='/projects/kumar-lab/choij/lepr_poses/2023-09-07 TOM_TotalQueryForConfluence.xlsx', help='Path to the JCMS metadata file (Excel). Defaults to the 2023-09-07 file if not specified.')
	parser.add_argument('--remove_experiments', type=str, default='', help='Comma-separated list of experiment IDs to remove (e.g., MDB0003,MDX0008)')
	parser.add_argument('--output_dir', type=str, required=True, help='Output directory for all plot files (will be created if it does not already exist)')
	args = parser.parse_args(argv)

	remove_experiments = [x.strip() for x in args.remove_experiments.split(',') if x.strip()] if args.remove_experiments else []

	generate_behavior_plots(
		behavior=args.behavior,
		results_file=args.results_file,
		jmcrs_data=args.jmcrs_data,
		remove_experiments=remove_experiments,
		output_dir=args.output_dir
	)

if __name__ == "__main__":
	import sys
	main(sys.argv[1:])
