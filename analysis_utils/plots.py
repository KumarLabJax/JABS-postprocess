import numpy as np
import pandas as pd
import plotnine as p9
import mizani

# Plotnine plot objects, which are returned here can be modified, plotted (blocking), plotted (non-blocking), or saved directly to disk as a figure (png or svg)
# New layers can be added to an existing plot through the + operator
# Plots can be plotted either using print(plot) [blocking] or plot.draw().show() [non-blocking]
# Plots can be saved to disk using plot.save('filename.ext', ...) see help(plot.save) for options

# Plotting time vs feature with groupings
# Generates a plotnine figure (which can be modified after returned)
# Handles the formatting under the hood
# If you want to remove the data plotted in favor of something else (eg points), pass draw_data=False
# TODO: Add in the light cycle blocks based on detected 'lights_on' field
def generate_time_vs_feature_plot(df: pd.DataFrame, time: str='zt_time_hour', feature: str='rel_time_behavior', factor: str='Strain', draw_data=True):
	col_types = df.dtypes
	df_copy = pd.DataFrame.copy(df)
	if not pd.api.types.is_categorical_dtype(col_types[factor]):
		df_copy[factor] = df_copy[factor].astype('category')
	# Start building the plot
	plot = p9.ggplot(df, p9.aes(x=time, y=feature, color=factor, fill=factor))
	# Add in the line + background
	if draw_data:
		plot = plot + p9.stat_summary(fun_ymin=lambda x: np.mean(x)-np.std(x)/np.sqrt(len(x)), fun_ymax=lambda x: np.mean(x)+np.std(x)/np.sqrt(len(x)), fun_y=np.mean, geom=p9.geom_smooth)
	# Clean up some formatting
	plot = plot + p9.theme_bw()
	# Try to handle the different types of times
	# With full datetime, rotate
	if pd.api.types.is_datetime64_any_dtype(col_types[time]):
		plot = plot + p9.theme(axis_text_x=p9.element_text(rotation=90, hjust=0.5))
	# Timedelta, rotate and force breaks to hour format
	elif pd.api.types.is_timedelta64_dtype(col_types[time]) or pd.api.types.is_timedelta64_ns_dtype(col_types[time]):
		plot = plot + p9.theme(axis_text_x=p9.element_text(rotation=90, hjust=0.5)) + p9.scale_x_timedelta(labels=mizani.formatters.timedelta_format('h'))
		# breaks=mizani.breaks.timedelta_breaks(n_breaks)
	# 
	plot = plot + p9.labs(color=factor)
	plot = plot + p9.scale_color_brewer(type='qual', palette='Set1')
	plot = plot + p9.scale_fill_brewer(type='qual', palette='Set1', guide=False)
	return plot
