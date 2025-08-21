import numpy as np
import pandas as pd
import plotnine as p9
from mizani import formatters

# Plotnine plot objects, which are returned here can be modified, plotted (blocking), plotted (non-blocking), or saved directly to disk as a figure (png or svg)
# New layers can be added to an existing plot through the + operator
# Plots can be plotted either using print(plot) [blocking] or plot.draw().show() [non-blocking]
# Plots can be saved to disk using plot.save('filename.ext', ...) see help(plot.save) for options


# Plotting time vs feature with groupings
# Generates a plotnine figure (which can be modified after returned)
# Handles the formatting under the hood
# If you want to remove the data plotted in favor of something else (eg points), pass draw_data=False
def generate_time_vs_feature_plot(
    df: pd.DataFrame,
    time: str = "zt_time_hour",
    feature: str = "rel_time_behavior",
    factor: str = "Strain",
    draw_data: bool = True,
    title: str = None,
):
    # Detect the time datatype
    col_types = df.dtypes
    df_copy = pd.DataFrame.copy(df)
    if not isinstance(col_types[factor], pd.CategoricalDtype):
        df_copy[factor] = df_copy[factor].astype("category")
    # Make a custom df for the lights block
    light_df = df.groupby([time, factor])[[feature, "lights_on"]].mean().reset_index()
    # Max across the factor
    light_df = light_df.groupby(time)[[feature, "lights_on"]].max().reset_index()
    light_df["lights_val"] = (
        (1 - light_df["lights_on"]) * 1.1 * np.max(light_df[feature])
    )
    if pd.api.types.is_timedelta64_dtype(
        col_types[time]
    ) or pd.api.types.is_timedelta64_ns_dtype(col_types[time]):
        _light_width = 60 * 60 * 10**9
    else:
        _light_width = 1
    # Start building the plot
    plot = p9.ggplot(df)
    # Add in the line + background
    if draw_data:
        # Plot the background light rectangles first
        # plot = plot + p9.geom_bar(p9.aes(x=time, y='lights_val'), light_df, width=light_width, stat='identity', fill='lightgrey')
        plot = plot + p9.stat_summary(
            p9.aes(x=time, y=feature, color=factor, fill=factor),
            fun_y=np.mean,
            geom=p9.geom_point,
        )
        plot = plot + p9.stat_summary(
            p9.aes(x=time, y=feature, color=factor, fill=factor),
            fun_ymin=lambda x: np.mean(x) - np.std(x) / np.sqrt(len(x)),
            fun_ymax=lambda x: np.mean(x) + np.std(x) / np.sqrt(len(x)),
            fun_y=np.mean,
            geom=p9.geom_smooth,
        )
    # Clean up some formatting
    plot = plot + p9.theme_bw()
    # Try to handle the different types of times
    # With full datetime, rotate
    if pd.api.types.is_datetime64_any_dtype(col_types[time]):
        plot = plot + p9.theme(axis_text_x=p9.element_text(rotation=90, hjust=0.5))
    # Timedelta, rotate and force breaks to hour format
    elif pd.api.types.is_timedelta64_dtype(
        col_types[time]
    ) or pd.api.types.is_timedelta64_ns_dtype(col_types[time]):
        plot = (
            plot
            + p9.theme(axis_text_x=p9.element_text(rotation=90, hjust=0.5))
            + p9.scale_x_timedelta(labels=formatters.timedelta_format("h"))
        )
        # breaks=mizani.breaks.timedelta_breaks(n_breaks)
    #
    if title is not None:
        plot = plot + p9.labs(title=title, color=factor, y=feature)
    else:
        plot = plot + p9.labs(color=factor, y=feature)
    plot = plot + p9.scale_color_brewer(type="qual", palette="Set1")
    plot = plot + p9.scale_fill_brewer(type="qual", palette="Set1", guide=False)
    return plot
