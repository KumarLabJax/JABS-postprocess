import pandas as pd
import plotnine as p9
import re
import sys
import numpy as np
from itertools import chain
from analysis_utils.parse_table import read_ltm_summary_table
from analysis_utils.circadian import to_fraction_str, make_phase_df
import plotly.graph_objects as go
from plotly.subplots import make_subplots 
import argparse

RED = '#e41a1c'
BLUE = '#377eb8'

# Run this line before starting up the interactive python session for accessing libraries
# export PYTHONPATH=/JABS-postprocess/
# Alternatively, now we can use the singularity image at /projects/kumar-lab/JABS/JABS-Postprocessing-2023-02-07.sif

def main(argv): 
    parser = argparse.ArgumentParser(description='Runs circadian phase analysis on a single behavior')
    parser.add_argument('--behavior', help='Name of behavior to be analyzed', required=True)
    parser.add_argument('--results_file', help='Path to results file containing behavior prediction data', required=True)
    parser.add_argument('--jmcrs_data', help='Path to the metadata for the mouse experiments', required=True)
    parser.add_argument('--filter_lixit', help='File containing experiments to filter out videos with lixit problems', default=None)
    parser.add_argument('--filter_food_hopper', help='File containing experiments to filter out videos with food hopper problems', default=None)
    args = parser.parse_args()
    run_analysis(args)


def run_analysis(args):
    #-------------------------------
    # Example multi-day summary plot
    #-------------------------------
    # Read in the summary results
    results_file = args.results_file
    jmcrs_data = args.jmcrs_data
    _, df = read_ltm_summary_table(results_file, jmcrs_metadata=jmcrs_data)

    if args.filter_food_hopper:
        # Filter out bad videos for certain behavior
        filter_out = pd.read_csv(args.filter_food_hopper).drop_duplicates().values.tolist()
        filter_list = list(chain.from_iterable(filter_out))
        df = df[~df['exp_prefix'].isin(filter_list)]

    if args.filter_lixit:
        # Filter out experiments with lixit problems 
        filter_out = pd.read_csv(args.filter_lixit, header=None)
        filter_out = [re.sub('.*(MD[XB][0-9]+).*', '\\1', x) for x in filter_out[0]]
        df = df[~np.isin(df['ExptNumber'], filter_out)]

    # Delete out bins where no data exists
    no_data = np.all(df[['time_no_pred', 'time_not_behavior', 'time_behavior']] == 0, axis=1)
    df = df[~no_data].reset_index()

    phase_df = make_phase_df(df, 'bout_behavior', trim_start=1, trim_end=1)

    #=== THIS PLOT SHOWS THE DOMINANT AMPLITUDE ===#

    # Plot the frequencies and the resulting amplitudes from the fft 
    (
        p9.ggplot(phase_df, p9.aes(x='freq', y='amplitude', group='group')) 
        + p9.geom_line(alpha=0.15, color=BLUE)
        + p9.theme_bw()
        + p9.labs(title='Eating Circadian Power Spectral Density', x='Frequency', y='Amplitude')
        + p9.theme(title=p9.element_text(hjust=0.5))
        + p9.scale_x_continuous(labels=list(map(to_fraction_str, [(0, 0), (1, 24), (1, 12), (1, 8), (1, 6), (5, 24), (1, 4), (7, 24), (1, 3), (3, 8), (5, 12)])), breaks=[0, 1 / 24, 1 / 12, 1 / 8, 1 / 6, 5 / 24, 1 / 4, 7 / 24, 1 / 3, 3 / 8, 5 / 12])
        + p9.coord_cartesian(xlim=(0, 5 / 12))
    ).save('freq_amp.png')

    #Filter out the frequencies below or above the allowed range
    filtered_phase_df = phase_df[phase_df['freq'].between(1/25, 1/22)]

    # Find dominant period by taking the period associated with the max amplitude
    peak_df = filtered_phase_df.groupby('group').apply(lambda x: x.iloc[np.argmax(x['amplitude'])]).reset_index(drop=True)

    # Plot the dominant periods
    (
        p9.ggplot(peak_df, p9.aes(x='period'))
        + p9.geom_histogram(binwidth=0.5, fill=BLUE, color='black')
        + p9.theme_bw()
        + p9.labs(x='Dominant Period', y='Count', title='Dominant Period Comparison')
        + p9.theme(title=p9.element_text(hjust=0.5))
    ).save('period_hist.png')

    #=== SHOW THE PHASE SYNCHRONY ===#

    def pol2cart(rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return x, y

    def cart2pol(x, y):
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return (rho, phi)

    # Shift the phase values by pi/2 to show the peak time of eating rather than when they begin to move towards their peak time of eating
    peak_df['phase'] = peak_df['phase'] + (np.pi) / 2
    # Put the phase values on a scale of 0 to 2pi for plotting
    peak_df['phase'] = peak_df['phase'] % (2 * np.pi)
    peak_df['r'] = 1
    peak_df['x'], peak_df['y'] = pol2cart(peak_df['r'], peak_df['phase'])

    b6_df = peak_df[peak_df['strain'] == 'C57BL/6J']
    btbr_df = peak_df[peak_df['strain'] == 'BTBR T<+> ltpr3<tf>/J']

    center_b6x = sum(b6_df['x']) / len(b6_df['x'])
    center_b6y = sum(b6_df['y']) / len(b6_df['y'])
    center_btbrx = sum(btbr_df['x']) / len(btbr_df['x'])
    center_btbry = sum(btbr_df['y']) / len(btbr_df['y'])

    r_b6, theta_b6 = cart2pol(center_b6x, center_b6y)
    r_btbr, theta_btbr = cart2pol(center_btbrx, center_btbry)

    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'polar'}]*2]*1)

    fig.add_trace(go.Scatterpolar(
            name = 'C57BL/6J', 
            r = b6_df['r'],
            theta = b6_df['peak_phase'], 
            thetaunit = 'radians',
            mode = 'markers',
            marker = dict(color = 'blue')
        ), 1, 1)
    fig.add_trace(go.Scatterpolar(
            name = "BTBR T<+> ltpr3<tf>/J",
            r = btbr_df['r'],
            theta = btbr_df['peak_phase'],
            thetaunit = 'radians',
            mode = 'markers', 
            marker = dict(color = 'red')
        ), 1, 2)
    fig.update_layout(
        title = "Peak Phase Comparion Between Strains",
        polar = dict(
            radialaxis = dict(tickvals = []),
            angularaxis = dict(
                thetaunit = 'radians',
                dtick = 45,
                rotation = 90,
                direction = 'clockwise',
                tickmode = 'array', 
                tickvals = [0, 90, 180, 270],
                ticktext = ['0', '6', '12', '18'])
        ),
        polar2 = dict(
            radialaxis = dict(tickvals = []),
            angularaxis = dict(
                thetaunit = 'radians',
                dtick = 45,
                rotation = 90,
                direction = 'clockwise',
                tickmode = 'array', 
                tickvals = [0, 90, 180, 270],
                ticktext = ['0', '6', '12', '18']
            )
        ))
    fig.add_trace(go.Scatterpolar(
            name = "Mean Vector C57BL/6J",
            r = [0, r_b6], 
            theta = [0, theta_b6],
            thetaunit = 'radians', 
            mode = 'lines',
            marker = dict(color = 'blue')
        ), 1, 1)
    fig.add_trace(go.Scatterpolar(
            name = "Mean Vector BTBR T<+> ltpr3<tf>/J",
            r = [0, r_btbr], 
            theta = [0, theta_btbr], 
            thetaunit = 'radians', 
            mode = 'lines', 
            marker = dict(color = 'red')
        ), 1, 2)

    fig.write_image('raleigh.png')

if __name__ == '__main__':
    main(sys.argv[1:])