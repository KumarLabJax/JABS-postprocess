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
    filtered_phase_df = phase_df
    for instance in phase_df.index: 
        freq_list = phase_df['freq'][instance]
        mask = np.ones(len(freq_list), dtype=bool)
        for freq in freq_list: 
            if freq >= 1/22 or freq <= 1/25: 
                index = phase_df['freq'][instance].tolist().index(freq)
                mask[index] = False
        filtered_phase_df['freq'][instance] = freq_list[mask,...]
        filtered_phase_df['amplitude'][instance] = filtered_phase_df['amplitude'][instance][mask,...]
        filtered_phase_df['period'][instance] = filtered_phase_df['period'][instance][mask,...]
        filtered_phase_df['phase'][instance] = filtered_phase_df['phase'][instance][mask,...]

    # Find dominant period by taking the period associated with the max amplitude
    period_df = pd.DataFrame(columns=['mouse_id', 'dominant_period', 'amplitude', 'strain'])
    ray_df = pd.DataFrame(columns=['mouse_id', 'peak_phase', 'amplitude', 'strain'])
    for mouse in phase_df.index: 
        amp_list = phase_df['amplitude'][mouse].tolist()
        if type(amp_list) == float:
            max_amp = amp_list
            dom_period = phase_df['period'][mouse].tolist()
            peak_phase = phase_df['period'][mouse].tolist()
        else:
            max_amp = max(amp_list)
            max_index = amp_list.index(max_amp)
            dom_period = phase_df['period'][mouse][max_index]
            peak_phase = phase_df['phase'][mouse][max_index]
        tmp = {'mouse_id': mouse, 'dominant_period': dom_period, 'amplitude': max_amp, 'strain': phase_df['strain'][mouse]}
        tmp2 = {'mouse_id': mouse, 'peak_phase': peak_phase, 'amplitude': max_amp, 'strain': phase_df['strain'][mouse]}
        tmp = pd.DataFrame([tmp])
        tmp2 = pd.DataFrame([tmp2])
        period_df = pd.concat([period_df, tmp])
        ray_df = pd.concat([ray_df, tmp2])

    # Plot the dominant periods
    (p9.ggplot(period_df, p9.aes(x='dominant_period')) + 
        p9.geom_histogram(binwidth=0.5, fill=BLUE, color='black') + 
        p9.theme_bw() + 
        p9.labs(x='Dominant Period', y='Count', title='Dominant Period Comparison') + 
        p9.theme(title = p9.element_text(hjust=0.5))
        ).save('period_hist.png')

    #=== SHOW THE PHASE SYNCHRONY ===#

    def pol2cart(rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return x, y

    def cart2pol(x, y):
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return(rho, phi)

    # Put the phase values on a scale of 0 to 2pi for plotting 
    ray_df['peak_phase'] = ray_df['peak_phase'] % (2*np.pi)
    # Shift the phase values by pi/2 to show the peak time of eating rather than when they begin to move towards their peak time of eating
    ray_df['peak_phase'] = ray_df['peak_phase'] + (np.pi)/2
    ray_df['r'] = 1
    ray_df['x'], ray_df['y'] = pol2cart(ray_df['r'], ray_df['peak_phase'])

    b6_df = ray_df[ray_df['strain'] == 'C57BL/6J']
    btbr_df = ray_df[ray_df['strain'] == 'BTBR T<+> ltpr3<tf>/J']

    center_b6x = sum(b6_df['x'])/len(b6_df['x'])
    center_b6y = sum(b6_df['y'])/len(b6_df['y'])
    center_btbrx = sum(btbr_df['x'])/len(btbr_df['x'])
    center_btbry = sum(btbr_df['y'])/len(btbr_df['y'])

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