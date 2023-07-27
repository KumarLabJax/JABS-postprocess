import pandas as pd
import re
import numpy as np
import scipy
from itertools import chain
from analysis_utils.parse_table import read_ltm_summary_table
from plotnine import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pycircstat import *
import argparse, sys

# Run this line before starting up the interactive python session for accessing libraries
# export PYTHONPATH=/JABS-postprocess/
# Alternatively, now we can use the singularity image at /projects/kumar-lab/JABS/JABS-Postprocessing-2023-02-07.sif

RED = '#e41a1c'
BLUE = '#377eb8'

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

def main(argv): 
    parser = argparse.ArgumentParser(description="Compares the circadian phase FFT analysis for two behaviors")
    parser.add_argument('--first_behavior', help='Name of First Behavior', required=True)
    parser.add_argument('--second_behavior', help='Name of Second Behavior', required=True)
    parser.add_argument('--first_results', help='Path to the results file conatining prediction data for the first behavior', required=True)
    parser.add_argument('--second_results', help='Path to the results file containing prediction data for the second behavior', required=True)
    parser.add_argument('--jmcrs_data', help='Path to the metadata for the mouse experiments', required=True)
    parser.add_argument('--filter_lixit', help='Whether or not to filter out videos with lixit problems', default=False, action='store_true')
    parser.add_argument('--filter_food_hopper', help='Whether or not to filter out videos with food hopper problems', default=False, action='store_true')
    args = parser.parse_args()
    run_analysis(args)

def run_analysis(args):
    # This code compares the circadian phase analysis of two behaviors\

    #-------------------------------
    # Example multi-day summary plot
    #-------------------------------
    # Read in the summary results
    BEHAVIOR1 = args.first_behavior
    BEHAVIOR2 = args.second_behavior
    results_file1 = args.first_results
    results_file2 = args.second_results
    jmcrs_data = args.jmcrs_data
    _, df1 = read_ltm_summary_table(results_file1, jmcrs_metadata=jmcrs_data)
    _, df2 = read_ltm_summary_table(results_file2, jmcrs_metadata=jmcrs_data)

    if args.filter_food_hopper: 
        # Filter out experiments with food hopper problems
        filter_out = pd.read_csv('/Users/hamilc/Jax/analysis_figures/male/to_remove.csv').drop_duplicates().values.tolist()
        filter_list = list(chain.from_iterable(filter_out))
        df1 = df1[~df1['exp_prefix'].isin(filter_list)]
        df2 = df2[~df2['exp_prefix'].isin(filter_list)]

    if args.filter_lixit:
        # Filter out experiments with lixit problems 
        filter_out = pd.read_csv('/Users/hamilc/Jax/analysis_figures/male/questionable_lixit.csv', header=None)
        filter_out = [re.sub('.*(MD[XB][0-9]+).*', '\\1', x) for x in filter_out[0]]
        df1 = df1[~np.isin(df1['ExptNumber'], filter_out)]
        df2 = df2[~np.isin(df2['ExptNumber'], filter_out)]

    # Delete out bins where no data exists
    no_data = np.all(df1[['time_no_pred','time_not_behavior','time_behavior']]==0, axis=1)
    df1 = df1[~no_data].reset_index()
    no_data = np.all(df2[['time_no_pred','time_not_behavior','time_behavior']]==0, axis=1)
    df2 = df2[~no_data].reset_index()

    # Give each entry a unique id based on their experiment, id, and behavior
    df1["Unique_animal"] = df1['longterm_idx'].astype(str) + df1['ExptNumber'] + 'a'
    df2["Unique_animal"] = df2['longterm_idx'].astype(str) + df2['ExptNumber'] + 'b'
    df = pd.concat([df1, df2])

    # Filter out one experiment over one day for poster plot
    df1 = df[np.isin(df['ExptNumber'], ["MDB0049", "MDB0015"])]
    df1 = df1[df1['zt_exp_time'].dt.days == 3]
    df['avg_bout_length'] = df['time_behavior']/df['bout_behavior']

    # Amplitude and Phase
    # using FFT for Amplitude and phase information
    # function for using fft to get the A and phase of the signals
    def get_fft_amplitude_and_phase_scipy(data, fs):
        N = 512 # number of sample points
        freq = scipy.fft.fftfreq(N, 1/fs)
        fft_vals = scipy.fft.fft(data - np.mean(data), n=N) # Since the power of signal lies in the symmetric part of the fft spectrum, we consider only the positive half
        mask = freq > 0
        freq = freq[mask]
        period = 1/freq
        # calc the amplitude and phase
        amplitude = np.abs(fft_vals[mask]) / N # abs(Y)/npts
        phase = np.angle(fft_vals[mask]) # np.angle(Y) 
        return freq, amplitude, phase, period

    # Define the frequency sampling and cutoff frequency
    fs = 1.0  # sampling frequency: 1 data point per hour
    order = 5 # Order of the Butterworth filter
    filtered_data_dict = {}
    amplitude_phase_dict_eat = {}
    amplitude_phase_dict_drink = {}

    # removing the first and last video per experiment
    df_new = df.groupby('Unique_animal').apply(lambda group: group.iloc[1:-1]).reset_index(drop=True)
    # Removing experiments MDX0017 and MDX0005 because it has missing data
    df_new = df_new[df_new['ExptNumber'] != 'MDX0017']
    df_new = df_new[df_new['ExptNumber'] != 'MDX0005']

    # Process the FFT for each animal for each behavior
    for mouse in df_new['Unique_animal'].unique():
        df_mouse = df_new[df_new['Unique_animal'] == mouse]
        temp = pd.Series(data=df_mouse['bout_behavior'].values,index=df_mouse['relative_exp_time'].values)
        temp = temp.fillna(method='bfill')
        temp = temp.resample('H').sum()
        if len(temp) > 3 * order:  # 3*order is the default padlen in scipy's filtfilt. so making sure that there are enough data points for the filtfilt function to work properly
            freq, amplitude, phase, period = get_fft_amplitude_and_phase_scipy(temp.values, fs)
            amplitude_phase_dict_eat[mouse] = {'freq': freq, 'amplitude': amplitude, 'phase': phase, 'period': period, 'mouse': mouse, 'strain': np.unique(df_mouse['Strain'])[0], 'behavior': np.unique(df_mouse['behavior'])[0], 'room': np.unique(df_mouse['Room'])[0]}
        else:
            print(f'Skipping mouse {mouse} due to insufficient data points.')

    phase_df = pd.DataFrame.from_dict(amplitude_phase_dict_eat, orient='index')

    def to_fraction(x):
        if x[0] == 0: 
            return 0
        else: 
            return '{}/{}'.format(x[0], x[1])

    plot_df = phase_df.explode(['freq', 'amplitude', 'phase', 'period'])
    plot_df[['freq', 'amplitude', 'phase', 'period']] = plot_df[['freq', 'amplitude', 'phase', 'period']].astype(float)
    (ggplot(plot_df, aes(x='freq', y='amplitude', group='mouse')) + 
            geom_line(alpha = 0.15, color=BLUE) + 
            theme_bw() + 
            labs(title='Eating Circadian Power Spectral Density', x='Frequency', y='Amplitude') + 
            theme(title = element_text(hjust = 0.5)) + 
            scale_x_continuous(labels=list(map(to_fraction, [(0,0), (1,24), (1,12), (1,8), (1,6), (5,24), (1,4), (7,24), (1,3), (3,8), (5,12)])), breaks=[0, 1/24, 1/12, 1/8, 1/6, 5/24, 1/4, 7/24, 1/3, 3/8, 5/12]) + 
            coord_cartesian(xlim=(0, 5/12)) + 
            facet_wrap('behavior')).save('freq_amp.png')

    # Filter the data so as to only focus on the circadian frequencies
    for instance in phase_df.index: 
        freq_list = phase_df['freq'][instance]
        mask = np.ones(len(freq_list), dtype=bool)
        for freq in freq_list: 
            if freq >= 1/22 or freq <= 1/25: 
                index = phase_df['freq'][instance].tolist().index(freq)
                mask[index] = False
        phase_df['freq'][instance] = freq_list[mask,...]
        phase_df['amplitude'][instance] = phase_df['amplitude'][instance][mask,...]
        phase_df['period'][instance] = phase_df['period'][instance][mask,...]
        phase_df['phase'][instance] = phase_df['phase'][instance][mask,...]

    # Find dominant period and phase by taking the period associated with the max amplitude
    period_df = pd.DataFrame(columns=['mouse_id', 'dominant_period', 'amplitude', 'strain'])
    ray_df = pd.DataFrame(columns=['mouse_id', 'peak_phase', 'amplitude', 'strain'])
    for mouse in phase_df.index: 
        # Get the amplitudes for each mouse in the dataframe
        amp_list = phase_df['amplitude'][mouse].tolist()
        # If only one value 
        if type(amp_list) == float:
            max_amp = amp_list
            dom_period = phase_df['period'][mouse].tolist()
            peak_phase = phase_df['period'][mouse].tolist()
        else:
            # Find the index of the max amplitude and set the associated peak phase and period
            max_amp = max(amp_list)
            max_index = amp_list.index(max_amp)
            dom_period = phase_df['period'][mouse][max_index]
            peak_phase = phase_df['phase'][mouse][max_index]
        tmp = {'mouse_id': mouse, 'dominant_period': dom_period, 'amplitude': max_amp, 'strain': phase_df['strain'][mouse], 'behavior': phase_df['behavior'][mouse]}
        tmp2 = {'mouse_id': mouse, 'peak_phase': peak_phase, 'amplitude': max_amp, 'strain': phase_df['strain'][mouse], 'behavior': phase_df['behavior'][mouse]}
        tmp = pd.DataFrame([tmp])
        tmp2 = pd.DataFrame([tmp2])
        period_df = pd.concat([period_df, tmp])
        ray_df = pd.concat([ray_df, tmp2])

    # Plot the dominant periods
    (ggplot(period_df, aes(x='dominant_period')) + 
        geom_histogram(binwidth=0.5, fill=BLUE, color='black') + 
        theme_bw() + 
        labs(x='Dominant Period', y='Count', title='Dominant Period Comparison') + 
        theme(title = element_text(hjust=0.5))
        ).save('period_hist.png')

    def pol2cart(rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return x, y

    def cart2pol(x, y):
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return(rho, phi)

    # Calculate the phase difference for the two behaviors, put in new dataframe called phase_diff_df
    ray_df.set_index('mouse_id')
    phase_diff_df = pd.DataFrame(columns=['mouse_id', 'phase_diff', 'amplitude1', 'amplitude2', 'strain', 'phase1', 'phase2'])
    for mouse in np.unique(ray_df[ray_df['behavior'] == BEHAVIOR1]['mouse_id']):
        mouse_id = mouse[0:-1]
        if mouse_id + 'b' not in np.unique(ray_df['mouse_id']):
            continue
        phase1 = ray_df[ray_df['mouse_id'] == mouse_id + 'a']['peak_phase'].values[0]
        phase2 = ray_df[ray_df['mouse_id'] == mouse_id + 'b']['peak_phase'].values[0]
        phase1 = phase1 % (2*np.pi)
        phase2 = phase2 % (2*np.pi)
        phase_diff = phase1 - phase2
        amplitude1 = ray_df[ray_df['mouse_id'] == mouse_id + 'a']['amplitude'].values[0]
        amplitude2 = ray_df[ray_df['mouse_id'] == mouse_id + 'b']['amplitude'].values[0]
        strain = ray_df[ray_df['mouse_id'] == mouse]['strain'].values[0]
        tmp = {'mouse_id': mouse_id, 'phase_diff': phase_diff, 'amplitude1': amplitude1, 'amplitude2': amplitude2, 'strain': strain, 'phase1': phase1, 'phase2': phase2}
        tmp = pd.DataFrame([tmp])
        phase_diff_df = pd.concat([phase_diff_df, tmp])

    # Put the phase values on a scale of 0 to 2pi for plotting 
    ray_df['peak_phase'] = ray_df['peak_phase'] % (2*np.pi)
    # Shift the phase values by pi/2 to show the peak time of eating rather than when they begin to move towards their peak time of eating
    ray_df['peak_phase'] = ray_df['peak_phase'] + (np.pi)/2
    # Radius of 1 as default for rayleigh plot
    ray_df['r'] = 1
    ray_df['x'], ray_df['y'] = pol2cart(ray_df['r'], ray_df['peak_phase'])
    #ray_df[['mouse_id', 'peak_phase', 'strain', 'behavior']].to_csv('phase_analysis.csv')

    # Split the data by behavior
    ray_df1 = ray_df[ray_df['behavior'] == BEHAVIOR1]
    ray_df2 = ray_df[ray_df['behavior'] == BEHAVIOR2]

    # Split the data further by strain
    b6_df1 = ray_df1[ray_df1['strain'] == 'C57BL/6J']
    b6_df2 = ray_df2[ray_df2['strain'] == 'C57BL/6J']
    btbr_df1 = ray_df1[ray_df1['strain'] == 'BTBR T<+> ltpr3<tf>/J']
    btbr_df2 = ray_df2[ray_df2['strain'] == 'BTBR T<+> ltpr3<tf>/J']

    # Calculate centroids to determine mean phase and vector length 
    center_b6_x1 = sum(b6_df1['x'])/len(b6_df1['x'])
    center_b6_y1 = sum(b6_df1['y'])/len(b6_df1['y'])
    center_b6_x2 = sum(b6_df2['x'])/len(b6_df2['x'])
    center_b6_y2 = sum(b6_df2['y'])/len(b6_df2['y'])
    center_btbr_x1 = sum(btbr_df1['x'])/len(btbr_df1['x'])
    center_btbr_y1 = sum(btbr_df1['y'])/len(btbr_df1['y'])
    center_btbr_x2 = sum(btbr_df2['x'])/len(btbr_df2['x'])
    center_btbr_y2 = sum(btbr_df2['y'])/len(btbr_df2['y'])

    # Convert centroids back to polar coordinates for plotting
    r_b6_1, theta_b6_1 = cart2pol(center_b6_x1, center_b6_y1)
    r_b6_2, theta_b6_2 = cart2pol(center_b6_x2, center_b6_y2)
    r_btbr_1, theta_btbr_1 = cart2pol(center_btbr_x1, center_btbr_y1)
    r_btbr_2, theta_btbr_2 = cart2pol(center_btbr_x2, center_btbr_y2)

    # Statistics to determine if there is a significant difference between behavior distributions for each strain
    pval_b6, k_b6 = kuiper(b6_df1['peak_phase'], b6_df2['peak_phase'])
    pval_btbr, k_btbr = kuiper(btbr_df1['peak_phase'], btbr_df2['peak_phase'])

    # Statistics to determine if there is a significant difference between strains for each behavior
    pval_1, k_1 = kuiper(b6_df1['peak_phase'], btbr_df1['peak_phase'])
    pval_2, k_2 = kuiper(b6_df2['peak_phase'], btbr_df2['peak_phase'])
    with open('stats.txt', 'w') as outfile:
        outfile.write('Distribution comparisons using the kuiper test: \n')
        outfile.write(f'P-val that b6 {BEHAVIOR1} and {BEHAVIOR2} are the same distribution: {pval_b6[0]}\n')
        outfile.write(f'P-val that BTBR {BEHAVIOR1} and {BEHAVIOR2} are the same distribution: {pval_btbr[0]}\n')
        outfile.write(f'P-val that the {BEHAVIOR1} phases for btbr and b6js are the same distribution: {pval_1[0]}\n')
        outfile.write(f'P-val that the {BEHAVIOR2} phases for btbr and b6js are the same distribution: {pval_2[0]}\n\n')

    b6pval, T = watson_williams(b6_df1['peak_phase'], b6_df2['peak_phase'])
    btbrpval, T2 = watson_williams(btbr_df1['peak_phase'], btbr_df2['peak_phase'])
    pval1, T3 = watson_williams(b6_df1['peak_phase'], btbr_df1['peak_phase'])
    pval2, T4 = watson_williams(b6_df2['peak_phase'], btbr_df2['peak_phase'])
    with open('stats.txt', 'a') as outfile: 
        outfile.write('Distribution mean comparisons using the watson williams test: \n')
        outfile.write(f'P-val that the distributions for b6 {BEHAVIOR1} and {BEHAVIOR2} have the same mean: {b6pval}\n')
        outfile.write(f'P-val that the distributions for btbr {BEHAVIOR1} and {BEHAVIOR2} have the same mean: {btbrpval}\n')
        outfile.write(f'P-val that the distributions for {BEHAVIOR1} between b6 and btbr have the same mean: {pval1}\n')
        outfile.write(f'P-val that the distributions for {BEHAVIOR2} between b6 and btbr have the same mean: {pval2}\n\n')


    # Plot the compared distributions as a way of validating the statistical results
    (ggplot(ray_df1, aes(x='peak_phase', color='strain')) + 
        geom_density() + 
        theme_bw() + 
        labs(title=f'Phase Distribution Comparison for {BEHAVIOR1}')).save(f'{BEHAVIOR1}_phase_strain_comp.png')

    (ggplot(ray_df2, aes(x='peak_phase', color='strain')) + 
        geom_density() + 
        theme_bw() + 
        labs(title=f'Phase Distribution Comparison for {BEHAVIOR2}')).save(f'{BEHAVIOR2}_phase_strain_comp.png')

    (ggplot(ray_df[ray_df['strain'] == 'C57BL/6J'], aes(x='peak_phase', color='behavior')) + 
        geom_density() + 
        theme_bw() + 
        labs(title='Phase Distribution Comparison for BL6')).save('b6_behavior_phase_comp.png')

    (ggplot(ray_df[ray_df['strain'] == 'BTBR T<+> ltpr3<tf>/J'], aes(x='peak_phase', color='behavior')) + 
        geom_density() + 
        theme_bw() + 
        labs(title='Phase Distribution Comparison for BTBR')).save('btbr_behavior_phase_comp.png')

    # Alter the radii for rayleigh plot visualization purposes
    b6_df1['r'] = 1
    b6_df2['r'] = 1.07
    btbr_df1['r'] = 1
    btbr_df2['r'] = 1.07

    # Rayleigh Plot 
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'polar'}]*2]*1, subplot_titles=('C5JBL/6J', 'BTBR T<+> ltpr3<tf>/J'))

    fig.add_trace(go.Scatterpolar(
            name = BEHAVIOR1, 
            r = b6_df1['r'],
            theta = b6_df1['peak_phase'], 
            thetaunit = 'radians',
            mode = 'markers',
            marker = dict(color = RED)
        ), 1, 1)
    fig.add_trace(go.Scatterpolar(
            r = btbr_df1['r'],
            theta = btbr_df1['peak_phase'],
            thetaunit = 'radians',
            mode = 'markers', 
            marker = dict(color = RED),
            showlegend=False
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
            name = "Mean Vector",
            r = [0, r_b6_1], 
            theta = [0, theta_b6_1],
            thetaunit = 'radians', 
            mode = 'lines',
            marker = dict(color = RED),
            showlegend=False
        ), 1, 1)
    fig.add_trace(go.Scatterpolar(
            r = [0, r_btbr_1], 
            theta = [0, theta_btbr_1], 
            thetaunit = 'radians', 
            mode = 'lines', 
            marker = dict(color = RED),
            showlegend=False
        ), 1, 2)
    fig.add_trace(go.Scatterpolar(
            name = BEHAVIOR2, 
            r = b6_df2['r'],
            theta = b6_df2['peak_phase'], 
            thetaunit = 'radians',
            mode = 'markers',
            marker = dict(color = BLUE)
        ), 1, 1)
    fig.add_trace(go.Scatterpolar(
            r = btbr_df2['r'],
            theta = btbr_df2['peak_phase'],
            thetaunit = 'radians',
            mode = 'markers', 
            marker = dict(color = BLUE),
            showlegend=False
        ), 1, 2)
    fig.add_trace(go.Scatterpolar(
            name = "Mean Vector",
            r = [0, r_b6_2], 
            theta = [0, theta_b6_2],
            thetaunit = 'radians', 
            mode = 'lines',
            marker = dict(color = BLUE),
            showlegend=False
        ), 1, 1)
    fig.add_trace(go.Scatterpolar(
            r = [0, r_btbr_2], 
            theta = [0, theta_btbr_2], 
            thetaunit = 'radians', 
            mode = 'lines', 
            marker = dict(color = BLUE),
            showlegend=False
        ), 1, 2)

    fig.write_image('raleigh_plot.png')

    # Readjust the parameters for alternate visualization
    b6_df1['r'] = 0.7
    b6_df2['r'] = 1
    btbr_df1['r'] = 0.7
    btbr_df2['r'] = 1

    # Pairwise comparison of phase, shows pairwise trends
    fig2 = make_subplots(rows=2, cols=1, specs=[[{'type': 'scatter'}]*1]*2, subplot_titles=('C5JBL/6J', 'BTBR T<+> ltpr3<tf>/J'))
    for i in range(len(b6_df1['r'])):
        fig2.add_trace(go.Scatter(
            name = BEHAVIOR1, 
            y = [b6_df1['r'].tolist()[i]],
            x = [b6_df1['peak_phase'].tolist()[i]], 
            mode = 'markers',
            marker = dict(color = RED),
            showlegend=False
        ), 1, 1)
        fig2.add_trace(go.Scatter(
            name = BEHAVIOR2, 
            y = [b6_df2['r'].tolist()[i]],
            x = [b6_df2['peak_phase'].tolist()[i]], 
            mode = 'markers',
            marker = dict(color = BLUE),
            showlegend=False
        ), 1, 1)
        fig2.add_trace(go.Scatter(
            y = [b6_df1['r'].tolist()[i], b6_df2['r'].tolist()[i]],
            x = [b6_df1['peak_phase'].tolist()[i], b6_df2['peak_phase'].tolist()[i]], 
            mode = 'lines',
            marker = dict(color = 'lightgrey'),
            showlegend=False
        ), 1, 1)
    for i in range(len(btbr_df1['r'])):
        fig2.add_trace(go.Scatter(
            name = BEHAVIOR1, 
            y = [btbr_df1['r'].tolist()[i]],
            x = [btbr_df1['peak_phase'].tolist()[i]], 
            mode = 'markers',
            marker = dict(color = RED),
            showlegend=False
        ), 2, 1)
        fig2.add_trace(go.Scatter(
            name = BEHAVIOR2, 
            y = [btbr_df2['r'].tolist()[i]],
            x = [btbr_df2['peak_phase'].tolist()[i]], 
            mode = 'markers',
            marker = dict(color = BLUE),
            showlegend=False
        ), 2, 1)
        fig2.add_trace(go.Scatter(
            y = [btbr_df1['r'].tolist()[i], btbr_df2['r'].tolist()[i]],
            x = [btbr_df1['peak_phase'].tolist()[i], btbr_df2['peak_phase'].tolist()[i]], 
            mode = 'lines',
            marker = dict(color = 'lightgrey'),
            showlegend=False
        ), 2, 1)
    fig2.update_layout(
        title = "Pairwise Phase Comparion Between Strains",
        width = 1200
        )
    fig2.update_yaxes(visible=False)
    fig2.update_xaxes(title='Peak Phase (radians)', range=[min(pd.concat([btbr_df1, btbr_df2, b6_df1, b6_df2])['peak_phase']) - 0.1, max(pd.concat([btbr_df1, btbr_df2, b6_df1, b6_df2])['peak_phase']) + 0.1])

    fig2.write_image('phase_vis.png')

    (ggplot(phase_diff_df, aes(x='phase_diff', color='strain')) + 
        geom_density() + 
        theme_bw() + 
        geom_vline(xintercept=0, color='gray', size=1) + 
        coord_cartesian(xlim=(-1,1))
    ).save('phase_hist.png')


    # Set the parameters for the raleigh plots for the phase_diff values
    phase_diff_df['r'] = 1
    phase_diff_df['x'], phase_diff_df['y'] = pol2cart(phase_diff_df['r'], phase_diff_df['phase_diff'])

    # Create a data frame for plotting each of the strains
    b6_df = phase_diff_df[phase_diff_df['strain'] == 'C57BL/6J']
    btbr_df = phase_diff_df[phase_diff_df['strain'] == 'BTBR T<+> ltpr3<tf>/J']

    # Calculate where the mean vector should point to by calculating the centroid of the plotted points
    center_b6x = sum(b6_df['x'])/len(b6_df['x'])
    center_b6y = sum(b6_df['y'])/len(b6_df['y'])
    center_btbrx = sum(btbr_df['x'])/len(btbr_df['x'])
    center_btbry = sum(btbr_df['y'])/len(btbr_df['y'])
    r_b6, theta_b6 = cart2pol(center_b6x, center_b6y)
    r_btbr, theta_btbr = cart2pol(center_btbrx, center_btbry)

    # Display the phase differences for each strain
    with open('stats.txt', 'a') as outfile: 
        outfile.write('Values for the phase difference between behaviors\n')
        outfile.write(f"Mean Radius and Phase for B6J: {r_b6, theta_b6}\n")
        outfile.write(f"Shift of {theta_b6 * (24/(2*np.pi)) * 60} minutes\n")
        outfile.write(f"Mean Radius and Phase for BTBR: {r_btbr, theta_btbr}\n")
        outfile.write(f"Shift of {theta_btbr * (24/(2*np.pi)) * 60} minutes\n\n")

    # Plot the raleigh plot
    fig3 = make_subplots(rows=1, cols=2, specs=[[{'type': 'polar'}]*2]*1, subplot_titles=('C5JBL/6J', 'BTBR T<+> ltpr3<tf>/J'))

    fig3.add_trace(go.Scatterpolar(
            name = 'C57BL/6J', 
            r = b6_df['r'],
            theta = b6_df['phase_diff'], 
            thetaunit = 'radians',
            mode = 'markers',
            marker = dict(color = BLUE),
            showlegend=False
        ), 1, 1)
    fig3.add_trace(go.Scatterpolar(
            name = "BTBR T<+> ltpr3<tf>/J",
            r = btbr_df['r'],
            theta = btbr_df['phase_diff'],
            thetaunit = 'radians',
            mode = 'markers', 
            marker = dict(color = RED),
            showlegend=False
        ), 1, 2)
    fig3.update_layout(
        title = f"Peak Phase Difference Between {BEHAVIOR1} And {BEHAVIOR2}",
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
    fig3.add_trace(go.Scatterpolar(
            name = "Mean Vector C57BL/6J",
            r = [0, r_b6], 
            theta = [0, theta_b6],
            thetaunit = 'radians', 
            mode = 'lines',
            marker = dict(color = BLUE),
            showlegend=False
        ), 1, 1)
    fig3.add_trace(go.Scatterpolar(
            name = "Mean Vector BTBR T<+> ltpr3<tf>/J",
            r = [0, r_btbr], 
            theta = [0, theta_btbr], 
            thetaunit = 'radians', 
            mode = 'lines', 
            marker = dict(color = RED),
            showlegend=False
        ), 1, 2)

    fig3.write_image('raleigh_plot_diff.png')

    # Statistics to show if there is a significant difference between the mean and our calculated mean difference
    # We are expecting a mean of 0 if the two behaviors line up exactly
    h_b6, mu_b6, ci_b6 = mtest(b6_df['phase_diff'], 0)
    h_btbr, mu_btbr, ci_btbr = mtest(btbr_df['phase_diff'], 0)
    with open('stats.txt', 'a') as outfile:
        outfile.write('Statistical mtest to determine if the mean for the difference between behaviors is different than what we expect, 0:\n')
        outfile.write(f'Mean B6J: {mu_b6}\n')
        outfile.write(f'Hypothesis B6J: {h_b6[0]}\n')
        outfile.write(f'Confidence Interval B6J: {ci_b6}\n')
        outfile.write(f'Hypothesis BTBR: {h_btbr[0]}\n')
        outfile.write(f'Mean BTBR: {mu_btbr}\n')
        outfile.write(f'Confidence Interval BTBR: {ci_btbr}\n\n')

    # Statistical test to determine if the distribution is significantly not uniformly distributed around the circle
    # Corresponds to the vector length we see in the rayleigh plot
    pval_b6, V_b6 = rayleigh(b6_df['phase_diff'])
    pval_btbr, V_btbr = rayleigh(btbr_df['phase_diff'])
    with open('stats.txt', 'a') as outfile: 
        outfile.write('Statistics to determine if the differences are likely clustering or if they are more uniformly distributed thorughout the circle:\n')
        outfile.write(f"P-val for B6 rayleigh test: {pval_b6}\n")
        outfile.write(f"P-val for BTBR rayleigh test: {pval_btbr}\n\n")

if __name__ == '__main__': 
    main(sys.argv[1:])