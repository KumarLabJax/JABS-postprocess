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

RED = '#e41a1c'
BLUE = '#377eb8'
GREEN = '#4daf4a'
PURPLE = '#984ea3'
PINK = '#e7298a'
ORANGE = '#d95f02'

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

def main(argv): 
    parser = argparse.ArgumentParser(description="Compares the circadian phase FFT analysis for two behaviors")
    parser.add_argument('--first_behavior', help='Name of First Behavior', required=True)
    parser.add_argument('--second_behavior', help='Name of Second Behavior', required=True)
    parser.add_argument('--first_results', help='Path to the results file for the first behavior', required=True)
    parser.add_argument('--second_results', help='Path to the results file for the second behavior', required=True)
    parser.add_argument('--jmcrs_data', help='Path to the metadata for the mouse experiments', required=True)
    parser.add_argument('--filter_experiments', help='Whether or not to filter out videos with lixit/food hopper/data problems', default=False, action='store_true')
    args = parser.parse_args()
    run_analysis(args)

def run_analysis(args):
    # This code compares the circadian phase analysis of two behaviors

    # Run this line before starting up the interactive python session for accessing libraries
    # export PYTHONPATH=/JABS-postprocess/
    # Alternatively, now we can use the singularity image at /projects/kumar-lab/JABS/JABS-Postprocessing-2023-02-07.sif

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

    if args.filter_experiments: 
        # Filter out experiments with problems
        remove_experiments = ['MDB0003','MDB0004','MDB0048','MDB0011','MDX0005','MDX0008','MDX0017']
        df1 = df1[~np.isin(df1['ExptNumber'], remove_experiments)]
        df2 = df2[~np.isin(df2['ExptNumber'], remove_experiments)]

    # Filter out experiments that aren't in both dataframes after the filtering
    df1 = df1[df1['ExptNumber'].isin(df2['ExptNumber'])]
    df2 = df2[df2['ExptNumber'].isin(df1['ExptNumber'])]
    df1['behavior'] = BEHAVIOR1
    df2['behavior'] = BEHAVIOR2

    # Delete out bins where no data exists
    no_data = np.all(df1[['time_no_pred','time_not_behavior','time_behavior']]==0, axis=1)
    df1 = df1[~no_data].reset_index()
    no_data = np.all(df2[['time_no_pred','time_not_behavior','time_behavior']]==0, axis=1)
    df2 = df2[~no_data].reset_index()

    # Give each entry a unique id based on their experiment, id, and behavior
    # This way we can keep everything in the same dataframe but still access each one individually 
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
    amplitude_phase_dict = {}

    # removing the first and last video per experiment
    df_new = df.groupby('Unique_animal').apply(lambda group: group.iloc[1:-1]).reset_index(drop=True)
    # Removing experiments MDX0017 and MDX0005 because it has missing data
    df_new = df_new[df_new['ExptNumber'] != 'MDX0017']
    df_new = df_new[df_new['ExptNumber'] != 'MDX0005']

    # Process the FFT for each animal for each behavior, save values as list entries in a data frame 
    for mouse in df_new['Unique_animal'].unique():
        df_mouse = df_new[df_new['Unique_animal'] == mouse]
        temp = pd.Series(data=df_mouse['bout_behavior'].values,index=df_mouse['relative_exp_time'].values)
        temp = temp.fillna(method='bfill')
        temp = temp.resample('H').sum()
        if len(temp) > 3 * order:  # 3*order is the default padlen in scipy's filtfilt. so making sure that there are enough data points for the filtfilt function to work properly
            freq, amplitude, phase, period = get_fft_amplitude_and_phase_scipy(temp.values, fs)
            amplitude_phase_dict[mouse] = {'freq': freq, 'amplitude': amplitude, 'phase': phase, 'period': period, 'mouse': mouse, 'strain': np.unique(df_mouse['Strain'])[0], 'behavior': np.unique(df_mouse['behavior'])[0], 'room': np.unique(df_mouse['Room'])[0]}
        else:
            print(f'Skipping mouse {mouse} due to insufficient data points.')

    phase_df = pd.DataFrame.from_dict(amplitude_phase_dict, orient='index')

    # Used to format the axis for the frequency plot
    def to_fraction(x):
        if x[0] == 0: 
            return 0
        else: 
            return '{}/{}'.format(x[0], x[1])

    # Plot the frequency-amplitude space using fractions as the axis so the frequencies are more readable 
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

    # Filter the data so as to only focus on the circadian frequencies around 1/24
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
        tmp2 = {'mouse_id': mouse, 'peak_phase': peak_phase, 'amplitude': max_amp, 'strain': phase_df['strain'][mouse], 'behavior': phase_df['behavior'][mouse], 'room': phase_df['room'][mouse]}
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

    # Used to convert between polar and cartesian when finding the centroid of the points for the mean vector
    def pol2cart(rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return x, y

    def cart2pol(x, y):
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return(rho, phi)

    # Calculate the phase difference for the two behaviors, put in new dataframe called phase_diff_df
    # This will be used later in the phase_diff raleigh plot visualization
    # Each mouse has a unique id for each behavior that is the mouse id plus 'a' or 'b' dependent on if it's behavior one or two 
    ray_df.set_index('mouse_id')
    phase_diff_df = pd.DataFrame(columns=['mouse_id', 'phase_diff', 'amplitude1', 'amplitude2', 'strain', 'phase1', 'phase2', 'room'])
    for mouse in np.unique(ray_df[ray_df['behavior'] == BEHAVIOR1]['mouse_id']):
        mouse_id = mouse[0:-1]
        # if the mouse only has an entry for one behavior instead of both 
        if mouse_id + 'b' not in np.unique(ray_df['mouse_id']):
            continue
        # Calculate and add the phases and the phase difference to a new dataframe 
        phase1 = ray_df[ray_df['mouse_id'] == mouse_id + 'a']['peak_phase'].values[0]
        phase2 = ray_df[ray_df['mouse_id'] == mouse_id + 'b']['peak_phase'].values[0]
        phase1 = phase1 % (2*np.pi)
        phase2 = phase2 % (2*np.pi)
        phase_diff = phase1 - phase2
        amplitude1 = ray_df[ray_df['mouse_id'] == mouse_id + 'a']['amplitude'].values[0]
        amplitude2 = ray_df[ray_df['mouse_id'] == mouse_id + 'b']['amplitude'].values[0]
        strain = ray_df[ray_df['mouse_id'] == mouse]['strain'].values[0]
        room = ray_df[ray_df['mouse_id'] == mouse]['room'].values[0]
        tmp = {'mouse_id': mouse_id, 'phase_diff': phase_diff, 'amplitude1': amplitude1, 'amplitude2': amplitude2, 'strain': strain, 'phase1': phase1, 'phase2': phase2, 'room': room}
        tmp = pd.DataFrame([tmp])
        phase_diff_df = pd.concat([phase_diff_df, tmp])

    # Put the phase values on a scale of 0 to 2pi for plotting 
    ray_df['peak_phase'] = ray_df['peak_phase'] % (2*np.pi)
    # Shift the phase values by pi/2 to show the peak time of eating rather than when they begin to move towards their peak time of eating
    ray_df['peak_phase'] = ray_df['peak_phase'] + (np.pi)/2
    # Radius of 1 as default for rayleigh plot
    ray_df['r'] = 1
    ray_df['x'], ray_df['y'] = pol2cart(ray_df['r'], ray_df['peak_phase'])

    # Export the data as a csv for other analysis 
    #ray_df[['mouse_id', 'peak_phase', 'strain', 'behavior']].to_csv('phase_analysis.csv')

    # Create dictionaries with rooms and strains to make it simpler to access them later for plotting
    rooms = np.unique(ray_df['room'])
    strains = {'bl6': 'C57BL/6J', 'btbr': 'BTBR T<+> ltpr3<tf>/J'}

    # Split the data by behavior
    ray_df1 = ray_df[ray_df['behavior'] == BEHAVIOR1]
    ray_df2 = ray_df[ray_df['behavior'] == BEHAVIOR2]

    # Split the data further by strain
    b6_df1 = ray_df1[ray_df1['strain'] == strains['bl6']]
    b6_df2 = ray_df2[ray_df2['strain'] == strains['bl6']]
    btbr_df1 = ray_df1[ray_df1['strain'] == strains['btbr']]
    btbr_df2 = ray_df2[ray_df2['strain'] == strains['btbr']]
    strain_dfs = [b6_df1, btbr_df1, b6_df2, btbr_df2]

    # Calculate the centroids for each of the behaviors, strains, and rooms
    # This will produce 12 centroids total, beause there are 3 rooms * 2 strains * 2 behaviors
    room_centroids = {}
    for room in rooms: 
        room_centroids[room] = {}
        for strain_df in strain_dfs: 
            # The unique key in the dictionary for the room centroid will be the strain and behavior beacuse there
            # is only one combo of strain and behavior per room 
            strain = np.unique(strain_df['strain'])[0]
            behavior = np.unique(strain_df['behavior'])[0]
            key = strain + behavior
            df = strain_df[strain_df['room'] == room]
            x = sum(df['x'])/len(df['y'])
            y = sum(df['y'])/len(df['y'])
            r, theta = cart2pol(x, y)
            room_centroids[room][key] = (r, theta)
    

    # Statistics to determine if there is a significant difference between behavior distributions for each strain
    pval_b6, _ = kuiper(b6_df1['peak_phase'], b6_df2['peak_phase'])
    pval_btbr, _ = kuiper(btbr_df1['peak_phase'], btbr_df2['peak_phase'])

    # Statistics to determine if there is a significant difference between strains for each behavior
    pval_1, _ = kuiper(b6_df1['peak_phase'], btbr_df1['peak_phase'])
    pval_2, _ = kuiper(b6_df2['peak_phase'], btbr_df2['peak_phase'])
    with open('stats.txt', 'w') as outfile:
        outfile.write('Distribution comparisons using the kuiper test: \n')
        outfile.write(f'P-val that b6 {BEHAVIOR1} and {BEHAVIOR2} are the same distribution: {pval_b6[0]}\n')
        outfile.write(f'P-val that BTBR {BEHAVIOR1} and {BEHAVIOR2} are the same distribution: {pval_btbr[0]}\n')
        outfile.write(f'P-val that the {BEHAVIOR1} phases for btbr and b6js are the same distribution: {pval_1[0]}\n')
        outfile.write(f'P-val that the {BEHAVIOR2} phases for btbr and b6js are the same distribution: {pval_2[0]}\n\n')

    b6pval, _ = watson_williams(b6_df1['peak_phase'], b6_df2['peak_phase'])
    btbrpval, _ = watson_williams(btbr_df1['peak_phase'], btbr_df2['peak_phase'])
    pval1, _ = watson_williams(b6_df1['peak_phase'], btbr_df1['peak_phase'])
    pval2, _ = watson_williams(b6_df2['peak_phase'], btbr_df2['peak_phase'])
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


    # Set the radii for rayleigh plot visualization purposes
    b6_df1['r'] = 1
    b6_df2['r'] = 1
    btbr_df1['r'] = 1
    btbr_df2['r'] = 1

    colors = {rooms[0]: BLUE, rooms[1]: RED, rooms[2]: GREEN}
   
    #====== RAYLEIGH COMPARISON PLOT ======# 

    # Each subplot represents a behavior and strain 
    fig = make_subplots(rows=2, cols=2, specs=[[{'type': 'polar'}]*2]*2, subplot_titles=(strains['bl6'] + ' ' + BEHAVIOR1, strains['btbr'] + ' ' + BEHAVIOR1, strains['bl6'] + ' ' + BEHAVIOR2, strains['btbr'] + ' ' + BEHAVIOR2))

    # The common layout to assign to each subplot
    layout = dict(
            radialaxis = dict(tickvals = []),
            angularaxis = dict(
                thetaunit = 'radians',
                dtick = 45,
                rotation = 90,
                direction = 'clockwise',
                tickmode = 'array', 
                tickvals = [0, 90, 180, 270],
                ticktext = ['0', '6', '12', '18'])
        )

    fig.update_layout(
        title = "Peak Phase Comparion Between Strains",
        polar1 = layout,
        polar2 = layout,
        polar3 = layout,
        polar4 = layout, 
        width = 800,
        height = 600)
    fig.update_annotations(yshift=20)

    # For each of the four subplots, plot each of the mean vectors corresponding to the different rooms for that strain and behavior
    row = 1
    for behavior in [BEHAVIOR1, BEHAVIOR2]: 
        col = 1
        for strain in strains.values(): 
            for room in rooms:
                fig.add_trace(go.Scatterpolar(
                    name = f"Mean Vector {behavior}",
                    # Get r and theta from the room_centroids dictionary defined earlier, with the first key being the room, 
                    # and the second key being the unique combination of the strain and behavior
                    r = [0, room_centroids[room][strain + behavior][0]], 
                    theta = [0, room_centroids[room][strain + behavior][1]],
                    thetaunit = 'radians', 
                    mode = 'lines',
                    marker = dict(color = colors[room]), 
                    showlegend=False
                ), row, col)
            col += 1
        row += 1    

    # For each of the four subplots, plot each of the points as scatter points around the circle 
    # The row and column and used_rooms set are used to make sure that we are on the right subplot when we add the points
    row = 1
    used_rooms = set()
    for i, strain_df in zip(range(len(strain_dfs)), strain_dfs): 
        col = 1
        for room in rooms: 
            if room in used_rooms: 
                col = 2
            else: 
                used_rooms.add(room)
            fig.add_trace(go.Scatterpolar(
                name = np.unique(strain_df['behavior'])[0] + ' ' + room, 
                # We get our values from the dataframems containing all the values narrowed down by behavior and strain
                r = strain_df[strain_df['room'] == room]['r'],
                theta = strain_df[strain_df['room'] == room]['peak_phase'], 
                thetaunit = 'radians',
                mode = 'markers',
                marker = dict(color = colors[room]),
                showlegend=False
            ), row, col)
        if i == 1: 
            row += 1
            used_rooms.clear()

    fig.write_image('new_raleigh_plot_room_comp.png')


    # Readjust the parameters for alternate visualization
    b6_df1['r'] = 0.7
    b6_df2['r'] = 1
    btbr_df1['r'] = 0.7
    btbr_df2['r'] = 1

    # Pairwise comparison of phase, shows pairwise trends
    # This code could most likely be optimized and writted with less code, but I didn't have time to get to that
    fig2 = make_subplots(rows=2, cols=1, specs=[[{'type': 'scatter'}]*1]*2, subplot_titles=('C5JBL/6J', 'BTBR T<+> ltpr3<tf>/J'))

    for i in range(len(b6_df1['r'])):
        # Add each of the lines conneting each unique animals phase for each behavior
        fig2.add_trace(go.Scatter(
            y = [b6_df1['r'].tolist()[i], b6_df2['r'].tolist()[i]],
            x = [b6_df1['peak_phase'].tolist()[i], b6_df2['peak_phase'].tolist()[i]], 
            mode = 'lines',
            marker = dict(color = 'lightgrey'),
            showlegend=False
        ), 1, 1)

        # Depending on the room, plot the two points for each behavior for each animal 
        if b6_df1['room'].tolist()[i] == rooms[0]: 
            fig2.add_trace(go.Scatter(
                name = rooms[0], 
                y = [b6_df1['r'].tolist()[i]],
                x = [b6_df1['peak_phase'].tolist()[i]], 
                mode = 'markers',
                marker = dict(color = colors[rooms[0]]),
                showlegend=False
            ), 1, 1)
            fig2.add_trace(go.Scatter(
                y = [b6_df2['r'].tolist()[i]],
                x = [b6_df2['peak_phase'].tolist()[i]], 
                mode = 'markers',
                marker = dict(color = colors[rooms[0]]),
                showlegend=False
            ), 1, 1)
        elif b6_df1['room'].tolist()[i] == rooms[1]:
            fig2.add_trace(go.Scatter(
                name = rooms[1], 
                y = [b6_df1['r'].tolist()[i]],
                x = [b6_df1['peak_phase'].tolist()[i]], 
                mode = 'markers',
                marker = dict(color = colors[rooms[1]]),
                showlegend=False
            ), 1, 1)
            fig2.add_trace(go.Scatter(
                y = [b6_df2['r'].tolist()[i]],
                x = [b6_df2['peak_phase'].tolist()[i]], 
                mode = 'markers',
                marker = dict(color = colors[rooms[1]]),
                showlegend=False
            ), 1, 1)
        else: 
            fig2.add_trace(go.Scatter(
                name = rooms[2], 
                y = [b6_df1['r'].tolist()[i]],
                x = [b6_df1['peak_phase'].tolist()[i]], 
                mode = 'markers',
                marker = dict(color = colors[rooms[2]]),
                showlegend=False
            ), 1, 1)
            fig2.add_trace(go.Scatter(
                y = [b6_df2['r'].tolist()[i]],
                x = [b6_df2['peak_phase'].tolist()[i]], 
                mode = 'markers',
                marker = dict(color = colors[rooms[2]]),
                showlegend=False
            ), 1, 1)

    # Do teh same iteration routine for the second strain 
    for i in range(len(btbr_df1['r'])):
        fig2.add_trace(go.Scatter(
            y = [btbr_df1['r'].tolist()[i], btbr_df2['r'].tolist()[i]],
            x = [btbr_df1['peak_phase'].tolist()[i], btbr_df2['peak_phase'].tolist()[i]], 
            mode = 'lines',
            marker = dict(color = 'lightgrey'),
            showlegend=False
        ), 2, 1)

        if btbr_df1['room'].tolist()[i] == rooms[0]: 
            fig2.add_trace(go.Scatter(
                y = [btbr_df1['r'].tolist()[i]],
                x = [btbr_df1['peak_phase'].tolist()[i]], 
                mode = 'markers',
                marker = dict(color = colors[rooms[0]]),
                showlegend=False
            ), 2, 1)
            fig2.add_trace(go.Scatter(
                y = [btbr_df2['r'].tolist()[i]],
                x = [btbr_df2['peak_phase'].tolist()[i]], 
                mode = 'markers',
                marker = dict(color = colors[rooms[0]]),
                showlegend=False
            ), 2, 1)
        elif btbr_df1['room'].tolist()[i] == 'B6': 
            fig2.add_trace(go.Scatter(
                y = [btbr_df1['r'].tolist()[i]],
                x = [btbr_df1['peak_phase'].tolist()[i]], 
                mode = 'markers',
                marker = dict(color = colors[rooms[1]]),
                showlegend=False
            ), 2, 1)
            fig2.add_trace(go.Scatter(
                y = [btbr_df2['r'].tolist()[i]],
                x = [btbr_df2['peak_phase'].tolist()[i]], 
                mode = 'markers',
                marker = dict(color = colors[rooms[1]]),
                showlegend=False
            ), 2, 1)
        else: 
            fig2.add_trace(go.Scatter(
                y = [btbr_df1['r'].tolist()[i]],
                x = [btbr_df1['peak_phase'].tolist()[i]], 
                mode = 'markers',
                marker = dict(color = colors[rooms[2]]),
                showlegend=False
            ), 2, 1)
            fig2.add_trace(go.Scatter(
                y = [btbr_df2['r'].tolist()[i]],
                x = [btbr_df2['peak_phase'].tolist()[i]], 
                mode = 'markers',
                marker = dict(color = colors[rooms[2]]),
                showlegend=False
            ), 2, 1)
    fig2.update_layout(
        title = "Pairwise Phase Comparion Between Strains",
        width = 1200
        )
    fig2.update_yaxes(visible=False)
    fig2.update_xaxes(title='Peak Phase (radians)')

    fig2.write_image('phase_vis_room_comp.png')

    # Plot the two strains phase difference distributions to see their comparative variations
    (ggplot(phase_diff_df, aes(x='phase_diff', color='strain')) + 
        geom_density() + 
        theme_bw() + 
        geom_vline(xintercept=0, color='gray', size=1) + 
        coord_cartesian(xlim=(-1,1))
    ).save('phase_density.png')


    # Set the parameters for the raleigh plots for the phase_diff values
    phase_diff_df['r'] = 1
    phase_diff_df['x'], phase_diff_df['y'] = pol2cart(phase_diff_df['r'], phase_diff_df['phase_diff'])

    # Create a data frame for plotting each of the strains
    b6_df = phase_diff_df[phase_diff_df['strain'] == strains['bl6']]
    btbr_df = phase_diff_df[phase_diff_df['strain'] == strains['btbr']]

    # Calculate where the mean vector should point to by calculating the centroid of the plotted points, this is for the entire data set per strain 
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

    # ===== PHASE DIFFERENCES RAYLEIGH PLOT ===== # 
    fig3 = make_subplots(rows=1, cols=2, specs=[[{'type': 'polar'}]*2]*1, subplot_titles=('C5JBL/6J', 'BTBR T<+> ltpr3<tf>/J'))

    for room in rooms: 
        # Plot the points for each room in a different color 
        fig3.add_trace(go.Scatterpolar(
            name = room, 
            r = b6_df[b6_df['room'] == room]['r'],
            theta = b6_df[b6_df['room'] == room]['phase_diff'], 
            thetaunit = 'radians',
            mode = 'markers',
            marker = dict(color = colors[room])
        ), 1, 1)

        # Calculate the centroid for just the points corresponding to the specific room
        r, theta = cart2pol(sum(b6_df[b6_df['room'] == room]['x'])/len(b6_df[b6_df['room'] == room]['x']), sum(b6_df[b6_df['room'] == room]['y'])/len(b6_df[b6_df['room'] == room]['y']))

        # Plot the mean vector using the centroid
        fig3.add_trace(go.Scatterpolar(
            name = "Mean Vector",
            r = [0, r], 
            theta = [0, theta],
            thetaunit = 'radians', 
            mode = 'lines',
            marker = dict(color = colors[room]),
            showlegend=False
        ), 1, 1)
    
    # Do the same thing for the second strain 
    for room in rooms: 
        fig3.add_trace(go.Scatterpolar(
            name = room, 
            r = btbr_df[btbr_df['room'] == room]['r'],
            theta = btbr_df[btbr_df['room'] == room]['phase_diff'], 
            thetaunit = 'radians',
            mode = 'markers',
            marker = dict(color = colors[room]),
            showlegend=False
        ), 1, 2)

        r, theta = cart2pol(sum(btbr_df[btbr_df['room'] == room]['x'])/len(btbr_df[btbr_df['room'] == room]['x']), sum(btbr_df[btbr_df['room'] == room]['y'])/len(btbr_df[btbr_df['room'] == room]['y']))

        fig3.add_trace(go.Scatterpolar(
            name = "Mean Vector",
            r = [0, r], 
            theta = [0, theta],
            thetaunit = 'radians', 
            mode = 'lines',
            marker = dict(color = colors[room]),
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

    fig3.write_image('new_raleigh_plot_diff_room_comp.png')

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
