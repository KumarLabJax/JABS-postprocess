'''
Plot overlapping behavior bouts between mice within arenas. 
Example of how to run at the bottom of the script
'''
import argparse
import pandas as pd
import plotnine as p9
import numpy as np
import os
import sys
from types import SimpleNamespace

from analysis_utils.parse_table import read_ltm_summary_table, filter_experiment_time
from analysis_utils.plots import generate_time_vs_feature_plot
from itertools import chain


def plot_social_overlap_behaviors(args):
    '''
    Main function for plotting social overlapping behaviors.

    Args:
        args: Namespace of arguments. See `main` for arguments and descriptions.
    '''
    # One needs to add 'Index' to the beginning of the overlap csv file
    overlap_df = pd.read_csv(args.input_file, index_col='Index')

    remove_experiments = ['MDB0003','MDX0008','MDX0017','MDX0093']
    overlap_df = overlap_df[~np.isin(overlap_df['ExptNumber'], remove_experiments)]

    plot = (p9.ggplot(overlap_df, p9.aes(x='MiceIDs', fill='Strain')) + 
        p9.geom_histogram(color='black', binwidth = 1) +
        p9.theme_bw() + 
        p9.labs(title="Overlap Comparison") + 
        p9.facet_wrap('ExptNumber') + 
        p9.scale_color_brewer(type='qual', palette='Set1') +
        p9.theme(title = p9.element_text(hjust=0.5)))

    plot.save(str(args.output_folder) + 'two_overlap_' + str(args.behavior) + '.svg', width=13, height=8)


    time_plot = (p9.ggplot(overlap_df, p9.aes(x='Strain', y='overlap_len', fill='Strain')) + 
        p9.geom_boxplot() + 
        p9.theme_bw() + 
        p9.scale_color_brewer(type='qual', palette='Set1') +
        p9.labs(title='Overall Overlap Time Comparison', x='Strain', y='Number of Overlapping Frames Per Instance') + 
        p9.theme(title = p9.element_text(hjust=0.5)) + 
        p9.coord_cartesian(ylim=(0,500)))
        
    time_plot.save(str(args.output_folder) + 'frame_per_inst_' + str(args.behavior) + '.svg')

    num_comp_df = pd.DataFrame(columns=['ExptNumber', 'num_bouts', 'Strain','Sex'])
    len_df = pd.DataFrame(columns=['ExptNumber', 'total_len', 'MiceIDs', 'Strain','Sex'])
    time_df = pd.DataFrame(columns=['ExptNumber', 'total_time', 'Strain','Sex'])
    for cur_exp, exp_df in overlap_df.groupby('ExptNumber'): 
        num_bouts = len(exp_df['MiceIDs'])
        strain = np.unique(exp_df['Strain'])[0]
        sex = np.unique(exp_df['Sex'])[0]
        tmp = {'ExptNumber': cur_exp, 'num_bouts': num_bouts, 'Strain': strain, 'Sex': sex}
        tmp_df = pd.DataFrame([tmp])
        num_comp_df = pd.concat([num_comp_df, tmp_df], ignore_index=True)
        total_time = np.sum(exp_df['overlap_len'])/30
        tmp2 = {'ExptNumber': cur_exp, 'total_time': total_time, 'Strain': strain, 'Sex': sex}
        time_df = pd.concat([time_df, pd.DataFrame([tmp2])])
        for cur_mice, mice_df in exp_df.groupby('MiceIDs'): 
            #print(mice_df)
            total_len = sum(mice_df['overlap_len'])/30
            len_df = pd.concat([len_df, pd.DataFrame([{'ExptNumber': cur_exp, 'total_len': total_len, 'MiceIDs': cur_mice, 'Strain': strain, 'Sex': sex}])])

    num_comp_df['num_bouts'] = num_comp_df['num_bouts'].astype(int)
    box_plot = (p9.ggplot(num_comp_df, p9.aes(x='Strain', y='num_bouts', fill='Strain')) + 
        p9.geom_boxplot() + 
        p9.labs(title="Overall Overlap Amount Comparison", x="Strain", y="Total Instances Overlapping") +
        p9.scale_color_brewer(type='qual', palette='Set1') +
        p9.theme(title = p9.element_text(hjust=0.5)) + 
        p9.theme(axis_text=p9.element_text(size=18), axis_title=p9.element_text(size=17), axis_text_x=p9.element_text(rotation=0)) +
        p9.theme_bw())
    box_plot.save(str(args.output_folder) + 'num_overlap_comp_' + str(args.behavior) + '.svg')
    print(f"Mean number of overlapping instances: \nB6J ==> {np.mean(num_comp_df[num_comp_df['Strain'] == 'C57BL/6J']['num_bouts'])}\nBTBR ==> {np.mean(num_comp_df[num_comp_df['Strain'] != 'C57BL/6J']['num_bouts'])}")
    print(f"Median number of overlapping instances: \nB6J ==> {np.median(num_comp_df[num_comp_df['Strain'] == 'C57BL/6J']['num_bouts'])}\nBTBR ==> {np.median(num_comp_df[num_comp_df['Strain'] != 'C57BL/6J']['num_bouts'])}")

    box_plot_time = (p9.ggplot(time_df, p9.aes(x='Strain', y='total_time', fill='Strain')) + 
        p9.geom_boxplot() + 
        p9.theme_bw() + 
        p9.scale_color_brewer(type='qual', palette='Set1') +
        p9.labs(y='Total Overlap Time (seconds)', x='Strain') + 
        p9.theme(axis_text=p9.element_text(size=18), axis_title=p9.element_text(size=17), axis_text_x=p9.element_text(rotation=0)))
    box_plot_time.save(str(args.output_folder) + 'total_time_box_' + str(args.behavior) + '.svg')
    print(f"Mean time overlapping: \nB6J ==> {np.mean(time_df[time_df['Strain'] == 'C57BL/6J']['total_time'])}\nBTBR ==> {np.mean(time_df[time_df['Strain'] != 'C57BL/6J']['total_time'])}")
    print(f"Median time overlapping: \nB6J ==> {np.median(time_df[time_df['Strain'] == 'C57BL/6J']['total_time'])}\nBTBR ==> {np.median(time_df[time_df['Strain'] != 'C57BL/6J']['total_time'])}")

    overlap_df['overlap_len'] = overlap_df['overlap_len']/30
    box_plot_len = (p9.ggplot(overlap_df, p9.aes(x='Strain', y='overlap_len', fill='Strain')) + 
        p9.geom_boxplot() + 
        p9.theme_bw() + 
        p9.scale_color_brewer(type='qual', palette='Set1') +
        p9.labs(y='Average Overlap Bout Length (seconds)', x='Strain') + 
        p9.theme(axis_text=p9.element_text(size=18), axis_title=p9.element_text(size=17), axis_text_x=p9.element_text(rotation=0)) + 
        p9.coord_cartesian(ylim=(0,25)))
    box_plot_len.save(str(args.output_folder) + 'total_len_box_' + str(args.behavior) + '.svg')
    print(f"Mean overlap bout length: \nB6J ==> {np.mean(overlap_df[overlap_df['Strain'] == 'C57BL/6J']['overlap_len'])}\nBTBR ==> {np.mean(overlap_df[overlap_df['Strain'] != 'C57BL/6J']['overlap_len'])}")
    print(f"Median overlap bout length: \nB6J ==> {np.median(overlap_df[overlap_df['Strain'] == 'C57BL/6J']['overlap_len'])}\nBTBR ==> {np.median(overlap_df[overlap_df['Strain'] != 'C57BL/6J']['overlap_len'])}")

    num_overlap_dens = (p9.ggplot(num_comp_df, p9.aes(x='num_bouts', color='Strain')) + 
        p9.geom_density() + 
        p9.theme_bw() + 
        p9.scale_color_brewer(type='qual', palette='Set1') +
        p9.labs(title='Number of Instances Density Plot', x='Number of Instances'))
    num_overlap_dens.save(str(args.output_folder) + 'num_overlap_dens_' + str(args.behavior) + '.svg')

    avg_overlap_len_plot = (p9.ggplot(overlap_df, p9.aes('overlap_len', color='Strain')) + 
            p9.geom_density() + 
            p9.theme_bw() + 
            p9.scale_color_brewer(type='qual', palette='Set1') +
            p9.labs(title='Overlap Length by Strain', x='Overlap Length') + 
            p9.theme(title = p9.element_text(hjust=0.5)) + 
            p9.coord_cartesian(xlim=(0,1000)))
    avg_overlap_len_plot.save(str(args.output_folder) + 'overlap_dens_' + str(args.behavior) + '.svg')

    len_df['total_len'] = len_df['total_len'].astype(float)
    dur_perc = (p9.ggplot(len_df, p9.aes(x='MiceIDs', y='total_len', fill='Strain')) + 
            p9.geom_bar(stat='identity', color='black') + 
            p9.facet_wrap('ExptNumber') + 
            p9.theme_bw() + 
            p9.scale_color_brewer(type='qual', palette='Set1') +
            p9.theme(axis_text_y=p9.element_text(size=18), axis_title=p9.element_text(size=18), axis_text_x=p9.element_text(size=10)) + 
            p9.labs(x='Mice IDs', y='Overlap Duration (seconds)'))
    dur_perc.save(str(args.output_folder) + 'duration_' + str(args.behavior) + '.svg', width=16, height=8)

    strain_comp = (p9.ggplot(len_df, p9.aes(x='MiceIDs', y='total_len', fill='Strain')) + 
            p9.geom_bar(stat='identity') + 
            p9.scale_color_brewer(type='qual', palette='Set1') +
            p9.facet_wrap('Strain'))
    strain_comp.save(str(args.output_folder) + 'strain_comp_' + str(args.behavior) + '.svg')


def main(argv):
    '''Main function that parses arguments and runs minor checks
    
    Args:
        argv: Command-line arguments
    '''
    parser = argparse.ArgumentParser(description='Produces plots of overlapping behavior bouts')
    parser.add_argument('--input_file', help="Path to the input overlap.csv file. Generate using find_overlap.py script", type=str)
    parser.add_argument('--output_folder', help="Path to folder for plots to populate.", type=str)
    parser.add_argument('--behavior', help="Name of behavior being evaluated", type=str, default=None)
    args = parser.parse_args()

    assert os.path.exists(args.output_folder)
    assert os.path.exists(args.input_file)

    plot_social_overlap_behaviors(args)


if __name__ == "__main__":
    main(sys.argv[1:])

# args = SimpleNamespace(input_file='/projects/kumar-lab/choij/b6-btbr-plots/feeding_overlap.csv', output_folder='/projects/kumar-lab/choij/b6-btbr-plots/plots', behavior='Drinking')