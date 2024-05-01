import os
import sys
import pandas as pd
import argparse
import plotnine as p9
from math import log
from statistics import median, mean

def merge_json(input_folder, output_file): #THERE IS A BOUT FILE FOR THE BTBR B6J dataset
    """Combines all the annotation files into a single python object 

    Args:
        input_files: a list of all the annotation
    Returns: 

    """
    json_files = [pos_json for pos_json in os.listdir(input_folder)]
    result = pd.DataFrame()
    for i in range(len(json_files)):
        file = input_folder + json_files[i]
        f = open(file, "r")
        data = pd.read_json(f)
        result = pd.concat([result,data])
    result.to_json(output_file, orient = 'table', indent=2)
    return result


def multi_json_calc_bout_lengths(input_files, output_file, behavior):
    """ Calculates bout lengths for multiple json files

    Args:
    """
    file_data = merge_json(input_files, output_file)
    bout_data = file_data["labels"].values
    bouts = []
    for i in range(len(bout_data)):
        for j in range(len(bout_data[i][behavior])):
            start_val = list(bout_data[i][behavior][j].values())[0]
            end_val = list(bout_data[i][behavior][j].values())[1]
            bout_length = end_val - start_val
            if bout_length > 0:
                bouts.append(bout_length)
    return bouts

def single_csv_calc_bout_lengths(input_files):
    data = pd.read_csv(input_files)
    huddling_bouts = data[data["is_behavior"]==1]
    bouts = (huddling_bouts['duration']).tolist()
    return bouts
    
    
def plot_bouts(input_files, output_file, output_graph, behavior, is_multi_file):
    if is_multi_file:
        bouts = multi_json_calc_bout_lengths(input_files, output_file, behavior)
    else:
        bouts = single_csv_calc_bout_lengths(input_files)

    (
    p9.ggplot(pd.DataFrame(bouts), p9.aes(x=bouts))
    + p9.geom_histogram(binwidth=25)
    + p9.scale_y_log10()
    + p9.labs(title="Huddling Bout Lengths of Ground Truth Set", x="Bout Length (Frames)")
    ).save(output_graph, height=6, width=12, dpi=300)
    print(median(bouts))
    print(mean(bouts))


def main(argv):
    parser = argparse.ArgumentParser(description="Creates bout length plots")
    parser.add_argument('--input', help='Path to annotation folder or file')
    parser.add_argument('--output_json', help='Name of bout json file')
    parser.add_argument('--output_graph', help='Name of bout graph file')
    parser.add_argument('--behavior', help='Name of behavior being classified')
    parser.add_argument('--is_multi_file', help='True if the bout lengths of multiple files is being determined')

    args = parser.parse_args()

    assert os.path.exists(args.input)
    #assert os.path.isfile(args.input)
    
    plot_bouts(args.input, args.output_json, args.output_graph, args.behavior, args.is_multi_file)

    #json_to_dict(args.input_file, args.output_folder, args.input_video, args.behavior)


    
if __name__ == '__main__':
    main(sys.argv[1:])