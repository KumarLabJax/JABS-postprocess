"""Associated lines of code that deal with the comparison of predictions (from classify.py) and GT annotation (from a JABS project)."""

import pandas as pd
import plotnine as p9
import os
import sys
import numpy as np
import argparse
import warnings

import jabs_utils.read_utils as rutils
import jabs_utils.project_utils as putils
import analysis_utils.gt_utils as gutils
from jabs_utils.bout_utils import rle, filter_data
# from compare_gt import generate_iou_scan, read_annotation_dataframe

def evaluate_ground_truth(args):
    """Main function for evaluating ground truth annotations against classifier predictions.

    Args:
        args: Namespace of arguments. See `main` for arguments and descriptions
    """
    if hasattr(args, 'prediction_folder'):
        all_annotations = read_annotation_dataframe(args.ground_truth_folder, args.prediction_folder, args.interpolation_size, args.trim_time)
    else:
        raise NotImplementedError('Classifying using a classifier not currently implemented. Please use --prediction_folder instead.')

    performance_df = generate_iou_scan(all_annotations, args.stitch_scan, args.filter_scan, args.iou_thresholds, args.filter_ground_truth)
    melted_df = pd.melt(performance_df, id_vars=["behavior", "threshold", "stitch", "filter"])

    # Get the best f1 score filtering parameters for each thresholds
    # optimal_filter_df = performance_df.groupby(['behavior', 'threshold']).apply(lambda x: x.iloc[np.nanargmax(x['f1'].values)] if np.any(~np.isnan(x['f1'].values)) else x.iloc[0]).reset_index(drop=True)

    # Prototyping plot to show the relationship between threshold and optimal parameters
    # filter_selection_plot = (
    #     p9.ggplot(optimal_filter_df)
    #     + p9.geom_point(p9.aes(x='threshold', y='f1'), color='red')
    #     + p9.geom_point(p9.aes(x='threshold', y='stitch/np.max(stitch)'), color='blue')
    #     + p9.geom_point(p9.aes(x='threshold', y='filter/np.max(filter)'), color='green')
    #     + p9.facet_wrap('~behavior')
    #     + p9.theme_bw()
    # )

    middle_threshold = np.sort(args.iou_thresholds)[int(np.floor(len(args.iou_thresholds)/2))]

    (
        p9.ggplot(performance_df[performance_df['threshold'] == middle_threshold])
        + p9.geom_tile(p9.aes(x='stitch', y='filter', fill='f1'))
        + p9.geom_text(p9.aes(x='stitch', y='filter', label='np.round(f1, 2)'), color='black', size=2)
        # Obtain the highest F1 score to highlight it
        + p9.geom_point(performance_df[performance_df['threshold'] == middle_threshold].groupby(['behavior']).apply(lambda x: x.iloc[np.nanargmax(x['f1'].values)]), p9.aes(x='stitch', y='filter'), shape='*', size=3, fill='#ffffff00')
        + p9.facet_wrap('~behavior')
        + p9.theme_bw()
        + p9.labs(title=f'Performance at {middle_threshold} IoU')
    ).save(args.scan_output, height=6, width=12, dpi=300)

    winning_filters = performance_df[performance_df['threshold'] == middle_threshold].groupby(['behavior']).apply(lambda x: x.iloc[np.nanargmax(x['f1'].values)] if np.any(~np.isnan(x['f1'].values)) else x.iloc[0]).reset_index(drop=True)[['behavior', 'stitch', 'filter']]

    melted_winning = pd.concat([melted_df[(melted_df[['behavior', 'stitch', 'filter']] == row).all(axis='columns')] for _, row in winning_filters.iterrows()])

    (
        p9.ggplot(melted_winning[melted_winning['variable'].isin(['pr', 're', 'f1'])], p9.aes(x='threshold', y='value', color='variable'))
        + p9.geom_line()
        + p9.facet_wrap('~behavior')
        + p9.theme_bw()
    ).save(args.bout_output, height=6, width=12, dpi=300)


def read_annotation_dataframe(ground_truth_folder, prediction_folder, interpolation_size, trim_time) -> pd.DataFrame:
    """Read in the predictions and ground truth annotations.

    Args:
        ground_truth_folder: folder containing a JABS project with annotations
        prediction_folder: folder with predictions from classifiers
        interpolation_size: number of frames to allow interpolation on prediction data
        trim_time: number of frames to trim the prediction data to (if GT was not annotated for the full video length)

    Returns:
        Dataframe containing annotation and prediction data
    """
    gt_annotations = rutils.read_project_annotations(ground_truth_folder)

    pred_behaviors = putils.get_behaviors_in_folder(prediction_folder)
    pred_poses = putils.get_poses_in_folder(prediction_folder)
    raw_predictions = []
    for behavior in pred_behaviors:
        for pose_file in pred_poses:
            cur_video = putils.pose_to_video(pose_file)
            prediction_df = rutils.parse_predictions(
                putils.pose_to_prediction(pose_file, behavior),
                interpolate_size=interpolation_size,
                stitch_bouts=0,
                filter_bouts=0,
                trim_time=trim_time,
            )
            prediction_df['behavior'] = behavior
            prediction_df['video'] = cur_video
            raw_predictions.append(prediction_df)
    raw_predictions = pd.concat(raw_predictions).reset_index(drop=True)

    # Combine the df to make comparing animals a lot easier
    gt_annotations["is_gt"] = True
    raw_predictions["is_gt"] = False
    all_annotations = pd.concat([gt_annotations, raw_predictions])
    # Patch for some old projects where classifier name doesn't match gt name
    # all_annotations["behavior"] = [re.sub("-", "_", x) for x in all_annotations["behavior"]]
    all_annotations["mouse_idx"] = (
        all_annotations["video"] + "_" + all_annotations["animal_idx"].astype(str)
    )
    all_annotations["mouse_idx"] = all_annotations["mouse_idx"].astype("category")

    return all_annotations

def generate_iou_scan(all_annotations, stitch_scan, filter_scan, threshold_scan, filter_ground_truth: bool = False) -> pd.DataFrame:
    """Scans stitch and filter values to produce a bout-level performance metrics at varying IoU values.

    Args:
        all_annotations: pd.DataFrame returned from `read_annotation_dataframe`
        stitch_scan: list of potential stitching values to scan
        filter_scan: list of potential filter values to scan
        threshold_scan: list of potential iou thresholds to scan
        filter_ground_truth: allow identical stitching and filters to be applied to the ground truth data?

    Returns:
        pd.DataFrame containing performance across all combinations of the scan
    """
    # Loop over the animals by behavior
    performance_df = []
    for (cur_behavior, cur_animal), animal_df in all_annotations.groupby(['behavior', 'mouse_idx']):
        # For each animal, we want a matrix of intersections, unions, and ious
        pr_df = animal_df[~animal_df['is_gt']]
        if len(pr_df) == 0:
            warnings.warn(f'No predictions for {cur_animal} for behavior {cur_behavior}... skipping.')
            continue
        pr_bout_start, pr_bout_durations, pr_bout_states = pr_df['start'].values, pr_df['duration'].values, pr_df['is_behavior'].values
        # gt data only reads in 'behavior' data, so pad not behavior to the shape of predictions
        gt_df = animal_df[animal_df['is_gt']]
        gt_vector = np.zeros(pr_bout_start[-1] + pr_bout_durations[-1], dtype=pr_bout_states.dtype)
        for _, row in gt_df.iterrows():
            gt_vector[row['start']:row['start'] + row['duration']] = row['is_behavior']
        gt_bout_start, gt_bout_durations, gt_bout_states = rle(gt_vector)
        # ugly method to scan over each combination of stitch and filter in one line
        for cur_stitch, cur_filter in zip(*map(np.ndarray.flatten, np.meshgrid(stitch_scan, filter_scan))):
            if filter_ground_truth:
                cur_starts, cur_durations, cur_values = filter_data(gt_bout_start, gt_bout_durations, gt_bout_states, max_gap_size=cur_stitch, values_to_remove=[0])
                cur_starts, cur_durations, cur_values = filter_data(cur_starts, cur_durations, cur_values, max_gap_size=cur_filter, values_to_remove=[1])
                gt_bouts = np.stack([cur_starts[cur_values == 1], cur_durations[cur_values == 1]])
            else:
                gt_bouts = np.stack([gt_bout_start[gt_bout_states == 1], gt_bout_durations[gt_bout_states == 1]])
            # Always apply filters to predictions
            cur_starts, cur_durations, cur_values = filter_data(pr_bout_start, pr_bout_durations, pr_bout_states, max_gap_size=cur_stitch, values_to_remove=[0, -1])
            cur_starts, cur_durations, cur_values = filter_data(cur_starts, cur_durations, cur_values, max_gap_size=cur_filter, values_to_remove=[1, -1])
            pr_bouts = np.stack([cur_starts[cur_values == 1], cur_durations[cur_values == 1]])
            # Add iou metrics to the list
            int_mat, u_mat, iou_mat = gutils.get_iou_mat(gt_bouts.T, pr_bouts.T)
            for cur_threshold in threshold_scan:
                new_performance = {
                    'behavior': [cur_behavior],
                    'animal': [cur_animal],
                    'stitch': [cur_stitch],
                    'filter': [cur_filter],
                    'threshold': [cur_threshold],
                }
                metrics = gutils.calc_temporal_iou_metrics(iou_mat, cur_threshold)
                for key, val in metrics.items():
                    new_performance[key] = [val]
                performance_df.append(pd.DataFrame(new_performance))

    performance_df = pd.concat(performance_df)
    # Aggregate over animals
    performance_df = performance_df.groupby(['behavior', 'stitch', 'filter', 'threshold'])[['tp', 'fn', 'fp']].apply(np.sum).reset_index()
    # Re-calculate PR/RE/F1
    performance_df['pr'] = performance_df['tp'] / (performance_df['tp'] + performance_df['fp'])
    performance_df['re'] = performance_df['tp'] / (performance_df['tp'] + performance_df['fn'])
    performance_df['f1'] = 2 * (performance_df['pr'] * performance_df['re']) / (performance_df['pr'] + performance_df['re'])

    return performance_df


def main(argv):
    """Main function that parses arguments and runs minor checks.
    
    Args:
        argv: Command-line arguments
    """
    parser = argparse.ArgumentParser(description='Evaluates classifier performance on densely annotated ground truth data')
    parser.add_argument('--ground_truth_folder', help='Path to the JABS project which contains densely annotated ground truth data.', required=True)
    g1 = parser.add_mutually_exclusive_group()
    g1.add_argument('--prediction_folder', help='Path to the folder where behavior predictions were made.')
    g1.add_argument('--exported_classifier', help='Exported JABS classifier to generate predictions.')
    parser.add_argument('--stitch_scan', help='List of stitching (time gaps in frames to merge bouts together) values to test.', type=float, nargs='+', default=np.arange(5, 46, 5).tolist())
    parser.add_argument('--filter_scan', help='List of filter (minimum duration in frames to consider real) values to test.', type=float, nargs='+', default=np.arange(5, 46, 5).tolist())
    parser.add_argument('--iou_thresholds', help='List of intersection over union thresholds to scan.', type=float, nargs='+', default=np.arange(0.05, 1.01, 0.05))
    parser.add_argument('--interpolation_size', help='Number of frames to interpolate missing data.', default=0, type=int)
    parser.add_argument('--filter_ground_truth', help='Apply filters to ground truth data (default is only to filter predictions).', default=False, action='store_true')
    parser.add_argument('--scan_output', help='Output file to save the filter scan performance plot.', default=None)
    parser.add_argument('--bout_output', help='Output file to save the resulting bout performance plot.', default=None)
    parser.add_argument('--trim_time', help='Limit the duration in frames of videos for performance (e.g. only the first 2 minutes of a 10 minute video were densely annotated).', default=None, type=int)
    args = parser.parse_args()

    assert os.path.exists(args.ground_truth_folder)
    assert os.path.exists(args.prediction_folder)
    if args.scan_output is None and args.bout_output is None:
        print('Neither scan or bout outputs were selected, nothing to do. Please use --scan_output or --bout_output.')

    evaluate_ground_truth(args)


if __name__ == "__main__":
    main(sys.argv[1:])


# Leinani tuned the filters per-behavior
# behavior_filters = {'Approach': {'filter': 5}, 'Chase': {'filter': 9}, 'Leave': {'filter': 3}, 'Nose_genital': {'filter': 9}, 'Nose_nose': {'filter': 9}}
# Play behaviors
# behavior_filters = {
#     "Chase": {"filter": 37, "stitch": 7},
#     "Jerk": {"filter": 5, "stitch": 3},
# }
# default_filter = 5
# default_stitch = 5

# broad scan range
# scan_ranges = np.arange(3, 15, 3).tolist() + np.arange(15, 50, 5).tolist(); thresholds = np.arange(0.05, 1.01, 0.05)

# original classifiers
# scan_ranges = np.arange(1, 46); thresholds = [0.5]
# args = SimpleNamespace(ground_truth_folder='/media/bgeuther/Storage/TempStorage/SocialPaper/Play/Play-groundtruth/', prediction_folder='/media/bgeuther/Storage/TempStorage/SocialPaper/Play/Play-groundtruth/', stitch_scan=scan_ranges, filter_scan=scan_ranges, iou_thresholds=thresholds, interpolation_size=0, filter_ground_truth=False, scan_output='normal-iou-test.png', bout_output=None, trim_time=None)

# Winnig filters
# args = SimpleNamespace(ground_truth_folder='/media/bgeuther/Storage/TempStorage/SocialPaper/Play/Play-groundtruth/', prediction_folder='/media/bgeuther/Storage/TempStorage/SocialPaper/Play/Play-groundtruth/', stitch_scan=[1, 6], filter_scan=[6, 35], iou_thresholds=np.arange(0.01, 1.001, 0.01), interpolation_size=0, filter_ground_truth=False, scan_output=None, bout_output='pr-re-f1_normal_2023-01-02.png', trim_time=None)

# fft scans
# scan_ranges = np.arange(1, 46); thresholds = [0.5]
# args = SimpleNamespace(ground_truth_folder='/media/bgeuther/Storage/TempStorage/SocialPaper/Play/Play-groundtruth-fft/', prediction_folder='/media/bgeuther/Storage/TempStorage/SocialPaper/Play/Play-groundtruth-fft/', stitch_scan=scan_ranges, filter_scan=scan_ranges, iou_thresholds=thresholds, interpolation_size=0, filter_ground_truth=False, scan_output='fft-iou-test.png', bout_output=None, trim_time=None)

# Winnig filters
# args = SimpleNamespace(ground_truth_folder='/media/bgeuther/Storage/TempStorage/SocialPaper/Play/Play-groundtruth-fft/', prediction_folder='/media/bgeuther/Storage/TempStorage/SocialPaper/Play/Play-groundtruth-fft/', stitch_scan=[1, 3], filter_scan=[5, 34], iou_thresholds=np.arange(0.01, 1.001, 0.01), interpolation_size=0, filter_ground_truth=False, scan_output=None, bout_output='pr-re-f1_fft_2023-01-02.png', trim_time=None)
