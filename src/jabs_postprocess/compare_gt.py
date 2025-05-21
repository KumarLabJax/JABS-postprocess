"""Associated lines of code that deal with the comparison of predictions (from classify.py) and GT annotation (from a JABS project)."""

import pandas as pd
import plotnine as p9
import os
import sys
import numpy as np
import argparse
import warnings
from typing import List, Optional, Union

from jabs_postprocess.utils.project_utils import BoutTable, JabsProject, ClassifierSettings, Bouts


def evaluate_ground_truth(
    behavior: str,
    ground_truth_folder: str,
    prediction_folder: str,
    stitch_scan: List[float] = None,
    filter_scan: List[float] = None,
    iou_thresholds: List[float] = None,
    interpolation_size: int = 0,
    filter_ground_truth: bool = False,
    scan_output: Optional[str] = None,
    bout_output: Optional[str] = None,
    trim_time: Optional[int] = None,
    ethogram_output: Optional[str] = None,
):
    """Main function for evaluating ground truth annotations against classifier predictions.

    Args:
        behavior: Behavior to evaluate predictions
        ground_truth_folder: Path to the JABS project which contains densely annotated ground truth data
        prediction_folder: Path to the folder where behavior predictions were made
        stitch_scan: List of stitching (time gaps in frames to merge bouts together) values to test
        filter_scan: List of filter (minimum duration in frames to consider real) values to test
        iou_thresholds: List of intersection over union thresholds to scan
        interpolation_size: Number of frames to interpolate missing data
        filter_ground_truth: Apply filters to ground truth data (default is only to filter predictions)
        scan_output: Output file to save the filter scan performance plot
        bout_output: Output file to save the resulting bout performance plot
        trim_time: Limit the duration in frames of videos for performance
        ethogram_output: Output file to save the ethogram plot comparing GT and predictions
    """
    # Set default values if not provided
    stitch_scan = stitch_scan or np.arange(5, 46, 5).tolist()
    filter_scan = filter_scan or np.arange(5, 46, 5).tolist()
    iou_thresholds = iou_thresholds or np.arange(0.05, 1.01, 0.05).tolist()
    
    gt_df = BoutTable.from_jabs_annotation_folder(ground_truth_folder, behavior)._data
    # Settings to read in the unfiltered data
    dummy_settings = ClassifierSettings(behavior, 0, 0, 0)
    pred_df = JabsProject.from_prediction_folder(prediction_folder, dummy_settings).get_bouts()._data

    # Add dummy GT entries for videos present in predictions but missing from GT annotations
    # This ensures videos with no GT annotations (meaning "no behavior occurred") are properly represented
    
    # 1. Get all unique videos and animals from predictions
    pred_videos = pred_df[['video_name', 'animal_idx']].drop_duplicates()
    
    # 2. Get all unique videos and animals from GT
    gt_videos = gt_df[['video_name', 'animal_idx']].drop_duplicates() if not gt_df.empty else pd.DataFrame(columns=['video_name', 'animal_idx'])
    
    # 3. Find videos/animals in predictions but not in GT
    missing_gt = pred_videos.merge(gt_videos, on=['video_name', 'animal_idx'], how='left', indicator=True)
    missing_gt = missing_gt[missing_gt['_merge'] == 'left_only'].drop('_merge', axis=1)
    
    # 4. For each missing video/animal, create a dummy GT entry representing "all not-behavior"
    dummy_gt_rows = []
    for _, row in missing_gt.iterrows():
        # Find the corresponding prediction data for this video/animal to get duration
        pred_subset = pred_df[(pred_df['video_name'] == row['video_name']) & 
                             (pred_df['animal_idx'] == row['animal_idx'])]
        
        if not pred_subset.empty:
            # Calculate the full video duration from the predictions
            # This assumes predictions cover the entire video
            video_duration = int(pred_subset['start'].max() + pred_subset['duration'].max())
            
            # Create a dummy GT entry: one bout covering the entire video, marked as not-behavior (0)
            dummy_gt_rows.append({
                'video_name': row['video_name'],
                'animal_idx': row['animal_idx'],
                'start': 0,
                'duration': video_duration,
                'is_behavior': 0,  # 0 = not-behavior
            })
            warnings.warn(f"No GT annotations found for {row['video_name']} (animal {row['animal_idx']}). "
                         f"Creating dummy GT entry as 'not-behavior' for the entire video ({video_duration} frames).")
    
    # 5. Add the dummy GT entries to the original GT dataframe
    if dummy_gt_rows:
        dummy_gt_df = pd.DataFrame(dummy_gt_rows)
        gt_df = pd.concat([gt_df, dummy_gt_df], ignore_index=True)

    gt_df['is_gt'] = True
    pred_df['is_gt'] = False
    all_annotations = pd.concat([gt_df, pred_df])
    
    # We only want the positive examples for performance evaluation
    # (but for ethogram plotting later, we'll use the full all_annotations)
    performance_annotations = all_annotations[all_annotations['is_behavior'] == 1].copy()
    if not performance_annotations.empty:
        performance_annotations['behavior'] = behavior
    
    # TODO: Trim time?
    if trim_time is not None:
        warnings.warn('Time trimming is not currently supported, ignoring.')

    performance_df = generate_iou_scan(performance_annotations, stitch_scan, filter_scan, iou_thresholds, filter_ground_truth)
    melted_df = pd.melt(performance_df, id_vars=["threshold", "stitch", "filter"])

    # Get the best f1 score filtering parameters for each thresholds
    # optimal_filter_df = performance_df.groupby(['threshold']).apply(lambda x: x.iloc[np.nanargmax(x['f1'].values)] if np.any(~np.isnan(x['f1'].values)) else x.iloc[0]).reset_index(drop=True)

    # Prototyping plot to show the relationship between threshold and optimal parameters
    # filter_selection_plot = (
    #     p9.ggplot(optimal_filter_df)
    #     + p9.geom_point(p9.aes(x='threshold', y='f1'), color='red')
    #     + p9.geom_point(p9.aes(x='threshold', y='stitch/np.max(stitch)'), color='blue')
    #     + p9.geom_point(p9.aes(x='threshold', y='filter/np.max(filter)'), color='green')
    #     + p9.theme_bw()
    # )

<<<<<<< HEAD:src/jabs_postprocess/compare_gt.py
    middle_threshold = np.sort(iou_thresholds)[int(np.floor(len(iou_thresholds) / 2))]
=======
    middle_threshold = np.sort(args.iou_thresholds)[int(np.floor(len(args.iou_thresholds) / 2))]
    
>>>>>>> origin/master:compare_gt.py
    # Create a copy to avoid SettingWithCopyWarning
    subset_df = performance_df[performance_df['threshold'] == middle_threshold].copy()
    
    # Handle empty DataFrame case
    if subset_df.empty:
        warnings.warn(f"No performance data available for threshold {middle_threshold}.")
        # Create an empty plot
        plot = p9.ggplot() + p9.theme_bw() + p9.labs(title=f'No performance data for {middle_threshold} IoU')
        if scan_output:
            plot.save(scan_output, height=6, width=12, dpi=300)
        # Create default winning filters with first values from scan parameters
        winning_filters = pd.DataFrame({
            'stitch': [stitch_scan[0] if stitch_scan else 0],
            'filter': [filter_scan[0] if filter_scan else 0]
        })
    else:
        # Convert numeric columns to float to ensure continuous scale
        subset_df['stitch'] = subset_df['stitch'].astype(float)
        subset_df['filter'] = subset_df['filter'].astype(float)
        
        # Handle NaN values in f1 by replacing with 0 for plotting purposes
        subset_df['f1_plot'] = subset_df['f1'].fillna(0)
        
        # Convert f1 values to strings for labels
        subset_df['f1_label'] = subset_df['f1'].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "NA")
        
        # Create the plot with explicit scale types and proper handling of NaN values
        plot = (
            p9.ggplot(subset_df)
            + p9.geom_tile(p9.aes(x='stitch', y='filter', fill='f1_plot'))
            + p9.geom_text(p9.aes(x='stitch', y='filter', label='f1_label'), color='black', size=2)
            + p9.theme_bw()
            + p9.labs(title=f'Performance at {middle_threshold} IoU')
        )

        # Add the star point only if we have valid f1 scores
        if not subset_df['f1_plot'].isna().all() and len(subset_df['f1_plot']) > 0:
            best_idx = np.argmax(subset_df['f1_plot'])
            best_point = pd.DataFrame(subset_df.iloc[best_idx:best_idx+1])
            plot = plot + p9.geom_point(best_point, p9.aes(x='stitch', y='filter'), shape='*', size=3, color='white')

        # Add scales with explicit breaks
        plot = (
            plot
            + p9.scale_x_continuous(
                breaks=sorted(subset_df['stitch'].unique()),
                labels=[str(int(x)) for x in sorted(subset_df['stitch'].unique())]
            )
            + p9.scale_y_continuous(
                breaks=sorted(subset_df['filter'].unique()),
                labels=[str(int(x)) for x in sorted(subset_df['filter'].unique())]
            )
            + p9.scale_fill_continuous(na_value=0)
        )

        if scan_output:
            plot.save(scan_output, height=6, width=12, dpi=300)

        # Handle case where all f1_plot values are NaN or empty
        if subset_df['f1_plot'].isna().all() or len(subset_df) == 0:
            # Default to first row if available, otherwise use first values from scan parameters
            if len(subset_df) > 0:
                winning_filters = pd.DataFrame(subset_df.iloc[0:1])[['stitch', 'filter']]
            else:
                winning_filters = pd.DataFrame({
                    'stitch': [stitch_scan[0] if stitch_scan else 0],
                    'filter': [filter_scan[0] if filter_scan else 0]
                })
        else:
            winning_filters = pd.DataFrame(subset_df.iloc[np.argmax(subset_df['f1_plot'])]).T.reset_index(drop=True)[['stitch', 'filter']]

    if 'melted_winning' not in locals():
        melted_winning = pd.concat([melted_df[(melted_df[['stitch', 'filter']] == row).all(axis='columns')] for _, row in winning_filters.iterrows()])

    if bout_output:
        (
            p9.ggplot(melted_winning[melted_winning['variable'].isin(['pr', 're', 'f1'])], p9.aes(x='threshold', y='value', color='variable'))
            + p9.geom_line()
            + p9.theme_bw()
        ).save(bout_output, height=6, width=12, dpi=300)

    if ethogram_output is not None:
        # Prepare data for ethogram plot
        # Use all_annotations to include both behavior (1) and not-behavior (0) states
        plot_df = all_annotations.copy()
        # Add behavior column for all rows
        plot_df['behavior'] = behavior
        
        if not plot_df.empty:
            plot_df['end'] = plot_df['start'] + plot_df['duration']
            
            # Create a column to indicate source (GT or Pred) for the legend
            plot_df['source'] = np.where(plot_df['is_gt'], 'Ground Truth', 'Prediction')

            # combined column for faceting
            plot_df['animal_video_combo'] = plot_df['animal_idx'].astype(str) + " | " + plot_df['video_name'].astype(str)
            num_unique_combos = len(plot_df['animal_video_combo'].unique())

            if num_unique_combos > 0: # make sure there is something to plot, otherwise skip
                # Only show behavior=1 bouts in the ethogram
                # This filters out the "not-behavior" bouts which would clutter the visualization
                behavior_plot_df = plot_df[plot_df['is_behavior'] == 1]
                
                if not behavior_plot_df.empty:
                    ethogram_plot = (
                        p9.ggplot(behavior_plot_df) +
                        p9.geom_rect(p9.aes(xmin='start', xmax='end', ymin='0.5 * is_gt', ymax='0.5 * is_gt + 0.4', fill='source')) +
                        p9.theme_bw() +
                        p9.facet_wrap('~animal_video_combo', ncol=1, scales='free_x') + #row per each animal video combination
                        p9.scale_y_continuous(breaks=[0.2, 0.7], labels=['Pred', 'GT'], name='') +
                        p9.scale_fill_brewer(type='qual', palette='Set1') +
                        p9.labs(x='Frame', fill='Source', title=f'Ethogram for behavior: {behavior}') +
                        p9.expand_limits(x=0)  # start x-axis at 0
                    )
                    # Adjust height based on the number of unique animal-video combinations
                    ethogram_plot.save(ethogram_output, height=1.5 * num_unique_combos + 2, width=12, dpi=300, limitsize=False, verbose=False)
                    print(f"Ethogram plot saved to {ethogram_output}")
                else:
                    warnings.warn(f"No behavior instances found for behavior {behavior} after filtering for ethogram.")
            else:
                warnings.warn(f"No data to plot for behavior {behavior} after filtering for ethogram.")
        else:
            warnings.warn(f"No annotations found for behavior {behavior} to generate ethogram plot.")


def generate_iou_scan(all_annotations, stitch_scan, filter_scan, threshold_scan, filter_ground_truth: bool = False) -> pd.DataFrame:
    """Scans stitch and filter values to produce a bout-level performance metrics at varying IoU values.

    Args:
        all_annotations: BoutTable dataframe with an additional 'is_gt' column
        stitch_scan: list of potential stitching values to scan
        filter_scan: list of potential filter values to scan
        threshold_scan: list of potential iou thresholds to scan
        filter_ground_truth: allow identical stitching and filters to be applied to the ground truth data?

    Returns:
        pd.DataFrame containing performance across all combinations of the scan
    """
    # Loop over the animals
    performance_df = []
    for (cur_animal, cur_video), animal_df in all_annotations.groupby(['animal_idx', 'video_name']):
        # For each animal, we want a matrix of intersections, unions, and ious
        pr_df = animal_df[~animal_df['is_gt']]
        if len(pr_df) == 0:
            warnings.warn(f'No predictions for {cur_animal} in {cur_video}... skipping.')
            continue
        pr_obj = Bouts(pr_df['start'], pr_df['duration'], pr_df['is_behavior'])
        gt_df = animal_df[animal_df['is_gt']]
        gt_obj = Bouts(gt_df['start'], gt_df['duration'], gt_df['is_behavior'])

        full_duration = pr_obj.starts[-1] + pr_obj.durations[-1]
        pr_obj.fill_to_size(full_duration, 0)
        gt_obj.fill_to_size(full_duration, 0)

        # ugly method to scan over each combination of stitch and filter in one line
        for cur_stitch, cur_filter in zip(*map(np.ndarray.flatten, np.meshgrid(stitch_scan, filter_scan))):
            cur_filter_settings = ClassifierSettings('', interpolate=0, stitch=cur_stitch, min_bout=cur_filter)
            cur_pr = pr_obj.copy()
            cur_gt = gt_obj.copy()
            if filter_ground_truth:
                cur_gt.filter_by_settings(cur_filter_settings)
            # Always apply filters to predictions
            cur_pr.filter_by_settings(cur_filter_settings)

            # Add iou metrics to the list
            int_mat, u_mat, iou_mat = cur_gt.compare_to(cur_pr)
            for cur_threshold in threshold_scan:
                new_performance = {
                    'animal': [cur_animal],
                    'video': [cur_video],
                    'stitch': [cur_stitch],
                    'filter': [cur_filter],
                    'threshold': [cur_threshold],
                }
                metrics = Bouts.calculate_iou_metrics(iou_mat, cur_threshold)
                for key, val in metrics.items():
                    new_performance[key] = [val]
                performance_df.append(pd.DataFrame(new_performance))

    if not performance_df:
        warnings.warn(f"No valid ground truth and prediction pairs found for behavior across all files. Cannot generate performance metrics.")
        # Return an empty DataFrame with expected columns to prevent downstream errors
        return pd.DataFrame(columns=['stitch', 'filter', 'threshold', 'tp', 'fn', 'fp', 'pr', 're', 'f1'])

    performance_df = pd.concat(performance_df)
    # Aggregate over animals
    performance_df = performance_df.groupby(['stitch', 'filter', 'threshold'])[['tp', 'fn', 'fp']].apply(np.sum).reset_index()
    # Re-calculate PR/RE/F1
    performance_df['pr'] = performance_df['tp'] / (performance_df['tp'] + performance_df['fp'])
    performance_df['re'] = performance_df['tp'] / (performance_df['tp'] + performance_df['fn'])
    performance_df['f1'] = 2 * (performance_df['pr'] * performance_df['re']) / (performance_df['pr'] + performance_df['re'])

    return performance_df
