"Associated lines of code that deal with the comparison of predictions (from classify.py) and GT annotation (from a JABS project)."""

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

# Read in the gt annotations
gt_annotations_folder = '/Users/szadys/Desktop/revised_pose'
gt_annotations = rutils.read_project_annotations(gt_annotations_folder)

# Read in the gt predictions
predictions_folder = '/Users/szadys/Desktop/predictions'
pred_behaviors = putils.get_behaviors_in_folder(predictions_folder)
pred_poses = putils.get_poses_in_folder(predictions_folder)

# behavior filters for huddling NEED TO CHANGE
behavior_filters = {'Approach':5, 'Chase':9, 'Leave':3, 'Nose_genital':9, 'Nose_nose':9}
predictions = []
for behavior in pred_behaviors:
	for pose_file in pred_poses:
		cur_video = putils.pose_to_video(pose_file)
		# Read in only the first 2 minutes of the predictions
		if behavior in behavior_filters.keys():
			bout_filter = behavior_filters[behavior]
		else:
			bout_filter = 5
		prediction_df = rutils.parse_predictions(putils.pose_to_prediction(pose_file, behavior), stitch_bouts=5, filter_bouts=bout_filter, trim_time=(0,60*30*2))
		prediction_df['behavior'] = behavior
		prediction_df['video'] = cur_video
		prediction_df = prediction_df[prediction_df['is_behavior']==1]
		predictions.append(prediction_df)

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
