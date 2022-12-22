import pandas as pd
import plotnine as p9
import re
import numpy as np
from itertools import chain

from analysis_utils.parse_table import read_ltm_summary_table, filter_experiment_time
from analysis_utils.plots import generate_time_vs_feature_plot
import jabs_utils.read_utils as rutils
import jabs_utils.project_utils as putils
import analysis_utils.gt_utils as gutils

# Read in the gt annotations
gt_annotations_folder = '/media/bgeuther/Storage/TempStorage/leinani_social_behavior_classifiers/exported_gt/'
gt_annotations = rutils.read_project_annotations(gt_annotations_folder)

# Read in the gt predictions
predictions_folder = '/media/bgeuther/Storage/TempStorage/leinani_social_behavior_classifiers/exp_v5_poses/'
pred_behaviors = putils.get_behaviors_in_folder(predictions_folder)
pred_poses = putils.get_poses_in_folder(predictions_folder)
predictions = []
for behavior in pred_behaviors:
	for pose_file in pred_poses:
		cur_video = putils.pose_to_video(pose_file)
		# Read in only the first 2 minutes of the predictions
		prediction_df = rutils.parse_predictions(putils.pose_to_prediction(pose_file, behavior), stitch_bouts=5, filter_bouts=5, trim_time=(0,60*30*2))
		prediction_df['behavior'] = behavior
		prediction_df['video'] = cur_video
		prediction_df = prediction_df[prediction_df['is_behavior']==1]
		predictions.append(prediction_df)

predictions = pd.concat(predictions).reset_index(drop=True)

# Combine the df to make comparing animals a lot easier
gt_annotations['is_gt'] = True
predictions['is_gt'] = False
all_annotations = pd.concat([gt_annotations, predictions])
all_annotations['behavior'] = [re.sub('-','_',x) for x in all_annotations['behavior']]
all_annotations['mouse_idx'] = all_annotations['video'] + '_' + all_annotations['animal_idx'].astype(str)
all_annotations['mouse_idx'] = all_annotations['mouse_idx'].astype('category')

# Calculate performance metrics
performance_df = []
# Loop over the animals by behavior
for cur_behavior, tmp_df in all_annotations.groupby('behavior'):
	iou_list = []
	for cur_animal, animal_df in tmp_df.groupby('mouse_idx'):
		# For each animal, we want a matrix of intersections, unions, and ious
		gt_bouts = animal_df[animal_df['is_gt']][['start','duration']].values
		pr_bouts = animal_df[~animal_df['is_gt']][['start','duration']].values
		int_mat, u_mat, iou_mat = gutils.get_iou_mat(gt_bouts, pr_bouts)
		iou_list.append(iou_mat)
	# For each behavior, we can scan the thresholds for performances
	for threshold in np.arange(0,1,0.05):
		precision, recall, f1 = gutils.calc_temporal_iou_metrics(iou_list, threshold)
		performance_df.append(pd.DataFrame({'behavior':[cur_behavior], 'threshold':[threshold], 'precision':[precision], 'recall':[recall], 'f1':[f1]}))

performance_df = pd.concat(performance_df)
performance_df = pd.melt(performance_df, id_vars=['behavior','threshold'])

(
	p9.ggplot(performance_df, p9.aes(x='threshold', y='value', color='variable'))+
	p9.geom_line()+
	p9.theme_bw()+
	p9.facet_wrap('~behavior')+
	p9.labs(x='IoU Threshold', y='Performance', color='Metric')+
	p9.scale_color_brewer(type='qual', palette='Set1')
).draw().show()
