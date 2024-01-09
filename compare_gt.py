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
		# Also filter the gt based on Leinani's filters
		if cur_behavior in behavior_filters.keys():
			gt_bouts = animal_df[animal_df['is_gt']]
			gt_bouts = gt_bouts[gt_bouts['duration']>=behavior_filters[cur_behavior]][['start','duration']].values
		pr_bouts = animal_df[~animal_df['is_gt']][['start','duration']].values
		int_mat, u_mat, iou_mat = gutils.get_iou_mat(gt_bouts, pr_bouts)
		iou_list.append(iou_mat)
	# For each behavior, we can scan the thresholds for performances
	for threshold in np.arange(0.05,1.01,0.05):
		precision, recall, f1 = gutils.calc_temporal_iou_metrics(iou_list, threshold)
		performance_df.append(pd.DataFrame({'behavior':[cur_behavior], 'threshold':[threshold], 'precision':[precision], 'recall':[recall], 'f1':[f1]}))

performance_df = pd.concat(performance_df)
performance_df = pd.melt(performance_df, id_vars=['behavior','threshold'])

# Plot the performance
(
	p9.ggplot(performance_df, p9.aes(x='threshold', y='value', color='variable'))+
	p9.geom_line()+
	p9.theme_bw()+
	p9.facet_wrap('~behavior')+
	p9.labs(x='IoU Threshold', y='Performance', color='Metric')+
	p9.scale_color_brewer(type='qual', palette='Set1')
).draw().show()

# Plot the bouts
all_annotations['end'] = all_annotations['start']+all_annotations['duration']
factor_mouse = pd.factorize(all_annotations['mouse_idx'])
all_annotations['yax'] = factor_mouse[0]


(
	p9.ggplot(all_annotations[all_annotations['behavior']=='Leave'])+
	p9.geom_rect(p9.aes(xmin='start', xmax='end', ymin='yax + is_gt/2', ymax='yax + is_gt/2 + 0.5', fill='is_gt'))+
	p9.theme_bw()+
	p9.facet_wrap('~behavior')+
	p9.scale_y_continuous(breaks=np.arange(len(factor_mouse[1]))+0.5, labels=factor_mouse[1])+
	p9.scale_fill_brewer(type='qual', palette='Set1')+
	p9.geom_hline(p9.aes(yintercept='y'), pd.DataFrame({'y':np.arange(len(factor_mouse[1])+1)}))+
	p9.labs(x='Frame', fill='GT?')
).draw().show()


# Parameter tuning for filter/stitches
prediction_tuning = []
stitches_to_test = [3, 5, 7, 9, 11, 13, 15]
filters_to_test = [3, 5, 7, 9, 11, 13, 15]
for behavior in pred_behaviors:
	for pose_file in pred_poses:
		cur_video = putils.pose_to_video(pose_file)
		for cur_stitch in stitches_to_test:
			for cur_filter in filters_to_test:
				# Read in only the first 2 minutes of the predictions
				prediction_df = rutils.parse_predictions(putils.pose_to_prediction(pose_file, behavior), stitch_bouts=cur_stitch, filter_bouts=cur_filter, trim_time=(0,60*30*2))
				prediction_df['behavior'] = behavior
				prediction_df['video'] = cur_video
				prediction_df['stitch'] = cur_stitch
				prediction_df['filter'] = cur_filter
				prediction_df = prediction_df[prediction_df['is_behavior']==1]
				prediction_tuning.append(prediction_df)

prediction_tuning = pd.concat(prediction_tuning).reset_index(drop=True)

gt_annotations['is_gt'] = True
prediction_tuning['is_gt'] = False
tuning_annotations = pd.concat([gt_annotations, prediction_tuning])
tuning_annotations['behavior'] = [re.sub('-','_',x) for x in tuning_annotations['behavior']]
tuning_annotations['mouse_idx'] = tuning_annotations['video'] + '_' + tuning_annotations['animal_idx'].astype(str)
tuning_annotations['mouse_idx'] = tuning_annotations['mouse_idx'].astype('category')

performance_df = []
# Loop over the animals by behavior
for cur_behavior, tmp_df in tuning_annotations.groupby('behavior'):
	tuning_iou_list = []
	for cur_animal, animal_df in tmp_df.groupby('mouse_idx'):
		# For each animal, we want a matrix of intersections, unions, and ious
		gt_bouts = animal_df[animal_df['is_gt']][['start','duration']].values
		# Cycle over the new stitch + filter metrics
		for grp, pr_df in animal_df[~animal_df['is_gt']].groupby(['stitch','filter']):
			pr_bouts = pr_df[~pr_df['is_gt']][['start','duration']].values
			int_mat, u_mat, iou_mat = gutils.get_iou_mat(gt_bouts, pr_bouts)
			tuning_iou_list.append({'stitch':grp[0], 'filter':grp[1], 'iou':iou_mat})
	# For each behavior, we can scan the thresholds for performances
	for threshold in np.arange(0.05,1.01,0.05):
		# Different combinations
		combinations = np.array(np.meshgrid(stitches_to_test, filters_to_test)).T.reshape([-1,2])
		for cur_combo in combinations:
			iou_list = [x['iou'] for x in tuning_iou_list if x['stitch']==cur_combo[0] and x['filter']==cur_combo[1]]
			precision, recall, f1 = gutils.calc_temporal_iou_metrics(iou_list, threshold)
			performance_df.append(pd.DataFrame({'behavior':[cur_behavior], 'threshold':[threshold], 'precision':[precision], 'recall':[recall], 'f1':[f1], 'stitch':[cur_combo[0]], 'filter':[cur_combo[1]]}))

performance_df = pd.concat(performance_df)
performance_df = pd.melt(performance_df, id_vars=['behavior','threshold','stitch','filter'])

# Plot the performance
(
	p9.ggplot(performance_df[np.logical_and(performance_df['variable']=='f1',performance_df['threshold']==0.5)], p9.aes(x='stitch', y='filter', fill='value'))+
	p9.geom_tile(p9.aes(width=2, height=2), color='black')+
	p9.geom_text(p9.aes(label='np.round(value,2)'))+
	p9.theme_bw()+
	p9.facet_wrap('~behavior')+
	p9.labs(x='Stitch', y='Filter', fill='F1 @ 0.5 IoU')+
	p9.scale_color_brewer(type='qual', palette='Set1')
).draw().show()
