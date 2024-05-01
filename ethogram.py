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

'''
This is the original code modified from Brian's version of compare_gt.py to make a huddling specific graph for the 
original JABS classifier -- not meant for the heuristic huddling classifier
'''

gt_annotations_folder = '/Users/szadys/Desktop/MiceVideos'
gt_annotations = rutils.read_project_annotations(gt_annotations_folder)

print(gt_annotations)

# Read in the gt predictions
predictions_folder = '/Users/szadys/Desktop/MiceVideos'
pred_behaviors = putils.get_behaviors_in_folder(predictions_folder)
pred_poses = putils.get_poses_in_folder(predictions_folder)

print(pred_poses)

predictions = []

predictions = pd.concat(predictions).reset_index(drop=True)
gt_annotations['is_gt'] = True
predictions['is_gt'] = False
all_annotations = pd.concat([gt_annotations, predictions])
all_annotations['behavior'] = [re.sub('-','_',x) for x in all_annotations['behavior']]
all_annotations['mouse_idx'] = all_annotations['video'] + '_' + all_annotations['animal_idx'].astype(str)
all_annotations['mouse_idx'] = all_annotations['mouse_idx'].astype('category')

all_annotations['end'] = all_annotations['start']+all_annotations['duration']
factor_mouse = pd.factorize(all_annotations['mouse_idx'])
all_annotations['yax'] = factor_mouse[0]

huddles = all_annotations[all_annotations['behavior']=='huddling']
under_3600 = huddles[huddles['start']<3600]
under_3600.loc[under_3600["end"] > 3600, 'end'] = 3600
(p9.ggplot(under_3600)+ 
 p9.geom_rect(p9.aes(xmin='start', xmax='end',ymin='yax + is_gt/2', ymax='yax + is_gt/2 + 0.5', fill='is_gt'))+
 p9.theme_bw()+p9.facet_wrap('~behavior')+
 p9.scale_y_continuous(breaks=np.arange(len(factor_mouse[1]))+0.5,labels=factor_mouse[1])+
 p9.scale_fill_brewer(type='qual', palette='Set1')+p9.geom_hline(p9.aes(yintercept='y'), pd.DataFrame({'y':np.arange(len(factor_mouse[1])+1)}))+
 p9.labs(x='Frame', fill='Ground Truth?', labels= ('Ground Truth', 'Training Set'))).save("test.png",height=6, width=12, dpi=300)
