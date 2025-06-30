import re
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotnine as p9

folder = '/media/bgeuther/Storage/TempStorage/SocialPaper/Play/analysis-2023-07-20/juveniles/'
# These lists were generated via video names that exist on dropbox
# /media/bgeuther/Storage/TempStorage/B6-BTBR$ cat all_files_2023-08-14.txt | grep -E '\.avi$|_pose_est_v5\.h5$' | awk '{print $2}' | sed -e 's:\.avi::' -e 's:_pose_est_v5\.h5::' | uniq | sed -e 's:$:_pose_est_v5.h5:' | grep -E $pattern > /media/bgeuther/Storage/TempStorage/SocialPaper/Play/analysis-2023-07-20/juveniles/adult_male_list.txt 
flist = np.loadtxt(folder + 'juvenile_list.txt', dtype=str)
df = pd.DataFrame({'fname': flist, 'dataset': 'juvenile'})

flist2 = np.loadtxt(folder + 'adult_male_list.txt', dtype=str)
df = pd.concat([df, pd.DataFrame({'fname': flist2, 'dataset': 'adult_male'})]).reset_index(drop=True)
flist2 = np.loadtxt(folder + 'adult_female_list.txt', dtype=str)
df = pd.concat([df, pd.DataFrame({'fname': flist2, 'dataset': 'adult_female'})]).reset_index(drop=True)

df['project'], df['computer'], df['exp_date'], df['pose'] = np.split(np.array([row['fname'].split('/') for _, row in df.iterrows()]), 4, axis=-1)
df['exp'], df['day'], df['time'], _, _, df['pose_v'] = np.split(np.array([row['pose'].split('_') for _, row in df.iterrows()]), 6, axis=-1)
df['hour'], df['minute'], df['second'] = np.split(np.array([row['time'].split('-') for _, row in df.iterrows()]), 3, axis=-1)

# How uniform is the data loss?
# df.groupby(['dataset','exp']).apply(len)

# QA Reporting
# This qa was generated via (run in the total log folder):
# mlr --csv unsparsify *.csv > qa_2023-09-25.csv.csv
qa = pd.read_csv(folder + 'qa_2023-09-25.csv')
# cloud fname only contains the last 4 slashes
qa['fname'] = [re.sub('.*/([^/]*/[^/]*/[^/]*/[^/]*)$','\\1',x) for x in qa['video']]
# Note that if a video was rerun, keep the last entry
qa = qa.drop_duplicates(subset=['fname'], keep='last').reset_index(drop=True)

df = pd.merge(df, qa, how='left', on='fname')
df['time'] = [time.strptime(df.loc[i,'day'] + ' ' + df.loc[i,'time_x'], '%Y-%m-%d %H-%M-%S') for i in range(len(df))]
df['time'] = df['time'].apply(lambda x: pd.Timestamp(datetime(*x[:6])))
df['exp_date_timestamp'] = df['exp_date'].apply(lambda x: pd.Timestamp(datetime(*(time.strptime(x, '%Y-%m-%d')[:6]))))
df['rel_exp_time'] = df['time'] - df['exp_date_timestamp']

# Include metadata for summarizing recommendations
meta_df = pd.read_excel('/home/bgeuther/Downloads/2023-08-04 TOM_TotalQueryForConfluence.xlsx')
meta_df = meta_df[['ExptNumber', 'sex', 'Strain', 'Location']].drop_duplicates()
meta_df['Room'] = [x.split(' ')[0] if isinstance(x, str) else '' for x in meta_df['Location']]
meta_df['Computer'] = [re.sub('.*(NV[0-9]+).*', '\\1', x) if isinstance(x, str) else '' for x in meta_df['Location']]
meta_df['ExptCleaned'] = [re.sub('.*(MD[XB][0-9]+).*', '\\1', x) for x in meta_df['ExptNumber']]
df = pd.merge(df, meta_df, left_on='exp', right_on='ExptCleaned', how='left')

# Plot for QC
# (
#     p9.ggplot(df, p9.aes(x='time', y='avg_longtermid_count', color='dataset', shape='Strain'))
#     + p9.geom_point()
#     + p9.facet_wrap('~exp', scales='free_x')
#     + p9.theme_bw()
# ).draw().show()

(
    p9.ggplot(df, p9.aes(x='rel_exp_time', y='avg_longtermid_count', color='exp', shape='Strain'))
    # p9.geom_line() +
    + p9.geom_point()
    + p9.facet_grid('dataset~.')
    + p9.theme_bw()
).draw().show()

groupings = df.groupby(['exp','Strain','sex','dataset'])
quality_df = groupings.agg({'avg_longtermid_count': np.mean, 'hour':len}).reset_index()

# Set a really high bar of 10% missing data in the 4-day experiment.
low_count = quality_df['exp'][quality_df['hour'] < 86].to_list()

def print_quality_subset_summary(quality_df, threshold):
    subset_df = quality_df[quality_df['avg_longtermid_count']>threshold]
    removed_df = quality_df['exp'][quality_df['avg_longtermid_count']<threshold]
    print(f'Experiments: {removed_df.to_list()}')
    print(f'Experiment removed count: {len(removed_df)}')
    print(subset_df.groupby(['Strain','sex','dataset']).apply(len).reset_index())
    return removed_df.to_list()

low_quality = print_quality_subset_summary(quality_df[~np.isin(quality_df['exp'],np.array(low_count))], 2.5)
low_quality_2 = print_quality_subset_summary(quality_df[~np.isin(quality_df['exp'],np.array(low_count))], 2.0)

experiments_to_remove = np.unique(low_count + low_quality)
experiments_to_remove_2 = np.unique(low_count + low_quality_2)

# Save some useful outputs
df['fname'].to_csv(folder + time.strftime('%Y-%m-%d') + '_poses.txt', header=False, index=False)
np.savetxt(folder + time.strftime('%Y-%m-%d') + '_failed_qc.txt', experiments_to_remove, fmt='%s')
np.savetxt(folder + time.strftime('%Y-%m-%d') + '_failed_qc_lenient.txt', experiments_to_remove_2, fmt='%s')

missing_lixit = df['fname'][np.logical_or(np.logical_or(~df['lixit_side_check'].astype(bool), ~df['lixit_location_check'].astype(bool)), df['num_lixit'] != 1)].values
missing_food = df['fname'][~df['food_detected'].astype(bool)].values
missing_corners = df['fname'][~df['arena_detected'].astype(bool)].values
missing_static_objects = np.unique(np.concatenate([missing_lixit, missing_food, missing_corners]))
missing_non_lixit = np.unique(np.concatenate([missing_food, missing_corners]))

np.savetxt(folder + time.strftime('%Y-%m-%d') + '_missing_static_objs.txt', missing_static_objects, fmt='%s')
np.savetxt(folder + time.strftime('%Y-%m-%d') + '_missing_non_lixit.txt', missing_non_lixit, fmt='%s')
