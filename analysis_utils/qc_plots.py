import pandas as pd
import numpy as np
import re
import plotnine as p9
import time
from datetime import datetime

folder = '/media/bgeuther/Storage/TempStorage/SocialPaper/Play/analysis-2023-07-20/juveniles/'
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
df.groupby(['dataset','exp']).apply(len)
# dataset       exp    
    # adult_female  MDX0063    97
    #               MDX0064    97
    #               MDX0065    96
    #               MDX0066    96
    #               MDX0067    96
    #               MDX0068    97
    #               MDX0069    97
    #               MDX0089    95
    #               MDX0091    95
    #               MDX0092    94
    #               MDX0093    92
    # adult_male    MDB0003     1 --
    #               MDB0004    94
    #               MDB0007    97
    #               MDB0008    96
    #               MDB0009    94
    #               MDB0010    99
    #               MDB0011    62 --
    #               MDB0012    93
    #               MDB0013    97
    #               MDB0014    98
    #               MDB0015    98
    #               MDB0047    92
    #               MDB0048    94
    #               MDB0049    88 --
    #               MDB0050    92
    #               MDB0051    94
    #               MDB0052    89
    #               MDB0053    96
    #               MDB0054    96
    #               MDX0005    94
    #               MDX0006    97
    #               MDX0007    83 --
    #               MDX0008    98
    #               MDX0009    95
    #               MDX0010    93
    #               MDX0012    92
    #               MDX0013    96
    #               MDX0014    92
    #               MDX0017    31 --
    #               MDX0018    96
    # juvenile      MDB0284    93
    #               MDB0285    93
    #               MDB0288    93
    #               MDB0289    94
    #               MDB0290    92
    #               MDB0292    95
    #               MDB0293    95
    #               MDB0297    86 --
    #               MDB0298    97
    #               MDB0300    94
    #               MDB0301    93
    #               MDB0302    94
    #               MDB0303    94
    #               MDB0304    95
    #               MDB0305    92
    #               MDB0313    92
    #               MDX0653    95
    #               MDX0654    81 --


# QA Reporting
# This qa was generated via (run in the total log folder):
# mlr --csv unsparsify *.csv > qa_2023-09-11.csv
qa = pd.read_csv(folder + 'qa_2023-09-11.csv')
# cloud fname only contains the last 4 slashes
qa['fname'] = [re.sub('.*/([^/]*/[^/]*/[^/]*/[^/]*)$','\\1',x) for x in qa['video']]
# Note that if a video was rerun, keep the last entry
qa = qa.drop_duplicates(subset=['fname'], keep='last').reset_index(drop=True)

df = pd.merge(df, qa, how='left', on='fname')
df['time'] = [time.strptime(df.loc[i,'day'] + ' ' + df.loc[i,'time_x'], '%Y-%m-%d %H-%M-%S') for i in range(len(df))]
df['time'] = df['time'].apply(lambda x: pd.Timestamp(datetime(*x[:6])))

# Include metadata for summarizing recommendations
meta_df = pd.read_excel('/home/bgeuther/Downloads/2023-08-04 TOM_TotalQueryForConfluence.xlsx')
meta_df = meta_df[['ExptNumber','sex','Strain','Location']].drop_duplicates()
meta_df['Room'] = [x.split(' ')[0] if isinstance(x,str) else ''  for x in meta_df['Location']]
meta_df['Computer'] = [re.sub('.*(NV[0-9]+).*','\\1',x) if isinstance(x,str) else ''  for x in meta_df['Location']]
meta_df['ExptCleaned'] = [re.sub('.*(MD[XB][0-9]+).*','\\1',x) for x in meta_df['ExptNumber']]
df = pd.merge(df, meta_df, left_on='exp', right_on='ExptCleaned', how='left')

# Plot for QC
(
    p9.ggplot(df, p9.aes(x='time', y='avg_longtermid_count', color='dataset', shape='Strain')) + 
    p9.geom_point() +
    p9.facet_wrap('~exp', scales='free_x') +
    p9.theme_bw()
).draw().show()

groupings = df.groupby(['exp','Strain','sex','dataset'])
quality_df = groupings.agg({'avg_longtermid_count': np.mean, 'hour':len}).reset_index()

# Set a really high bar of <6 hours in the 4-day experiment missing.
low_count = quality_df['exp'][quality_df['hour'] < 90]

def print_quality_subset_summary(quality_df, threshold):
    subset_df = quality_df[quality_df['avg_longtermid_count']>threshold]
    removed_df = quality_df['exp'][quality_df['avg_longtermid_count']<threshold]
    print(f'Experiments: {removed_df.to_list()}')
    print(f'Experiment removed count: {len(removed_df)}')
    print(subset_df.groupby(['Strain','sex','dataset']).apply(len).reset_index())

print_quality_subset_summary(quality_df, 2.75)

