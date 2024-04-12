import pandas as pd
import numpy as np
import plotnine as p9
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from jabs_utils.read_utils import link_identities, read_pose_ids

# Test folder with bleach-marked mice
project_folder = '/media/bgeuther/Storage/TempStorage/test-behavior/'
# Black vs Albino coat color has a bit more interpretable data (better separation)
project_folder = '/media/bgeuther/Storage/TempStorage/UCSD_QA/full_run/2020-12-17/'
cur_experiment = project_folder
linking_dict = link_identities(cur_experiment)

video_keys = sorted(linking_dict.keys())
pose_centers = [read_pose_ids(project_folder + x + '_pose_est_v5.h5')[0] for x in video_keys]

# Calculate the PCA so that we can inspect how the data moves along shorter axes
# This isn't perfect since PCA is designed around euclidean distance.
# However, if we normalize to unit vectors both pre- and post- pca, angles are roughly preserved (and now sorted).
# Keeping all components should produce equivalent cosine similarity.
# See https://stats.stackexchange.com/a/453104
all_centers = np.concatenate(pose_centers)
all_centers = normalize(all_centers, axis=1)
pca = PCA(n_components=2)
pca.fit(all_centers)
transformed = normalize(pca.transform(all_centers), axis=0)

center_data = pd.DataFrame({
	# 'video': np.repeat(sorted(linking_dict.keys()), 3),
	'video': np.concatenate([np.repeat(video_keys[i], x.shape[0]) for i, x in enumerate(pose_centers)]),
	'video_num': np.concatenate([np.repeat(i, x.shape[0]) for i, x in enumerate(pose_centers)]),
	'animal_id': np.concatenate([np.arange(x.shape[0]) for x in pose_centers]),
} | {f'pca_{i}': transformed[:, i] for i in range(transformed.shape[1])} | {f'embed_{i}': all_centers[:, i] for i in range(all_centers.shape[1])}
)
center_data['longterm_id'] = [linking_dict[x['video']][x['animal_id']] for _, x in center_data.iterrows()]

melted_df = pd.melt(center_data, id_vars=['video_num', 'longterm_id', 'video', 'animal_id'], value_vars=[f'embed_{i}' for i in range(all_centers.shape[1])] + [f'pca_{i}' for i in range(transformed.shape[1])])

(
	p9.ggplot(melted_df, p9.aes(x='video_num', y='value', color='factor(longterm_id)'))
	+ p9.geom_line()
	+ p9.geom_point()
	+ p9.facet_wrap('variable')
	+ p9.theme_bw()
).draw().show()

(
	p9.ggplot(center_data, p9.aes(x='pca_0', y='pca_1', color='factor(longterm_id)'))
	+ p9.geom_point(size=7)
	+ p9.geom_path()
	+ p9.geom_text(p9.aes(label='video_num'), color='black')
	+ p9.theme_bw()
).draw().show()
