import pandas as pd
import numpy as np
import os, sys
import glob
import re
import argparse
import h5py
from datetime import datetime
import scipy
import globalflow as gflow

BEHAVIOR_CLASSIFY_VERSION = 1

# Run length encoding, implemented using numpy
# Accepts a 1d vector
# Returns a tuple containing (starts, durations, values)
def rle(inarray):
	ia = np.asarray(inarray)
	n = len(ia)
	if n == 0: 
		return (None, None, None)
	else:
		y = ia[1:] != ia[:-1]
		i = np.append(np.where(y), n - 1)
		z = np.diff(np.append(-1, i))
		p = np.cumsum(np.append(0, z))[:-1]
		return(p, z, ia[i])

# Removes states of RLE data based on filters
# Returns a new tuple of RLE data
def filter_data(starts, durations, values, max_gap_size: int, value_to_remove: int = 0):
	gaps_to_remove = np.logical_and(values==value_to_remove, durations<max_gap_size)
	new_durations = np.copy(durations)
	new_starts = np.copy(starts)
	new_states = np.copy(values)
	if np.any(gaps_to_remove):
		# Go through backwards removing gaps
		for cur_gap in np.where(gaps_to_remove)[0][::-1]:
			# Nothing earlier or later to join together, ignore
			if cur_gap == 0 or cur_gap == len(new_durations)-1:
				pass
			else:
				cur_duration = np.sum(new_durations[cur_gap-1:cur_gap+2])
				new_durations[cur_gap-1] = cur_duration
				new_durations = np.delete(new_durations, [cur_gap, cur_gap+1])
				new_starts = np.delete(new_starts, [cur_gap, cur_gap+1])
				new_states = np.delete(new_states, [cur_gap, cur_gap+1])
	return new_starts, new_durations, new_states

# Reads in a single file and returns a dataframe of events
# Note that we carry forward the "pose missing" and "not behavior" events alongside the "behavior" events
# TODO: Add in filtering
def parse_predictions(pred_file: os.path, threshold: float=0.5):
	# Read in the raw data
	with h5py.File(pred_file, 'r') as f:
		data = f['predictions/predicted_class'][:]
		probability = f['predictions/probabilities'][:]
	# Transform probabilities of binary classifier into those of the behavior
	probability[data==0] = 1-probability[data==0]
	# Apply a new threshold
	# Note that when data==-1, this indicates "no pose to predict on"
	data[np.logical_and(probability>=threshold, data!=-1)] = 1
	data[np.logical_and(probability<threshold, data!=-1)] = 0
	# RLE the data
	rle_data = []
	for idx in np.arange(len(data)):
		cur_starts, cur_durations, cur_values = rle(data[idx])
		tmp_df = pd.DataFrame({'animal_idx':idx, 'start':cur_starts, 'duration':cur_durations, 'is_behavior':cur_values})
		rle_data.append(tmp_df)
	rle_data = pd.concat(rle_data).reset_index(drop=True)
	return rle_data

# Makes a rle result of no predictions on any mice
def make_no_predictions(pose_file: os.path):
	pose_v = int(re.sub('.*_pose_est_v([2-5]).*', '\\1', pose_file))
	if pose_v == 2:
		n_animals = 1
		n_frames = np.shape(f['poseest/points'])[0]
	elif pose_v == 3:
		with h5py.File(pose_file, 'r') as f:
			n_animals = np.max(f['poseest/instance_count'][:])
			n_frames = np.shape(f['poseest/points'])[0]
	elif pose_v >= 4:
		with h5py.File(pose_file, 'r') as f:
			n_animals = np.shape(f['poseest/instance_id_center'])[0]
			n_frames = np.shape(f['poseest/points'])[0]
	rle_data = []
	for idx in np.arange(n_animals):
		rle_data.append(pd.DataFrame({'animal_idx':idx, 'start':[0], 'duration':[n_frames], 'is_behavior':-1}))
	rle_data = pd.concat(rle_data).reset_index(drop=True)
	return rle_data

# Generates a list of experiment folders in a project
# Assumes that all videos for a multi-day experiment exist in a single folder
# Assumes that the prediction paths look like 'project/EXPERIMENT_FOLDER/video_behavior/v1/behavior_name/video.h5'
def get_experiments_in_folder(folder: os.path):
	# Find all the behavior prediction folders (always named v1)
	possible_folders = glob.glob(folder + '**/v' + str(BEHAVIOR_CLASSIFY_VERSION), recursive=True)
	# Extract the folder 2 above that, which would be the folder containing all experiments in a 4-day grouping
	possible_folders = [re.sub('(.*)([^/]*/){2}v' + str(BEHAVIOR_CLASSIFY_VERSION),'\\1',x) for x in possible_folders]
	experiment_folder_list = list(set(possible_folders))
	return experiment_folder_list

# Generates a list of behavior predictions found in project folder
# Assumes that the prediction paths look like 'project/experiment_folder/video_behavior/v1/BEHAVIOR_NAME/video.h5'
def get_behaviors_in_folder(folder: os.path):
	possible_files = glob.glob(folder + '**/v' + str(BEHAVIOR_CLASSIFY_VERSION) + '/*', recursive=True)
	behaviors = [re.sub('.*/','',x) for x in possible_files]
	behaviors = list(set(behaviors))
	return behaviors

# Reads in a collection of files related to an experiment in a folder
def read_experiment_folder(folder: os.path, behavior: str):
	# Figure out what pose files exist
	files_in_experiment = sorted(glob.glob(folder + '/*_pose_est_v[2-5].h5'))
	all_predictions = []
	for cur_file in files_in_experiment:
		# Parse out the video name from the pose file
		video_name = re.sub('.*/([^/]*)_pose_est_v.*', '\\1', cur_file)
		date_format = r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}'
		video_prefix = re.sub('_' + date_format, '', video_name)
		# Extract date from the name
		time_str = re.search(date_format, video_name).group()
		formatted_time = str(datetime.strptime(time_str, '%Y-%m-%d_%H-%M-%S'))
		# Check if there are behavior predictions and read in data appropriately
		prediction_file = re.sub('_pose_est_v[2-5].h5', '_behavior/v1/' + behavior + '/' + video_name + '.h5', cur_file)
		if os.path.exists(prediction_file):
			predictions = parse_predictions(prediction_file)
		else:
			predictions = make_no_predictions(cur_file)
		# Toss data into the full matrix
		predictions['time'] = formatted_time
		predictions['exp_prefix'] = video_prefix
		predictions['video_name'] = video_name
		all_predictions.append(predictions)
	all_predictions = pd.concat(all_predictions).reset_index(drop=True)
	# Correct for identities across videos
	linking_dict = link_identities(folder)
	all_predictions['longterm_idx'] = [linking_dict[x][y] if x in linking_dict.keys() and y in linking_dict[x].keys() else -1 for x,y in zip(all_predictions['video_name'].values, all_predictions['animal_idx'])]
	return all_predictions

# Transforms raw data per-experiment into binned results
def generate_binned_results(df: pd.DataFrame, bin_size_frames: int=108000):
	raise NotImplementedError

# Generates a dictionary of dictionaries to link identities between files
# First layer of dictionaries is the file being translated
# Second layer of dictionaries contains the key of input identity and the value of the identity linked across files
def link_identities(folder: os.path, check_model: bool=False):
	files_in_experiment = sorted(glob.glob(folder + '/*_pose_est_v[2-5].h5'))
	vid_names = [re.sub('.*/([^/]*)_pose_est_v.*', '\\1', x) for x in files_in_experiment]
	# Read in all the center data
	center_locations = []
	identified_model = None
	for cur_file in files_in_experiment:
		cur_centers, cur_model = read_pose_ids(cur_file)
		# Check that new data conforms to the name of the previous model
		if identified_model is None:
			identified_model = cur_model
		elif check_model:
			assert identified_model == cur_model
		# Transform the data into a better format for indexing
		cur_centers = pd.DataFrame({'file':cur_file, 'id':np.arange(len(cur_centers)), 'centers':[x for x in cur_centers]})
		center_locations.append(cur_centers)
	# center_locations = pd.concat(center_locations)
	center_data = [[observation['centers'] for cur_index, observation in cur_vid.iterrows()] for cur_vid in center_locations]
	# Definition for cost of matching
	class GraphCosts(gflow.StandardGraphCosts):
		def __init__(self) -> None:
			super().__init__(
				penter=1e-3, pexit=1e-3, beta=0.05, max_obs_time=len(center_data) - 1
			)
		def transition_cost(self, x: gflow.FlowNode, y: gflow.FlowNode) -> float:
			tdiff = y.time_index - x.time_index
			# We can just log transform the cosine distances
			# Cosine distance should be from range 0-1
			# We also add 0.1 in between videos to penalize not excluding centers from a video
			# Finally, we add log(0.1) to get a good balance with enter/exits
			logprob = np.log(scipy.spatial.distance.cdist([x.obs], [y.obs], metric='cosine') + 0.1 * tdiff) + np.log(0.1)
			return logprob
	# Build and solve the graph
	flowgraph = gflow.build_flow_graph(center_data, GraphCosts())
	flowdict, ll, num_traj = gflow.solve(flowgraph)
	# Extract the tracks out of the dict
	track_starts = [key for key, val in flowdict['S'].items() if val == 1]
	tracklets = []
	for tracklet_idx, start in enumerate(track_starts):
		# Seed first values
		cur_node = start
		# Format is [global_id, vid_idx, id_in_vid]
		cur_tracklet = [[tracklet_idx, cur_node.time_index, cur_node.obs_index]]
		next_nodes = flowdict[cur_node]
		# Continue until terminated
		while 1 in next_nodes.values():
			next_node_idx = np.argmax(list(next_nodes.values()))
			cur_node = list(next_nodes.keys())[next_node_idx]
			if cur_node == 'T':
				break
			next_nodes = flowdict[cur_node]
			# Since the graph has v and u per observation, only add u
			if cur_node.tag == 'u':
				cur_tracklet.append([tracklet_idx, cur_node.time_index, cur_node.obs_index])
		tracklets.append(cur_tracklet)
	# Re-format into the dict of dicts for easier translation
	tracklets = np.concatenate(tracklets)
	vid_dict = {}
	for i, vid_name in enumerate(vid_names):
		matches_to_add = tracklets[:,1] == i
		if np.any(matches_to_add):
			vid_dict[vid_name] = dict(zip(tracklets[matches_to_add,0], tracklets[matches_to_add,2]))
	return vid_dict

# Helper function for reading a pose files identity data
def read_pose_ids(pose_file: os.path):
	pose_v = int(re.sub('.*_pose_est_v([2-5]).*', '\\1', pose_file))
	if pose_v == 2:
		raise NotImplementedError('Single mouse pose doesn\'t run on longterm experiments.')
	# No longterm IDs exist, provide a default value of the correct shape
	elif pose_v == 3:
		with h5py.File(pose_file, 'r') as f:
			num_mice = np.max()
			centers = np.zeros([num_mice, 0], dtype=np.float64)
			model_used = 'None'
		# Linking identities across multiple files does not yet support this, so throw an error here
		raise NotImplementedError('Pose v3 identities cannot be linked across videos.')
	elif pose_v >= 4:
		with h5py.File(pose_file, 'r') as f:
			centers = f['poseest/instance_id_center'][:]
			model_used = f['poseest/identity_embeds'].attrs['network']
	return centers, model_used

# Matches a pair of IDs using the cosine distance
# Pairs ids using a hungarian matching algorithm (lowest total cost)
def hungarian_match_ids(group1, group2):
	dist_mat = scipy.spatial.distance.cdist(group1, group2, metric='cosine')
	row_best, col_best = scipy.optimize.linear_sum_assignment(dist_mat)
	return row_best, col_best

# Writes the header of filers used in this script to file
def write_experiment_header(out_file: os.path, args):
	raise NotImplementedError

def write_experiment_bouts():
	raise NotImplementedError

def write_experiment_binned():
	raise NotImplementedError

def generate_behavior_tables(args, behavior: str):
	# Detect all the experiments in a folder
	exp_folders = get_experiments_in_folder(args.project_folder)
	# Read in all the experiments (RLE format)
	all_experiment_data = []
	for cur_experiment in exp_folders:
		experiment_data = read_experiment_folder(cur_experiment, behavior)
		all_experiment_data.append(experiment_data)
	# Merge experiments into a single project (RLE format)
	all_experiment_data = pd.concat(all_experiment_data)
	# TODO:
	# Write project bout output
	# Convert project into binned data
	# Write binned project output
	return all_experiment_data

def main(argv):
	parser = argparse.ArgumentParser(description='Script that transforms JABS behavior predictions for a project folder into an easier to work with set of files.')
	parser.add_argument('--project_folder', help='Folder that contains the project with both pose files and behavior prediction files', required=True)
	parser.add_argument('--stitch_gap', help='Number of frames in which frames sequential behavior prediction bouts will be joined, default=5', default=5, type=int)
	parser.add_argument('--min_bout_length', help='Minimum number of frames in which a behavior prediction must be to be considered, default=5', default=5, type=int)
	parser.add_argument('--out_bin_size', help='Time duration used in binning the results, default=60', default=60, type=int)
	parser.add_argument('--behavior', help='Behavior to produce a table for, default=all behaviors in project folder', default=None, type=str)
	parser.add_argument('--out_prefix', help='File prefix to write output tables (prefix_bouts.csv and prefix_summaries.csv), default=behavior', default=None, type=str)
	#
	args = parser.parse_args()
	# Detect or select which behavior we want tables for
	if args.behavior is None:
		behaviors = get_behaviors_in_folder(args.project_folder)
	else:
		behaviors = list(args.behavior)
	# Loop through all behaviors:
	print('Generating behavior tables for behaviors: ' + ', '.join(behaviors) + ' in ' + args.project_folder + '...')
	for behavior in behaviors:
		cur_table = generate_behavior_tables(args, behavior)

if __name__  == '__main__':
	main(sys.argv[1:])
