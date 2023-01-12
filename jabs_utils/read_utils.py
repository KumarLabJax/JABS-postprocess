import pandas as pd
import numpy as np
import h5py
import json
import os
import re
from datetime import datetime
import scipy
import globalflow as gflow
import cv2

import jabs_utils.project_utils as putils
from jabs_utils.bout_utils import rle, filter_data, get_bout_dists

# Reads in a single file and returns a dataframe of events
# threshold should only be adjusted if you know what you're doing (most ML classifiers expect to be using a 0.5 threshold)
# stitch_bouts is the length of a gap to merge
# filter_bouts is the minimum length of a bout to keep
# trim_time allows the user to read only in a portion of the data (Default of none for reading in all data)
# Note that we carry forward the "pose missing" and "not behavior" events alongside the "behavior" events
def parse_predictions(pred_file: os.path, threshold: float=0.5, interpolate_size: int=0, stitch_bouts: int=0, filter_bouts: int=0, trim_time: tuple[int, int]=None):
	# Read in the raw data
	with h5py.File(pred_file, 'r') as f:
		data = f['predictions/predicted_class'][:]
		probability = f['predictions/probabilities'][:]
	if trim_time is not None:
		data = data[:,trim_time[0]:trim_time[1]]
		probability = probability[:,trim_time[0]:trim_time[1]]
	# Early exit if no animals had predictions
	if np.shape(data)[0] == 0:
		return pd.DataFrame({'animal_idx':[-1], 'start':[0], 'duration':[0], 'is_behavior':[-1]})
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
		# Interpolate missing data first
		if interpolate_size > 0:
			cur_starts, cur_durations, cur_values = filter_data(cur_starts, cur_durations, cur_values, max_gap_size=interpolate_size, values_to_remove=[-1])
		# Filter out short gaps next
		if stitch_bouts > 0:
			cur_starts, cur_durations, cur_values = filter_data(cur_starts, cur_durations, cur_values, max_gap_size=stitch_bouts, values_to_remove=[0])
		# Filter out short predictions last
		if filter_bouts > 0:
			cur_starts, cur_durations, cur_values = filter_data(cur_starts, cur_durations, cur_values, max_gap_size=filter_bouts, values_to_remove=[1])
		tmp_df = pd.DataFrame({'animal_idx':idx, 'start':cur_starts, 'duration':cur_durations, 'is_behavior':cur_values})
		rle_data.append(tmp_df)
	rle_data = pd.concat(rle_data).reset_index(drop=True)
	return rle_data

# Makes a rle result of no predictions on any mice
def make_no_predictions(pose_file: os.path):
	pose_v = putils.get_pose_v(pose_file)
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

# Reads in JABS bout annotation files and places it in the same format as prodiction RLE
# If no behavior is specified, it will read all behaviors in the file
def parse_jabs_annotations(file, behavior: str=None):
	with open(file, 'r') as f:
		data = json.load(f)
	vid_name = data['file']
	df_list = []
	for animal_idx, labels in data['labels'].items():
		for cur_behavior, annotations in labels.items():
			if behavior is None or behavior == cur_behavior:
				try:
					# Alternative for only reading in positive annotations
					#df_list.append(pd.concat([pd.DataFrame({'animal_idx':[animal_idx], 'behavior':[cur_behavior], 'start':[x['start']], 'duration':[x['end']-x['start']+1], 'is_behavior':[1]}) for x in annotations if x['present']]))
					df_list.append(pd.concat([pd.DataFrame({'animal_idx':[animal_idx], 'behavior':[cur_behavior], 'start':[x['start']], 'duration':[x['end']-x['start']+1], 'is_behavior':[x['present']]}) for x in annotations]))
				except ValueError:
					print(cur_behavior + ' for ' + animal_idx + ' contained no positive annotations, skipping.')
	if len(df_list)>0:
		df_list = pd.concat(df_list)
		df_list['video'] = os.path.splitext(vid_name)[0]
	else:
		df_list = pd.DataFrame({'animal_idx':[], 'behavior':[], 'start':[], 'duration':[], 'is_behavior':[], 'video':[]})
	return df_list

# Reads in activity data into a matrix of shape [animal_idx, frame_idx] where each element contains the distance travelled in that frame
# Negative values indicate that the mouse was not present to make a measurement
# Smoothing indicates the number of frames to convolve an average (0 = no smoothing)
# Distance is calculated by the pose version.
# 	pose_v6 uses segmentation centroid motion
# 	pose_v2-5 uses pose centroid motion (ignoring tail points)
# Note that all_activity returned is NOT re-sorted by linking_dict (within-video ID -> within-experiment ID) and is still for an individual file.
def read_activity_folder(folder: os.path, activity_threshold: float, interpolate_size: int, stitch_bouts: int, filter_bouts: int, smooth: int=0, forced_pose_v: int=None, linking_dict: dict=None, activity_dict: dict={}):
	# Figure out what pose files exist
	files_in_experiment = putils.get_poses_in_folder(folder)
	all_predictions = []
	for cur_file in files_in_experiment:
		# Parse out the video name from the pose file
		video_name = putils.pose_to_video(cur_file)
		date_format = r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}'
		video_prefix = re.sub('_' + date_format, '', video_name)
		# Extract date from the name
		time_str = re.search(date_format, video_name).group()
		formatted_time = str(datetime.strptime(time_str, '%Y-%m-%d_%H-%M-%S'))
		if cur_file in activity_dict.keys():
			cur_activity = activity_dict[cur_file]
		else:
			cur_activity = parse_activity(cur_file, smooth=smooth, forced_pose_v=forced_pose_v)
			activity_dict[cur_file] = cur_activity
		# Convert the raw activity into activity bouts
		# Threshold the activity data
		activity_behavior = (cur_activity > activity_threshold).astype(np.int)
		activity_behavior[cur_activity<0] = -1
		rle_data = []
		for idx in np.arange(len(activity_behavior)):
			cur_starts, cur_durations, cur_values = rle(activity_behavior[idx])
			# Interpolate missing data first
			if interpolate_size > 0:
				cur_starts, cur_durations, cur_values = filter_data(cur_starts, cur_durations, cur_values, max_gap_size=interpolate_size, values_to_remove=[-1])
			# Filter out short gaps next
			if stitch_bouts > 0:
				cur_starts, cur_durations, cur_values = filter_data(cur_starts, cur_durations, cur_values, max_gap_size=stitch_bouts, values_to_remove=[0])
			# Filter out short predictions last
			if filter_bouts > 0:
				cur_starts, cur_durations, cur_values = filter_data(cur_starts, cur_durations, cur_values, max_gap_size=filter_bouts, values_to_remove=[1])
			tmp_df = pd.DataFrame({'animal_idx':idx, 'start':cur_starts, 'duration':cur_durations, 'is_behavior':cur_values})
			tmp_df['distance'] = get_bout_dists(cur_starts, cur_durations, cur_activity[idx])
			rle_data.append(tmp_df)
		rle_data = pd.concat(rle_data).reset_index(drop=True)
		# Toss data into the full matrix
		rle_data['time'] = formatted_time
		rle_data['exp_prefix'] = video_prefix
		rle_data['video_name'] = video_name
		all_predictions.append(rle_data)
	all_predictions = pd.concat(all_predictions).reset_index(drop=True)
	# Correct for identities across videos
	if linking_dict is None:
		linking_dict = link_identities(folder)
	all_predictions['longterm_idx'] = [linking_dict[x][y] if x in linking_dict.keys() and y in linking_dict[x].keys() else -1 for x,y in zip(all_predictions['video_name'].values, all_predictions['animal_idx'])]
	return all_predictions, linking_dict, activity_dict

# Reads in activity data for a single file
def parse_activity(pose_file: os.path, smooth: int=0, forced_pose_v: int=None):
	pose_v = putils.get_pose_v(pose_file)
	if forced_pose_v is not None:
		assert pose_v >= forced_pose_v
		pose_v = forced_pose_v
	# Read in the data to produce a center_data variable which is of shape [n_frame, n_animal, 2] containing all [x,y] centers
	if pose_v == 2 or pose_v == 3:
		raise NotImplementedError('Pose v2 and v3 distance not supported yet.')
	elif pose_v == 4 or pose_v == 5:
		with h5py.File(pose_file, 'r') as f:
			pose_data = f['poseest/points'][:,:,:10,:]
			animal_ids = f['poseest/instance_embed_id'][:]
			cm_per_px = f['poseest'].attrs['cm_per_pixel']
		animals = np.unique(animal_ids)
		animals = animals[animals!=0]
		center_data = np.zeros([pose_data.shape[0], max(animals), 2])
		for frame in np.arange(len(pose_data)):
			for i, cur_animal in enumerate(animal_ids[frame]):
				# Skip calculations for animals not assigned an ID
				if cur_animal == 0:
					pass
				else:
					center_data[frame,cur_animal-1,:] = get_pose_center(pose_data[frame,i])
	elif pose_v == 6:
		raise NotImplementedError('Pose v6 distance not supported yet.')
	else:
		raise NotImplementedError('Pose version not detected for ' + str(pose_file) + ' (' + str(pose_v) + '). Cannot interpret data.')
	# Mask out missing data before calculating gradients (distances)
	center_data = np.ma.array(center_data, mask=np.tile(np.expand_dims(np.all(center_data==0, axis=2), axis=-1), [1,1,2]))
	dists = np.gradient(center_data, axis=0)
	dists = np.hypot(dists[:,:,0], dists[:,:,1])
	# We need to fill the data before the smoothing because the convolve doesn't operate on masked arrays
	# We happen to use a fill value of 0 to not corrupt actual distances in averages too badly
	dists.fill_value = 0
	dists_mask = dists.mask
	dists = dists.filled()
	# Smooth if requested
	if smooth > 0:
		smoothed_dists = []
		for animal_idx in np.arange(np.shape(dists)[1]):
			# Mean smooth using convolve
			smoothed_dists.append(scipy.signal.fftconvolve(dists[:,animal_idx], np.ones([smooth])/smooth, mode='same'))
		dists = np.stack(smoothed_dists, axis=1)
	# Finally convert to pixel space
	dists = dists * cm_per_px
	# Fix any negative values
	dists[dists_mask] = -1
	# Return the filled matrix transposed (distances stored as animal_id x frame)
	return dists.T

# Returns a center of a pose using the centroid of a convex hull
def get_pose_center(pose: np.array):
	tmp_pose = pose[np.all(pose!=0,axis=1),:]
	if len(tmp_pose)>3:
		convex_hull = cv2.convexHull(tmp_pose.astype(np.float32))
		moments = cv2.moments(convex_hull)
		# If the area is 0, return a default value
		if moments['m00'] == 0:
			return np.array([0,0])
		return np.array([moments['m10']/moments['m00'], moments['m01']/moments['m00']])
	elif len(tmp_pose)>0:
		return np.mean(tmp_pose, axis=0)
	else:
		return np.array([0,0])

# Returns a center of a segmentation
def get_segmentation_center(contours: np.array):
	raise NotImplementedError('Segmentation centers not supported yet.')

# Reads in a collection of files related to an experiment in a folder
# Warning: If linking_dict is supplied but does not contain correct keys, it will unassign identity data
def read_experiment_folder(folder: os.path, behavior: str, interpolate_size: int, stitch_bouts: int, filter_bouts: int, linking_dict: dict=None, activity_dict: dict={}):
	# Figure out what pose files exist
	files_in_experiment = putils.get_poses_in_folder(folder)
	all_predictions = []
	for cur_file in files_in_experiment:
		# Parse out the video name from the pose file
		video_name = putils.pose_to_video(cur_file)
		date_format = r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}'
		video_prefix = re.sub('_' + date_format, '', video_name)
		# Extract date from the name
		time_str = re.search(date_format, video_name).group()
		formatted_time = str(datetime.strptime(time_str, '%Y-%m-%d_%H-%M-%S'))
		# Check if there are behavior predictions and read in data appropriately
		prediction_file = putils.pose_to_prediction(cur_file, behavior)
		if os.path.exists(prediction_file):
			predictions = parse_predictions(prediction_file, interpolate_size=interpolate_size, stitch_bouts=stitch_bouts, filter_bouts=filter_bouts)
			# Add in distance traveled during bouts, if available
			if cur_file in activity_dict.keys():
				predictions['distance'] = 0
				for idx in np.unique(predictions['animal_idx']):
					if idx != -1:
						rows_to_assign = predictions['animal_idx']==idx
						predictions.loc[rows_to_assign,'distance'] = get_bout_dists(predictions.loc[rows_to_assign,'start'], predictions.loc[rows_to_assign,'duration'], activity_dict[cur_file][idx])
			else:
				pass
		else:
			predictions = make_no_predictions(cur_file)
		# Toss data into the full matrix
		predictions['time'] = formatted_time
		predictions['exp_prefix'] = video_prefix
		predictions['video_name'] = video_name
		all_predictions.append(predictions)
	all_predictions = pd.concat(all_predictions).reset_index(drop=True)
	# Correct for identities across videos
	if linking_dict is None:
		linking_dict = link_identities(folder)
	all_predictions['longterm_idx'] = [linking_dict[x][y] if x in linking_dict.keys() and y in linking_dict[x].keys() else -1 for x,y in zip(all_predictions['video_name'].values, all_predictions['animal_idx'])]
	return all_predictions, linking_dict

# Reads in all the annotations of a given project folder
def read_project_annotations(folder: os.path, behavior: str=None):
	if re.search('rotta/annotations', folder):
		annotation_folder = folder
	else:
		annotation_folder = folder + '/rotta/annotations/'
	json_files = [x for x in os.listdir(annotation_folder) if os.path.splitext(x)[1] == '.json']
	jabs_annotations = pd.concat([parse_jabs_annotations(annotation_folder + '/' + x) for x in json_files])
	return jabs_annotations

# Definition for cost of matching for use in global flow graph identity linking
class GraphCosts(gflow.StandardGraphCosts):
	def __init__(self, max_track_len: int=2) -> None:
		super().__init__(
			penter=1e-3, pexit=1e-3, beta=0.05, max_obs_time=max_track_len - 1
		)
	def transition_cost(self, x: gflow.FlowNode, y: gflow.FlowNode) -> float:
		tdiff = y.time_index - x.time_index
		# We can just log transform the cosine distances
		# Cosine distance should be from range 0-1
		# We also add 0.1 in between videos to penalize not excluding centers from a video
		# Finally, we add log(0.1) to get a good balance with enter/exits
		logprob = np.log(scipy.spatial.distance.cdist([x.obs], [y.obs], metric='cosine') + 0.1 * tdiff) + np.log(0.1)
		return logprob

# Generates a dictionary of dictionaries to link identities between files
# First layer of dictionaries is the file being translated
# Second layer of dictionaries contains the key of input identity and the value of the identity linked across files
def link_identities(folder: os.path, check_model: bool=False):
	files_in_experiment = putils.get_poses_in_folder(folder)
	vid_names = [putils.pose_to_video(x) for x in files_in_experiment]
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
	# Build and solve the graph
	flowgraph = gflow.build_flow_graph(center_data, GraphCosts(len(center_data)))
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
	pose_v = pose_v = putils.get_pose_v(pose_file)
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
# Currently not used
def hungarian_match_ids(group1, group2):
	dist_mat = scipy.spatial.distance.cdist(group1, group2, metric='cosine')
	row_best, col_best = scipy.optimize.linear_sum_assignment(dist_mat)
	return row_best, col_best
