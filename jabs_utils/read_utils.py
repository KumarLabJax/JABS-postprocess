from __future__ import annotations
import pandas as pd
import numpy as np
import h5py
import json
import os
import re
from datetime import datetime
import scipy
import cv2
from itertools import chain

import jabs_utils.project_utils as putils
from jabs_utils.bout_utils import rle, filter_data, get_bout_dists

from typing import Tuple, List, Union

# Reads in a single file and returns a dataframe of events
# threshold should only be adjusted if you know what you're doing (most ML classifiers expect to be using a 0.5 threshold)
# stitch_bouts is the length of a gap to merge
# filter_bouts is the minimum length of a bout to keep
# trim_time allows the user to read only in a portion of the data (Default of none for reading in all data)
# Note that we carry forward the "pose missing" and "not behavior" events alongside the "behavior" events
def parse_predictions(pred_file: os.path, threshold_min: float=0.5, threshold_max=1.0, interpolate_size: int=0, stitch_bouts: int=0, filter_bouts: int=0, trim_time: Tuple[int, int]=None):
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
	data[np.logical_and(probability>=threshold_min, data!=-1)] = 1
	data[np.logical_and(probability<threshold_min, data!=-1)] = 0
	# Unassign any "behavior" bouts that are above the max threshold
	# Note: This should only be used for things like searching for low probability bouts
	if threshold_max < 1:
		data[np.logical_and(probability>threshold_max, data!=-1)] = 0
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
	if len(rle_data)>0:
		rle_data = pd.concat(rle_data).reset_index(drop=True)
	# If no animals exist, just fill some junk data
	else:
		rle_data = pd.DataFrame({'animal_idx':[0], 'start':[0], 'duration':[0], 'is_behavior':[-1]})
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


def read_activity_folder(folder: os.path, activity_threshold: float, interpolate_size: int, stitch_bouts: int, filter_bouts: int, smooth: int = 0, forced_pose_v: int = None, linking_dict: dict = None, activity_dict: dict = {}):
	"""Reads in all activity data for a given project folder.

	Args:
		folder: Project folder to read in data. Must be in a standard project format. See `get_poses_in_folder` for more details.
		activity_threshold: Threshold in cm/s for calling an "active" bout
		interpolate_size: Maximum frame gap to interpolate missing behavior calls (no pose)
		stitch_bouts: Maximum frame gap of not-active to merge adjacent active calls
		filter_bouts: Minimum frame duration of an active bout to be considered real
		smooth: Smoothing window on calculation of distance travelled. 0 = no smoothing
		forced_pose_v: Force activity to be calculated based on a specific pose version. v2-5 uses keypoint centroids. v6+ uses segmentation centroid.
		linking_dict: Dictonary of cached identities from `link_identities`
		activity_dict: Dictionary of cached activity data from `read_activity_folder`
	"""
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
			try:
				cur_activity = parse_activity(cur_file, smooth=smooth, forced_pose_v=forced_pose_v)
			# Skip any files that raise an error
			except (NotImplementedError, ValueError):
				continue
			activity_dict[cur_file] = cur_activity
		# Convert the raw activity into activity bouts
		# Threshold the activity data
		activity_behavior = (cur_activity > activity_threshold).astype(np.int64)
		activity_behavior[cur_activity < 0] = -1
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
			tmp_df = pd.DataFrame({'animal_idx': idx, 'start': cur_starts, 'duration': cur_durations, 'is_behavior': cur_values})
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
	all_predictions['longterm_idx'] = [linking_dict[x][y] if x in linking_dict.keys() and y in linking_dict[x].keys() else -1 for x, y in zip(all_predictions['video_name'].values, all_predictions['animal_idx'])]
	return all_predictions, linking_dict, activity_dict


def parse_activity(pose_file: os.path, smooth: int = 0, forced_pose_v: int = None):
	"""Reads in activity data for a single file.

	Args:
		pose_file: file to read in activity from
		smooth: smoothing window
		forced_pose_v: optional pose version to downgrade to
	"""
	pose_v = putils.get_pose_v(pose_file)
	if forced_pose_v is not None:
		assert pose_v >= forced_pose_v
		pose_v = forced_pose_v

	with h5py.File(pose_file, 'r') as f:
		if 'cm_per_pixel' in f['poseest'].attrs:
			cm_per_px = f['poseest'].attrs['cm_per_pixel']
		else:
			raise ValueError('Activity cannot be calculated without cm_per_px value')
	# Read in the data to produce a center_data variable which is of shape [n_frame, n_animal, 2] containing all [x,y] centers
	if pose_v < 6:
		pose_data = read_pose_file(pose_file, forced_pose_v)
		if pose_data.shape[1] == 0:
			raise ValueError('Pose file didn\'t have any animals to calculate distance')
		center_data = np.zeros([pose_data.shape[0], pose_data.shape[1], 2])
		for frame in np.arange(len(pose_data)):
			for cur_animal in range(pose_data.shape[1]):
				# Ignore tail points for convex hull/center
				center_data[frame, cur_animal, :] = get_pose_center(pose_data[frame, cur_animal, :10, :])
	elif pose_v == 6:
		raise NotImplementedError('Pose v6 distance not supported yet.')
	else:
		raise NotImplementedError('Pose version not detected for ' + str(pose_file) + ' (' + str(pose_v) + '). Cannot interpret data.')
	# Mask out missing data before calculating gradients (distances)
	center_data = np.ma.array(center_data, mask=np.tile(np.expand_dims(np.all(center_data == 0, axis=2), axis=-1), [1, 1, 2]))
	dists = np.gradient(center_data, axis=0)
	dists = np.hypot(dists[:, :, 0], dists[:, :, 1])
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
			smoothed_dists.append(scipy.signal.fftconvolve(dists[:, animal_idx], np.ones([smooth])/ smooth, mode='same'))
		dists = np.stack(smoothed_dists, axis=1)
	# Finally convert to pixel space
	dists = dists * cm_per_px
	# Fix any negative values
	dists[dists_mask] = -1
	# Return the filled matrix transposed (distances stored as animal_id x frame)
	return dists.T


def read_pose_file(pose_file: os.path, force_version: int=None) -> np.ndarray:
	"""Reads pose data from a pose file.

	Args:
		pose_file: The pose file to read
		force_version: Ignore the version inside the pose file and retrieve the older version data. Forced version must be < version of file.

	Returns:
		numpy array of the pose data in shape [frame, animal, keypoint, 2]. Keypoints are x,y and are in pixel units

		Animal data returned is sorted by identity. Data without predictions stores 0,0 keypoints. Pose_v3 gets collapsed into pseudo-identities using the same approach that JABS uses.
	"""
	with h5py.File(pose_file, 'r') as f:
		if 'version' in f['poseest'].attrs:
			detected_pose_version = f['poseest'].attrs['version'][0]
		else:
			detected_pose_version = 2
	if force_version:
		if force_version == 2 and detected_pose_version != 2:
			raise ValueError('Multi mouse files cannot be downgraded to single mouse pose for reading purposes.')
		elif force_version > detected_pose_version:
			raise ValueError(f'Reading version can only be a downgrade: file version: {detected_pose_version}, request version: {force_version}')
		else:
			detected_pose_version = force_version

	# Single mouse poses
	if detected_pose_version == 2:
		with h5py.File(pose_file, 'r') as f:
			pose_data = f['poseest/points'][:]
		return np.flip(np.expand_dims(pose_data, axis=1), axis=-1)

	# Multi mouse poses
	if detected_pose_version == 3:
		id_key = 'poseest/instance_track_id'
		raise NotImplementedError('pose_v3 reading not implemented yet...')
	elif detected_pose_version > 3:
		id_key = 'poseest/instance_embed_id'

	with h5py.File(pose_file, 'r') as f:
		pose_data = f['poseest/points'][:]
		animal_ids = f[id_key][:]
		if detected_pose_version > 3:
			max_valid_id = f['poseest/instance_id_center'].shape[0]
			animal_ids[animal_ids > max_valid_id] = 0
		id_mask = animal_ids == 0

	# Sort by identity into a full matrix
	sorted_pose_data = np.zeros([pose_data.shape[0], np.max(animal_ids), 12, 2], dtype=pose_data.dtype)
	sorted_pose_data[np.where(id_mask == 0)[0], animal_ids[id_mask == 0] - 1, :, :] = pose_data[id_mask == 0, :, :]
	return np.flip(sorted_pose_data, axis=-1)


def get_pose_center(pose: np.array) -> np.ndarray:
	"""Returns a center of pose using the centroid of a convex hull.

	Args:
		pose: a pose for a single mouse in a single frame. shape of [12, 2]. Any keypoints at location 0,0 are assumed "no prediction
	"""
	tmp_pose = pose[np.all(pose != 0, axis=1), :]
	if len(tmp_pose) > 3:
		convex_hull = cv2.convexHull(tmp_pose.astype(np.float32))
		moments = cv2.moments(convex_hull)
		# If the area is 0, return a default value
		if moments['m00'] == 0:
			return np.array([0,0])
		return np.array([moments['m10']/moments['m00'], moments['m01']/moments['m00']])
	# If we have 1-2 keypoints, just average them.
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
		# Check if there are behavior predictions and read in data appropriately
		prediction_file = putils.pose_to_prediction(cur_file, behavior)
		if os.path.exists(prediction_file):
			predictions = parse_predictions(prediction_file, interpolate_size=interpolate_size, stitch_bouts=stitch_bouts, filter_bouts=filter_bouts)
			# Add in distance traveled during bouts, if available
			if cur_file in activity_dict.keys():
				predictions['distance'] = 0.0
				for idx in np.unique(predictions['animal_idx']):
					if idx != -1:
						rows_to_assign = predictions['animal_idx']==idx
						predictions.loc[rows_to_assign,'distance'] = get_bout_dists(predictions.loc[rows_to_assign,'start'], predictions.loc[rows_to_assign,'duration'], activity_dict[cur_file][idx])
			else:
				pass
		else:
			predictions = make_no_predictions(cur_file)
		# Toss data into the full matrix
		# Extract date from the name
		try:
			# TODO: update date format to also handle just a date (no time)
			time_str = re.search(date_format, video_name).group()
			formatted_time = str(datetime.strptime(time_str, '%Y-%m-%d_%H-%M-%S'))
			predictions['time'] = formatted_time
		except:
			predictions['time'] = 'NA'
		predictions['exp_prefix'] = video_prefix
		predictions['video_name'] = video_name
		all_predictions.append(predictions)
	all_predictions = pd.concat(all_predictions).reset_index(drop=True)
	# Correct for identities across videos
	if np.all(all_predictions['time']!='NA'):
		if linking_dict is None:
			linking_dict = link_identities(folder)
		all_predictions['longterm_idx'] = [linking_dict[x][y] if x in linking_dict.keys() and y in linking_dict[x].keys() else -1 for x,y in zip(all_predictions['video_name'].values, all_predictions['animal_idx'])]
	else:
		all_predictions['longterm_idx'] = all_predictions['animal_idx']
		linking_dict = {}
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

class VideoTracklet():
	"""A collection of video observations for an individual."""
	def __init__(self, track_id: Union[int, List[int]], embeds: List[np.ndarray] = []):
		"""Initializes a video tracklet.

		Args:
			track_id: track of list of tracks in this tracklet
			embeds: center data for the track of shape [n_videos, embed_dim]
		"""
		self._track_id = track_id if isinstance(track_id, list) else [track_id]
		assert len(embeds.shape) == 2
		self._embeddings = embeds

	@property
	def track_id(self):
		"""List of ids contained in this tracklet."""
		return self._track_id

	@property
	def embeddings(self):
		"""Embedding data for this tracklet."""
		return self._embeddings

	@classmethod
	def from_tracklets(cls, tracklets: List[VideoTracklet]):
		"""Builds a new tracklet from 1 or more tracklets.

		Args:
			tracklets:
				List of VideoTracklet objects to join together
		"""
		all_track_ids = list(chain.from_iterable([x.track_id for x in tracklets]))
		all_embeddings = np.concatenate([x.embeddings for x in tracklets])
		return cls(all_track_ids, all_embeddings)

	@staticmethod
	def cosine_distance(first: VideoTracklet, second: VideoTracklet, default_val: float=np.nan):
		"""Calculates the cosine distance between two tracklets.

		Args:
			first: first tracklet to compare
			second: second tracklet to compare
			default_val: value supplied if distance invalid

		Returns:
			Returns the smallest cosine distance between the two tracklets.
			Since tracklets can contain multiple centers, any pair can result in this minimum.
			If either tracklet has invalid embedding data, default_val.
		"""
		embeds_1 = first.embeddings
		embeds_2 = second.embeddings

		if embeds_1.shape[0] == 0 or embeds_2.shape[0] == 0:
			return default_val

		distance = scipy.spatial.distance.cdist(embeds_1, embeds_2, metric='cosine')
		return np.min(distance)

	def compare_to(self, other: VideoTracklet):
		"""Compares this tracklet with another."""
		return self.cosine_distance(self, other)

class Fragment():
	"""A collection of tracklets that overlap in time."""
	def __init__(self, tracklets: List[VideoTracklet]):
		"""Initializes a fragment object.

		Args:
			tracklets: List of tracklets
		"""
		self._tracklets = tracklets
		self._separation = self._calculate_mean_separation()

	@property
	def separation(self):
		"""Average separation between tracklets."""
		return self._separation

	def _calculate_mean_separation(self):
		"""Calculates the mean cosine distance between contained tracklets."""
		separations = []
		for i in np.arange(len(self._tracklets) - 1):
			for j in np.arange(len(self._tracklets) - 1 - i) + i + 1:
				distance = VideoTracklet.cosine_distance(self._tracklets[i], self._tracklets[j])
				if np.isnan(distance):
					separations.append(0.0)
				else:
					separations.append(distance)
		return np.mean(separations)

	def hungarian_match_ids(self, other: Fragment):
		"""Matches this fragment with another through hungarian matching."""
		distance_cost = np.zeros([len(self._tracklets), len(other._tracklets)], dtype=np.float64)
		for i in np.arange(len(self._tracklets)):
			for j in np.arange(len(other._tracklets)):
				distance_cost[i, j] = VideoTracklet.cosine_distance(self._tracklets[i], other._tracklets[j])
		row_best, col_best = scipy.optimize.linear_sum_assignment(distance_cost)
		return row_best, col_best


# Generates a dictionary of dictionaries to link identities between files
# First layer of dictionaries is the file being translated
# Second layer of dictionaries contains the key of input identity and the value of the identity linked across files
def link_identities(folder: os.path, check_model: bool=False):
	files_in_experiment = putils.get_poses_in_folder(folder)
	vid_names = [putils.pose_to_video(x) for x in files_in_experiment]
	# Read in all the center data
	fragments = []
	identified_model = None
	for cur_file in files_in_experiment:
		cur_centers, cur_model = read_pose_ids(cur_file)
		# Check that new data conforms to the name of the previous model
		if identified_model is None:
			identified_model = cur_model
		elif check_model:
			assert identified_model == cur_model
		tracklets = [VideoTracklet(i, cur_centers[i, :].reshape([1, -1])) for i in range(len(cur_centers))]
		new_fragment = Fragment(tracklets)
		fragments.append(new_fragment)
	# Sort by best fragment and start matching from there
	fragment_qualities = np.asarray([x.separation for x in fragments])
	fragment_lengths = np.asarray([len(x._tracklets) for x in fragments])
	# Adjust qualities such that fragments of the correct number are matched first...
	# TODO:
	# Is the median the correct operation here, or should we allow the user to choose?
	fragment_qualities[fragment_lengths != np.ceil(np.median(fragment_lengths))] -= 1
	sorting_order = np.argsort(-fragment_qualities)
	# Seed first value
	best_fragment_idx = np.where(sorting_order == 0)
	cur_fragment = fragments[best_fragment_idx[0][0]]
	vid_dict = {vid_names[best_fragment_idx[0][0]]: {x: x for x in range(len(cur_fragment._tracklets))}}
	for match_count in np.arange(len(fragment_qualities) - 1) + 1:
		next_fragment_index = np.where(sorting_order == match_count)
		next_fragment = fragments[next_fragment_index[0][0]]
		hungarian_match = Fragment.hungarian_match_ids(cur_fragment, next_fragment)
		vid_dict[vid_names[next_fragment_index[0][0]]] = {new_id: old_id for old_id, new_id in zip(*hungarian_match)}
		# Warning. We add tracklets in order based on current fragment
		# This will preserve tracklets in cur_fragment, but discard unmatched in next_fragment.
		new_tracklets = []
		for i in range(len(cur_fragment._tracklets)):
			if i in hungarian_match[0]:
				match_idx = hungarian_match[0] == i
				new_tracklets.append(VideoTracklet.from_tracklets([cur_fragment._tracklets[hungarian_match[0][match_idx][0]], next_fragment._tracklets[hungarian_match[1][match_idx][0]]]))
			else:
				new_tracklets.append(cur_fragment._tracklets[i])
		cur_fragment = Fragment(new_tracklets)

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
