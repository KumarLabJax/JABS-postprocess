"""Utilities related to clipping and rendering data."""

import imageio
import h5py
import cv2
import os
import numpy as np
from typing import Optional, Union, Tuple, List
from pathlib import Path
import warnings

BEHAVIOR_COLORS = [
	(77, 175, 74),   # green
	(152, 78, 163),  # purple
]
BEHAVIOR_BLOCK_SIZE = 25
MOUSE_POSE_LINES = [
	# spine
	[0, 3],
	[3, 6],
	[6, 9],
	[9, 10],
	[10, 11],
	# ears
	[0, 1],
	[0, 2],
	# front paws
	[6, 4],
	[6, 5],
	# rear paws
	[9, 7],
	[9, 8],
]
OBJECT_COLORS = {
	'lixit': (55, 126, 184),		# Blue
	'food_hopper': (255, 127, 0), 	# Orange
	'corners': (75, 175, 74),		# Green
}
FLIPPED_OBJECTS = [
	'lixit',
	'food_hopper',
]


def write_video_clip(in_vid_f: Union[str, Path], out_vid_f: Union[str, Path], clip_idxs: Union[List, np.ndarray], behavior_data: Union[List, np.ndarray] = None, pose_data: Optional[np.ndarray] = None):
	"""Writes a clip of a video.

	Args:
		in_vid_f: Input video filename.
		out_vid_f: Output video filename.
		clip_idxs: List or array of frame indices to place in the clipped video. Frames not present in the video will be ignored without warnings. Must be castable to int.
		behavior_data: (Optional) If provided, will render a behavior indicator on the video. Must be same length as clip_idxs. If poses is also provided, must be of shape [frame, animal].
		pose_data: (Optional) Pose to render on the video. Must be same length as clip_idxs. Must be of shape [frame, animal, 12, 2].
	"""
	if behavior_data is not None:
		assert len(behavior_data) == len(clip_idxs)
	if pose_data is not None:
		assert len(pose_data) == len(clip_idxs)
		assert pose_data.ndim == 4
		assert pose_data.shape[2] == 12
		assert pose_data.shape[3] == 2
		if behavior_data is not None:
			assert pose_data.shape[1] == behavior_data.shape[1]

	in_vid = imageio.get_reader(in_vid_f)
	out_vid = imageio.get_writer(out_vid_f, fps=30, codec='mpeg4', quality=10)
	# Copy the frames from the input into the output
	for i, frame in enumerate(clip_idxs):
		try:
			next_frame = in_vid.get_data(int(frame))
		except IndexError:
			continue
		if behavior_data is not None and pose_data is not None:
			for cur_animal in range(pose_data.shape[1]):
				pose_color = BEHAVIOR_COLORS[behavior_data[i, cur_animal] > 0]
				next_frame = render_pose(next_frame, pose_data[i, cur_animal], pose_color)
		if behavior_data is not None and not (pose_data is not None):
			behavior_color = BEHAVIOR_COLORS[np.any(behavior_data[i] > 0)]
			out_location = np.array(next_frame.shape[:2]) - BEHAVIOR_BLOCK_SIZE
			out_location[0] /= 2
			next_frame = cv2.circle(next_frame, tuple(out_location), BEHAVIOR_BLOCK_SIZE, behavior_color, -1)
		out_vid.append_data(next_frame)
	in_vid.close()
	out_vid.close()


def write_pose_clip(in_pose_f: Union[str, Path], out_pose_f: Union[str, Path], clip_idxs: Union[List, np.ndarray]):
	"""Writes a clip of a pose file.

	Args:
		in_pose_f: Input video filename
		out_pose_f: Output video filename
		clip_idxs: List or array of frame indices to place in the clipped video. Frames not present in the video will be ignored without warnings. Must be castable to int.
	"""
	# Extract the data that may have frames as the first dimension
	all_data = {}
	all_attrs = {}
	all_compression_flags = {}
	with h5py.File(in_pose_f, 'r') as in_f:
		all_pose_fields = ['poseest/' + key for key in in_f['poseest'].keys()]
		if 'static_objects' in in_f.keys():
			all_static_fields = ['static_objects/' + key for key in in_f['static_objects'].keys()]
		else:
			all_static_fields = []
		# Warning: If number of frames is equal to number of animals in id_centers, the centers will be cropped as well
		# However, this should future-proof the function to not depend on the pose version as much by auto-detecting all fields and copying them
		frame_len = in_f['poseest/points'].shape[0]
		# Adjust the clip_idxs to safely fall within the available data
		adjusted_clip_idxs = np.array(clip_idxs)[np.isin(clip_idxs, np.arange(frame_len))]
		# Cycle over all the available datasets
		for key in np.concatenate([all_pose_fields, all_static_fields]):
			# Clip data that has the shape
			if in_f[key].shape[0] == frame_len:
				all_data[key] = in_f[key][adjusted_clip_idxs]
				if len(in_f[key].attrs.keys()) > 0:
					all_attrs[key] = dict(in_f[key].attrs.items())
			# Just copy other stuff as-is
			else:
				all_data[key] = in_f[key][:]
				if len(in_f[key].attrs.keys()) > 0:
					all_attrs[key] = dict(in_f[key].attrs.items())
			all_compression_flags[key] = in_f[key].compression_opts
		all_attrs['poseest'] = dict(in_f['poseest'].attrs.items())
	# Write the data out
	if os.path.exists(out_pose_f):
		warnings.warn(f'Warning: Overwriting pose file: {out_pose_f}')
	with h5py.File(out_pose_f, 'w') as out_f:
		for key, data in all_data.items():
			if all_compression_flags[key] is None:
				out_f.create_dataset(key, data=data)
			else:
				out_f.create_dataset(key, data=data, compression='gzip', compression_opts=all_compression_flags[key])
		for key, attrs in all_attrs.items():
			for cur_attr, data in attrs.items():
				out_f[key].attrs.create(cur_attr, data)


def read_pose_file(pose_file: os.path, force_version: int = None) -> np.ndarray:
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


def render_pose(frame: np.ndarray[np.uint8], pose_kpts: np.ndarray, color: Tuple[np.uint8, np.uint8, np.uint8]) -> np.ndarray:
	"""Renders a single 12-keypoint mouse pose onto a frame.

	Args:
		frame: The frame to render the pose on
		pose_kpts: An array of shape [12, 2] for the mouse pose. Keypoint pairs that are (0,0) are considered not present and will not be rendered.
		color: The color to render the pose as
	"""
	assert pose_kpts.ndim == 2
	assert pose_kpts.shape == (12, 2)
	assert frame.ndim == 3
	keypoints_int = pose_kpts.astype(np.int64)
	out_frame = np.copy(frame)
	# Render the keypoints
	for i in range(12):
		kp = [keypoints_int[i, 0], keypoints_int[i, 1]]
		if np.any(np.asarray(kp) != 0):
			out_frame = cv2.circle(out_frame, tuple(kp), 3, color, -1)
	# Render the lines
	for connection in MOUSE_POSE_LINES:
		kp1 = [keypoints_int[connection[0], 0], keypoints_int[connection[0], 1]]
		kp2 = [keypoints_int[connection[1], 0], keypoints_int[connection[1], 1]]
		# Don't render the line if either keypoints don't exist
		if np.any(np.asarray(kp1) != 0) and np.any(np.asarray(kp2) != 0):
			out_frame = cv2.line(out_frame, tuple(kp1), tuple(kp2), color, 2, cv2.LINE_AA)
	return out_frame


def render_object(frame: np.ndarray[np.uint8], object_kpts: np.ndarray, color: Tuple[np.uint8, np.uint8, np.uint8]):
	"""Render a single object onto a frame.

	Args:
		frame: The frame to render the object on
		object_kpts: An array of keypoints of shape [n_kpts, 2] where the second dimension is (x,y). Can be any numeric dtype
		color: The color to render the object as
	"""
	assert object_kpts.ndim == 2
	assert frame.ndim == 3
	keypoints_int = object_kpts.astype(np.int64)
	out_frame = np.copy(frame)
	for i in range(keypoints_int.shape[0]):
		out_frame = cv2.circle(out_frame, (keypoints_int[i, 0], keypoints_int[i, 1]), 5, color)
		if i < keypoints_int.shape[0] - 1:
			out_frame = cv2.line(out_frame, (keypoints_int[i, 0], keypoints_int[i, 1]), (keypoints_int[i + 1, 0], keypoints_int[i + 1, 1]), color)
	return out_frame


def render_static_objects(video: str, pose: str, out_png: Optional[str] = None, frame_idx: int = 0):
	"""Render all the static objects from a pose file onto a video.

	Args:
		video: filename of the video to retrieve a frame
		pose: pose file to retrieve the static object data
		out_png: (optional) output filename for the render. Defaults to <video>_static_objects.png
		frame_idx: frame from the video to render. Defaults to 
	"""
	if out_png:
		out_fname = out_png
	else:
		out_fname = os.path.splitext(video)[0] + '_static_objects.png'
	vid_reader = imageio.get_reader(video)
	frame = vid_reader.get_data(frame_idx)
	static_obj_data = {}
	with h5py.File(pose, 'r') as f:
		if 'static_objects' not in f.keys():
			raise ValueError(f'Static objects not present in {pose}')
		obj_grp = f['static_objects']
		keyed_objects = obj_grp.keys()
		for cur_obj in keyed_objects:
			static_obj_data[cur_obj] = obj_grp[cur_obj][:]
	for obj_name, obj_kpts in static_obj_data.items():
		if obj_name in FLIPPED_OBJECTS:
			keypoints = np.flip(obj_kpts, axis=-1)
		else:
			keypoints = obj_kpts
		if obj_name in OBJECT_COLORS.keys():
			frame = render_object(frame, keypoints, OBJECT_COLORS[obj_name])
		else:
			frame = render_object(frame, keypoints, (255, 255, 255))
	imageio.imwrite(out_fname, frame)
