# Utilities related to clipping and rendering data

import imageio
import h5py
import cv2
import os
import numpy as np
from typing import Optional, Union, Tuple, List
from pathlib import Path
import warnings

# Approximately the bottom middle of an 800x800 frame
# TODO: Have this location float based on the input video (currently will fail on a 480x480 video)
behavior_indicator_idxs = (slice(750, 775), slice(350, 450), slice(None))


def write_video_clip(in_vid_f: Union[str, Path], out_vid_f: Union[str, Path], clip_idxs: Union[List, np.ndarray], behavior_idxs: Union[List, np.ndarray] = None):
	"""Writes a clip of a video.

	Args:
		in_vid_f: Input video filename
		out_vid_f: Output video filename
		clip_idxs: List or array of frame indices to place in the clipped video. Frames not present in the video will be ignored without warnings. Must be castable to int.
		behavior_idxs: (Optional) If provided, will render a second video with a behavior indicator. Must be same length as clip_idxs.
	"""
	if behavior_idxs:
		assert len(behavior_idxs) == len(clip_idxs)

	in_vid = imageio.get_reader(in_vid_f)
	out_vid = imageio.get_writer(out_vid_f, fps=30, codec='mpeg4', quality=10)
	out_behavior_vid = None
	if behavior_idxs is not None:
		behavior_vid_f = os.path.splitext(out_vid_f)[0] + '_behavior.avi'
		# Don't overwrite the video if it already exists
		if os.path.exists(behavior_vid_f):
			print('Not overwriting behavior video: ' + behavior_vid_f)
		else:
			out_behavior_vid = imageio.get_writer(behavior_vid_f, fps=30, codec='mpeg4', quality=10)
	# Copy the frames from the input into the output
	for idx in clip_idxs:
		# Test to see if the video frame exists to read
		try:
			next_frame = in_vid.get_data(int(idx))
		except IndexError:
			continue
		out_vid.append_data(next_frame)
		if out_behavior_vid is not None:
			# Behavior is currently active
			if np.isin(idx, behavior_idxs):
				next_frame[behavior_indicator_idxs] = (77, 175, 74)  # green
			# Behavior is not active
			else:
				next_frame[behavior_indicator_idxs] = (152, 78, 163)  # purple
			out_behavior_vid.append_data(next_frame)
	in_vid.close()
	out_vid.close()
	if out_behavior_vid is not None:
		out_behavior_vid.close()


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


object_colors = {
	'lixit': (55, 126, 184),		# Blue
	'food_hopper': (255, 127, 0), 	# Orange
	'corners': (75, 175, 74),		# Green
}

flipped_objects = [
	'lixit',
	'food_hopper',
]


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
		if obj_name in flipped_objects:
			keypoints = np.flip(obj_kpts, axis=-1)
		else:
			keypoints = obj_kpts
		if obj_name in object_colors.keys():
			frame = render_object(frame, keypoints, object_colors[obj_name])
		else:
			frame = render_object(frame, keypoints, (255, 255, 255))
	imageio.imwrite(out_fname, frame)
