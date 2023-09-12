import imageio
import h5py
import os
import numpy as np

# Approximately the bottom middle of an 800x800 frame
# TODO: Have this location float based on the input video (currently will fail on a 480x480 video)
behavior_indicator_idxs = (slice(750,775), slice(350,450), slice(None))

# Basic helper function to write out video clip data
# If behavior_idxs is provided, it writes a 2nd video with a a behavior status bar on the bottom
def write_video_clip(in_vid_f, out_vid_f, clip_idxs, behavior_idxs=None):
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
		except:
			break
		out_vid.append_data(next_frame)
		if out_behavior_vid is not None:
			# Behavior is currently active
			if np.isin(idx, behavior_idxs):
				next_frame[behavior_indicator_idxs] = (77,175,74) # green
			# Behavior is not active
			else:
				next_frame[behavior_indicator_idxs] = (152,78,163) # purple
			out_behavior_vid.append_data(next_frame)
	in_vid.close()
	out_vid.close()
	if out_behavior_vid is not None:
		out_behavior_vid.close()

# Function that reads in and clips a pose file
def write_pose_clip(in_pose_f, out_pose_f, clip_idxs):
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
		print('Warning: Overwriting pose file: ' + out_pose_f)
	with h5py.File(out_pose_f, 'w') as out_f:
		for key, data in all_data.items():
			if all_compression_flags[key] is None:
				out_f.create_dataset(key, data=data)
			else:
				out_f.create_dataset(key, data=data, compression='gzip', compression_opts=all_compression_flags[key])
		for key, attrs in all_attrs.items():
			for cur_attr, data in attrs.items():
				out_f[key].attrs.create(cur_attr, data)
