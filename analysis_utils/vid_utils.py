import imageio
import numpy as np

behavior_indicator_idxs = (slice(750,775), slice(350,450), slice(None))

# Basic helper function to write out video clip data
def write_video_clip(in_vid_f, out_vid_f, clip_idxs, behavior_idxs=None):
	in_vid = imageio.get_reader(in_vid_f)
	out_vid = imageio.get_writer(out_vid_f, fps=30, codec='mpeg4', quality=10)
	# Copy the frames from the input into the output
	for idx in clip_idxs:
		# Test to see if the video frame exists to read
		try:
			next_frame = in_vid.get_data(int(idx))
		except:
			break
		if behavior_idxs is not None:
			# Behavior is currently active
			if np.isin(idx, behavior_idxs):
				next_frame[behavior_indicator_idxs] = (77,175,74) # green
			# Behavior is not active
			else:
				next_frame[behavior_indicator_idxs] = (152,78,163) # purple
		out_vid.append_data(next_frame)
	in_vid.close()
	out_vid.close()