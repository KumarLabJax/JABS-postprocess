import numpy as np
import h5py
import argparse
import sys
import os
from analysis_utils.clip_utils import write_video_clip, write_pose_clip, read_pose_file
from typing import Union


def get_time_in_frames(location: Union[float, int], unit: str, fps: int = 30) -> int:
	"""Converts start and end in arbitrary units into frames.

	Args:
		location: Starting location
		unit: Units of start and end. Choices of frames, seconds, minutes, hours. Allows shortened versions of choices.
		fps: Frames per second used in calculation

	Returns:
		The requested time in frames
	"""
	unit_char = unit[0]
	if unit_char == 'f':
		return int(location)
	elif unit_char == 's':
		return int(location * fps)
	elif unit_char == 'm':
		return int(location * fps * 60)
	elif unit_char == 'h':
		return int(location * fps * 60 * 60)
	else:
		raise NotImplementedError(f'{unit} is unsupported. Pick from [frame, second, minute, hour].')


def main(argv):
	parser = argparse.ArgumentParser(description='Clips the video with optional behavior rendering')
	parser.add_argument('--input_video', help='Path to input video for clipping', required=True)
	parser.add_argument('--output_video', help='Path to output clipped video', required=True)
	parser.add_argument('--start', help='Start time of the clip to produce (default beginning of video)', default=0, type=float)
	g1 = parser.add_mutually_exclusive_group()
	g1.add_argument('--end', help='End time of the clip to produce (default full video)', type=float)
	g1.add_argument('--duration', help='Duration of the clip to produce', type=float)
	parser.set_defaults(end=-1)
	parser.add_argument('--time_units', help='Units used when clipping (default second)', choices=['frame', 'frames', 'f', 'second', 'seconds', 's', 'minute', 'minutes', 'm', 'hour', 'hours', 'h'], default='s', type=str)
	parser.add_argument('--pose_file', help='Optional path to input pose file. Required to clip pose and render pose.', default=None)
	parser.add_argument('--out_pose', help='Write the clipped pose file as well.', default=None)
	parser.add_argument('--render_pose', help='Render the pose on the video clip.', default=False, action='store_true')
	parser.add_argument('--behavior_file', help='Optional path to behavior predictions. If provided, will render predictions on the video.', default=None)
	parser.add_argument('--overwrite', '-o', help='Overwrite the output video if it already exists', default=False, action='store_true')
	args = parser.parse_args()

	assert os.path.exists(args.input_video)
	if os.path.exists(args.output_video) and not args.overwrite:
		raise FileExistsError(f'{args.output_video} exists. Use --overwrite if you wish to overwrite it.')

	start_frame = get_time_in_frames(args.start, args.time_units)

	if args.duration is None:
		if args.end == -1:
			# If we want the whole video, we need to get a hint at how big it is
			# TODO: This is probably overkill and will slow down the clip function...
			end_frame = sys.maxsize
		else:
			end_frame = get_time_in_frames(args.end, args.time_units)
	else:
		end_frame = get_time_in_frames(args.start + args.duration, args.time_units)

	behavior_data = None
	if args.behavior_file:
		with h5py.File(args.behavior_file, 'r') as f:
			max_frames = f['predictions/predicted_class'].shape[1]
			end_frame = np.clip(end_frame, 0, max_frames - 1)
			behavior_data = f['predictions/predicted_class'][:, start_frame:end_frame]
			# behavior data is stored as [animal, frame]
			behavior_data = np.transpose(behavior_data)

	pose_data = None
	if args.pose_file:
		pose_data = read_pose_file(args.pose_file)
		max_frames = pose_data.shape[0] + 1
		end_frame = np.clip(end_frame, 0, max_frames - 1)
		pose_data = pose_data[start_frame:end_frame]
	
	pose_for_video = pose_data if args.render_pose else None
	write_video_clip(args.input_video, args.output_video, range(start_frame, end_frame), behavior_data, pose_for_video)
	if args.out_pose and args.pose_file is not None:
		write_pose_clip(args.pose_file, args.out_pose, range(start_frame, end_frame))


if __name__ == '__main__':
	main(sys.argv[1:])
