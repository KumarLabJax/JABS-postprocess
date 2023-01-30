import pandas as pd
import numpy as np
import argparse, sys, os, re
from analysis_utils.clip_utils import write_video_clip, write_pose_clip
from jabs_utils.project_utils import get_behaviors_in_folder, get_predictions_in_folder, get_poses_in_folder, pose_to_prediction, pose_to_video
from jabs_utils.read_utils import parse_predictions
from jabs_utils.bout_utils import filter_data, get_arena_bouts

def main(argv):
	parser = argparse.ArgumentParser(description='Scans through a project folder for low predictions for a behavior')
	parser.add_argument('--input_video_folder', help='Path to folder with videos', required=True)
	# Setup the scan mode
	g1 = parser.add_mutually_exclusive_group()
	g1.add_argument('--bouts', help='Generates clips for bouts of behavior', dest='mode', action='store_const', const='bout')
	g1.add_argument('--uncertain', help='Generates clips for uncertainty blocks', dest='mode', action='store_const', const='uncertain')
	parser.set_defaults(mode='bout')
	parser.add_argument('--behavior', help='Behavior to sample for low probabilities', required=True)
	parser.add_argument('--output_video_folder', help='Path to folder for output videos', required=True)
	parser.add_argument('--pad_length', help='Length of padding to both sides of a bout in frames (default=30)', type=int, default=30)
	parser.add_argument('--longest_bouts', help='Export only the longest n bouts (default is export all bouts)', type=int, default=None)
	parser.add_argument('--shortest_bouts', help='Export only the shortest n bouts (default is export all bouts)', type=int, default=None)
	parser.add_argument('--overlay_behavior', help='Overlays a marker on the video to indicate when the behavior is occurring', default=False, action='store_true')
	parser.add_argument('--threshold', help='Threshold for bouts. Default is 0.5. Only used with --bouts', type=float, default=0.5)
	parser.add_argument('--tolerance', help='Tolerance for uncertainty. Default is 0.1 (0.4-0.6 probabilities). Only used with --uncertain', type=float, default=0.1)
	parser.add_argument('--preserve_individual', help='Disables collapsing invidual animal predictions into an arena prediction for sampling.', default=False, action='store_true')
	args = parser.parse_args()

	# Figure out what videos are available and read in their data
	df_list = []
	exp_folders = get_predictions_in_folder(args.input_video_folder)
	for cur_exp in exp_folders:
		files_in_experiment = get_poses_in_folder(cur_exp)
		for cur_pose_file in files_in_experiment:
			prediction_file = pose_to_prediction(cur_pose_file, args.behavior)
			# Read in the necessary bout data
			if args.mode == 'bout':
				predictions = parse_predictions(prediction_file, interpolate_size=args.pad_length, stitch_bouts=args.pad_length, filter_bouts=5)
			elif args.mode == 'uncertain':
				predictions = parse_predictions(prediction_file, threshold_min=0.5-args.tolerance, threshold_max=0.5+args.tolerance, interpolate_size=args.pad_length, stitch_bouts=args.pad_length, filter_bouts=5)
			else:
				raise(NotImplementedError(args.mode + ' not implemented.'))
			# Convert the animal data into arena data
			if not args.preserve_individual:
				# rle data is in format starts, durations, states
				rle_data = get_arena_bouts(predictions['start'],predictions['duration'],predictions['is_behavior'])
				filtered_rle_data = filter_data(rle_data[0], rle_data[1], rle_data[2], max_gap_size=args.pad_length, values_to_remove=[0])
				predictions = pd.DataFrame({'animal_idx':0, 'start':filtered_rle_data[0], 'duration':filtered_rle_data[1], 'is_behavior':filtered_rle_data[2]})
			# Only look at behavior
			predictions = predictions[predictions['is_behavior'] == 1]
			# Add some additional useful fields
			vid_base = pose_to_video(cur_pose_file)
			predictions['full_video_path'] = os.path.dirname(cur_pose_file) + '/' + vid_base + '.avi'
			predictions['full_pose_path'] = cur_pose_file
			predictions['video_name'] = vid_base
			df_list.append(predictions[predictions['is_behavior']==1])
	df = pd.concat(df_list)

	# Filter the bouts as necessary
	filter_df_list = []
	if args.longest_bouts is not None:
		filter_df_list.append(df.sort_values('duration', ascending=False).head(args.longest_bouts))
	if args.shortest_bouts is not None:
		filter_df_list.append(df.sort_values('duration').head(args.shortest_bouts))
	# If any of the filters were used, switch to only the filtered bouts.
	if len(filter_df_list)>0:
		df = pd.concat(filter_df_list).drop_duplicates()

	# Export all the video clips requested
	for _,row in df.iterrows():
		full_video_path = row['full_video_path']
		start_frame = np.clip(row['start'] - (args.pad_length), 0, None)
		# We generate a new video based on the new start frame in the clip
		out_vid_f = args.output_video_folder + row['video_name'] + "_" + str(start_frame) + ".avi"
		end_frame = np.clip(row['start'] + row['duration'] + (args.pad_length), 0, None)
		clip_idxs = np.arange(start_frame, end_frame)
		if args.overlay_behavior:
			behavior_idxs = np.arange(row['start'], row['start']+row['duration'])
			write_video_clip(in_vid_f=full_video_path, out_vid_f=out_vid_f, clip_idxs=clip_idxs, behavior_idxs=behavior_idxs)
		else:
			write_video_clip(in_vid_f=full_video_path, out_vid_f=out_vid_f, clip_idxs=clip_idxs)
		out_pose_f = args.output_video_folder + re.sub('(.*)(_pose_est_v[0-9]+.*)', '\\1_' + str(start_frame) + '\\2', os.path.basename(row['full_pose_path']))
		write_pose_clip(in_pose_f=row['full_pose_path'], out_pose_f=out_pose_f, clip_idxs=clip_idxs)

if __name__  == '__main__':
	main(sys.argv[1:])