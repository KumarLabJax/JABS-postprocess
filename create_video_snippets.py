import pandas as pd
import numpy as np
import argparse, sys
from analysis_utils.clip_utils import write_video_clip

def main(argv):
	parser = argparse.ArgumentParser(description='Clips the video')
	parser.add_argument('--bout_file', help='Path to the bout file', required=True)
	parser.add_argument('--input_video_folder', help='Path to folder with videos', required=True)
	parser.add_argument('--output_video_folder', help='Path to folder for output videos', required=True)
	parser.add_argument('--pad_length', help='Length of padding to both sides of a bout in frames (default=30)', type=int, default=30)
	parser.add_argument('--longest_bouts', help='Export only the longest n bouts (default is export all bouts)', type=int, default=None)
	parser.add_argument('--shortest_bouts', help='Export only the shortest n bouts (default is export all bouts)', type=int, default=None)
	parser.add_argument('--overlay_behavior', help='Overlays a marker on the video to indicate when the behavior is occurring', default=False, action='store_true')
	args = parser.parse_args()

	# read in the bout table
	results_bouts = (args.bout_file)
	header_data = pd.read_csv(results_bouts, nrows=1)
	df = pd.read_csv(results_bouts, skiprows=2)

	# subsetting rows only for when there is behavioral bout
	subset_df = df.loc[df['is_behavior']==1, ]
	filter_df_list = []
	if args.longest_bouts is not None:
		filter_df_list.append(subset_df.sort_values('duration', ascending=False).head(args.longest_bouts))
	if args.shortest_bouts is not None:
		filter_df_list.append(subset_df.sort_values('duration').head(args.shortest_bouts))
	# If any of the filters were used, switch to only the filtered bouts.
	if len(filter_df_list)>0:
		subset_df = pd.concat(filter_df_list).drop_duplicates()

	# Export all the video clips requested
	for _,row in subset_df.iterrows():
		full_video_path = (args.input_video_folder) + row['video_name'] + ".avi"
		start_frame = np.clip(row['start'] - (args.pad_length), 0, None)
		# We generate a new video based on the new start frame in the clip
		out_vid_f = (args.output_video_folder) + row['video_name'] + "_" + str(start_frame) + ".avi"
		end_frame = np.clip(row['start'] + row['duration'] + (args.pad_length), 0, None)
		clip_idxs = np.arange(start_frame, end_frame)
		if args.overlay_behavior:
			behavior_idxs = np.zeros(clip_idxs.shape, dtype=np.uint8)
			behavior_idxs[row['start']:row['start']+row['duration']] = 1
			write_video_clip(in_vid_f=full_video_path, out_vid_f=out_vid_f, clip_idxs=clip_idxs, behavior_idxs=behavior_idxs)
		else:
			write_video_clip(in_vid_f=full_video_path, out_vid_f=out_vid_f, clip_idxs=clip_idxs)

if __name__  == '__main__':
	main(sys.argv[1:])