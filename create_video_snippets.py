# testing line
# from types import SimpleNamespace
# args = SimpleNamespace(bout_file = "/projects/kumar-lab/choij/validate-drinking/Results_2023-01-24_Drinking_bouts.csv", input_video_folder = "/projects/kumar-lab/choij/DLC-videos/", output_video_folder = "/projects/kumar-lab/choij/validate-drinking/", pad_length=30)

import pandas as pd
import plotnine as p9
import numpy as np
import imageio
import argparse, sys

def main(argv):
	parser = argparse.ArgumentParser(description='Clips the video')
	parser.add_argument('--bout_file', help='Path to the bout file', required=True)
	parser.add_argument('--input_video_folder', help='Path to folder with videos', required=True)
	parser.add_argument('--output_video_folder', help='Path to folder for output videos', required=True)
	parser.add_argument('--pad_length', help='Length of bout padding in frames', type=int, default=30)
	args = parser.parse_args()

	# read in the bout table
	results_bouts = (args.bout_file)
	header_data = pd.read_csv(results_bouts, nrows=1)
	df = pd.read_csv(results_bouts, skiprows=2)

	# subsetting rows only for when there is behavioral bout
	df = df.loc[df['is_behavior']==1, ]

	for _,row in df.iterrows():
		full_video_path = (args.input_video_folder) + row['video_name'] + ".avi"
		in_vid = imageio.get_reader(full_video_path)
		start_frame = np.clip(row['start'] - (args.pad_length), 0, len(in_vid))
		out_vid = imageio.get_writer((args.output_video_folder) + row['video_name'] + "_" + str(start_frame) + ".avi", fps=30, codec='mpeg4', quality=10)
		end_frame = np.clip(row['start'] + row['duration'] + (args.pad_length), 0, len(in_vid))
		clip_idxs = np.arange(start_frame, end_frame)
		for idx in clip_idxs:
			out_vid.append_data(in_vid.get_data(int(idx)))
		in_vid.close()
		out_vid.close()

if __name__  == '__main__':
	main(sys.argv[1:])