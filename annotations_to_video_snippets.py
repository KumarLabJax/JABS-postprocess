import json
import sys
import create_video_snippets
import argparse
import os
import pandas as pd
from analysis_utils.clip_utils import write_video_clip
from typing import Union
from pathlib import Path

def json_to_dict(file, output, video, behavior):
    """
    Args:

    """
    f = open(file)
    bouts = pd.read_json(f)
    bouts = bouts["labels"].values

    for i in range(len(bouts)):
        for j in range(len(bouts[i][behavior])):
            start_val = list(bouts[i][behavior][j].values())[0]
            end_val = list(bouts[i][behavior][j].values())[1]
            clip_name = Path(output).stem + "_Mouse_" + str(i) + "_Bout_" + str(j) + ".avi"
            write_video_clip(video,clip_name,range(start_val,end_val))

def get_time_in_seconds(location: Union[float, int], unit: str, fps: int = 30) -> float:
	"""Converts start and end in arbitrary units into seconds.

	Args:
		location: Starting location
		unit: Units of start and end. Choices of frames, seconds, minutes, hours. Allows shortened versions of choices.
		fps: Frames per second used in calculation

	Returns:
		The requested time in frames
	"""
	unit_char = unit[0]
	if unit_char == 'f':
		return int(location/fps)
	elif unit_char == 's':
		return int(location)
	elif unit_char == 'm':
		return int(location/fps * 60)
	elif unit_char == 'h':
		return int(location/fps * 60 * 60)
	else:
		raise NotImplementedError(f'{unit} is unsupported. Pick from [frame, second, minute, hour].')


def jabs_to_activity_net_json(file, output, video, behavior):
    f = open(file)
    bouts = pd.read_json(f)

    for i in range(len(bouts)):
        for j in range(len(bouts[i][behavior])):
            start_val = list(bouts[i][behavior][j].values())[0]
            end_val = list(bouts[i][behavior][j].values())[1]
            clip_name = Path(output).stem + "_Mouse_" + str(i) + "_Bout_" + str(j)
            get_time_in_seconds(start_val)
            get_time_in_seconds(end_val)
            video_metadata = { 
                  clip_name : {
                        "duration_second": 211.53,
                        "duration_frame": 6337,
                        "annotations": [
                            {
                                "segment": [
                                30.025882995319815,
                                205.2318595943838
                                ],
                                "label": "Rock climbing"
                            }
                        ],
                        "feature_frame": 6336,
                        "fps": 30.0,
                        "rfps": 29.9579255898
                    }
                }

def main(argv):
    parser = argparse.ArgumentParser(description="Formats JSON into data for create video snippets")
    parser.add_argument('--input_file', help='Path to JSON file')
    parser.add_argument('--output_folder', help='Name of output video')
    parser.add_argument('--input_video', help='Path to video that is being snipped')
    parser.add_argument('--behavior', help='Name of behavior being classified')
    args = parser.parse_args()
    assert os.path.exists(args.input_file)
    json_to_dict(args.input_file, args.output_folder, args.input_video, args.behavior)

if __name__ == '__main__':
    main(sys.argv[1:])