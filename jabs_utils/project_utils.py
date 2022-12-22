import glob
import re
import os

BEHAVIOR_CLASSIFY_VERSION = 1
POSE_REGEX_STR = '_pose_est_v[2-5].h5'

# Generates a list of experiment folders in a project
# Assumes that all videos for a multi-day experiment exist in a single folder
# Assumes that the prediction paths look like 'project/EXPERIMENT_FOLDER/video_behavior/v1/behavior_name/video.h5'
def get_predictions_in_folder(folder: os.path):
	# This glob can't use trailing slash
	if folder[-1] == '/':
		folder = folder[:-1]
	# Find all the behavior prediction folders (always named v1)
	possible_folders = glob.glob(folder + '**/v' + str(BEHAVIOR_CLASSIFY_VERSION), recursive=True)
	# Extract the folder 2 above that, which would be the folder containing all experiments in a 4-day grouping
	possible_folders = [re.sub('(.*)([^/]*/){2}v' + str(BEHAVIOR_CLASSIFY_VERSION),'\\1',x) for x in possible_folders]
	experiment_folder_list = list(set(possible_folders))
	return experiment_folder_list

# Generates a list of behavior predictions found in project folder
# Assumes that the prediction paths look like 'project/experiment_folder/video_behavior/v1/BEHAVIOR_NAME/video.h5'
def get_behaviors_in_folder(folder: os.path):
	# This glob requires a trailing slash
	if folder[-1] != '/':
		folder = folder[:-1]
	possible_files = glob.glob(folder + '**/v' + str(BEHAVIOR_CLASSIFY_VERSION) + '/*', recursive=True)
	behaviors = [re.sub('.*/','',x) for x in possible_files]
	behaviors = list(set(behaviors))
	return behaviors

# Detects the pose files available in a folder
# Returns a sorted list of pose files
# TODO: If multiple pose files exist, only pick the newest
def get_poses_in_folder(folder: os.path):
	# This glob can't use trailing slash
	if folder[-1] == '/':
		folder = folder[:-1]
	return sorted(glob.glob(folder + '**/*' + POSE_REGEX_STR))

# Returns a prediction file from a pose filename and behavior
def pose_to_prediction(file: os.path, behavior: str):
	video_name = pose_to_video(file)
	return video_to_prediction(os.path.dirname(file) + '/' + video_name, behavior)

# Returns a prediction file from a video filename and behavior
def video_to_prediction(file: os.path, behavior: str):
	file_no_folder = os.path.basename(file)
	folder = os.path.dirname(file)
	vid_noext, ext = os.path.splitext(file_no_folder)
	return folder + '/' + re.sub('$', '_behavior/v1/' + behavior + '/' + vid_noext + '.h5', vid_noext)

# Returns the video filename from a given pose filename
# Note that this returns ONLY the filename (without directory or extension)
# In some cases, you may want to retain the folder
def pose_to_video(file: os.path):
	file_no_folder = os.path.basename(file)
	return re.sub(POSE_REGEX_STR, '', file_no_folder)
