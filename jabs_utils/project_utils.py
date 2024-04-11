import glob
import re
import os

BEHAVIOR_CLASSIFY_VERSION = 1
POSE_REGEX_STR = '_pose_est_v[2-5].h5'


def get_predictions_in_folder(folder: os.path):
	"""Generates a list of experiment folders in a project.

	Args:
		folder: folder to scan for experiment groups

	Returns:
		list of folders containing experimental groups

	Notes:
		Requires that all videos for a single multi-day experiment exists in a single folder.
		The prediction paths should look like '[experiment]/[video]_behavior/v1/[behavior]/[video].h5'.
		This function returns [experiment]
	"""
	# This glob requires a trailing slash
	if folder[-1] != '/':
		folder = folder + '/'
	# Find all the behavior prediction folders (always named v1)
	possible_folders = glob.glob(folder + '**/v' + str(BEHAVIOR_CLASSIFY_VERSION), recursive=True)
	# Extract the folder 2 above that, which would be the folder containing all experiments in a 4-day grouping
	possible_folders = [re.sub('(.*)([^/]*/){2}v' + str(BEHAVIOR_CLASSIFY_VERSION), '\\1', x) for x in possible_folders]
	experiment_folder_list = list(set(possible_folders))
	return experiment_folder_list


def get_behaviors_in_folder(folder: os.path):
	"""Generates a list of behavior predictions found in a project folder.

	Args:
		folder: folder path to search for behavioral predictions

	Returns:
		list of all behavioral predictions for this experiment

	Notes:
		The prediction paths should look like '[experiment]/[video]_behavior/v1/[behavior]/[video].h5'.
		This function returns all [behavior] values present.
	"""
	# This glob requires a trailing slash
	if folder[-1] != '/':
		folder = folder + '/'
	possible_files = glob.glob(folder + '**/v' + str(BEHAVIOR_CLASSIFY_VERSION) + '/*', recursive=True)
	behaviors = [re.sub('.*/', '', x) for x in possible_files]
	behaviors = list(set(behaviors))
	return behaviors


def get_poses_in_folder(folder: os.path):
	"""Detects the pose files available in a folder.

	Args:
		folder: folder containing pose files

	Returns:
		a sorted list of pose files present in the folder

	TODO:
		de-duplicate videos that contain multiple pose files by picking the newest only
	"""
	# This glob requires a trailing slash
	if folder[-1] != '/':
		folder = folder + '/'
	return sorted(glob.glob(folder + '*' + POSE_REGEX_STR))


def pose_to_prediction(file: os.path, behavior: str):
	"""Translates a pose file into its behavioral prediction file.

	Args:
		file: pose file
		behavior: behavior string

	Returns:
		expected prediction filename

	Notes:
		The prediction paths should look like '[experiment]/[video]_behavior/v1/[behavior]/[video].h5'.
		The pose file should be located in '[experiment]/[video][pose_suffix].h5'.
	"""
	video_name = pose_to_video(file)
	return video_to_prediction(os.path.dirname(file) + '/' + video_name, behavior)


def video_to_prediction(file: os.path, behavior: str):
	"""Translates a video file into its behavioral prediction file.

	Args:
		file: video file
		behavior: behavior string

	Returns:
		expected prediction filename

	Notes:
		The prediction paths should look like '[experiment]/[video]_behavior/v1/[behavior]/[video].h5'.
		The video file should be located in '[experiment]/[video].avi'.
	"""
	file_no_folder = os.path.basename(file)
	folder = os.path.dirname(file)
	vid_noext, ext = os.path.splitext(file_no_folder)
	return folder + '/' + re.sub('$', '_behavior/v1/' + behavior + '/' + vid_noext + '.h5', vid_noext)


def pose_to_video(file: os.path):
	"""Translates a pose file into the video filename.

	Args:
		file: pose file

	Returns:
		video filename without directory or extension
	"""
	file_no_folder = os.path.basename(file)
	return re.sub(POSE_REGEX_STR, '', file_no_folder)


def get_pose_v(pose_file: os.path):
	"""Gathers the pose version given a pose filename.

	Args:
		pose_file: pose file

	Returns:
		pose version based on the filename

	Notes:
		The pose filename does not gaurantee data contained matches this version. The data contained should be reflected within the h5 'version' attribute field of the 'poseest' group within the file.
	"""
	pose_ext = re.sub('.*(' + POSE_REGEX_STR + ').*', '\\1', pose_file)
	pose_ext = os.path.splitext(pose_ext)[0]
	pose_v = int(re.sub('[^0-9]', '', pose_ext))
	return pose_v