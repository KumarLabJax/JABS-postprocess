from pathlib import Path
import re
from datetime import datetime
from copy import deepcopy


POSE_REGEX_STR = '_pose_est_v([2-6]).h5'
PREDICTION_REGEX_STR = '_behavior.h5'
FEATURE_REGEX_STR = 'features.h5'
DATE_REGEX_STR = '[0-9]{4}-[0-9]{2}-[0-9]{2}'
DATE_FMT = '%Y-%m-%d'
TIME_REGEX_STR = '[0-9]{2}-[0-9]{2}-[0-9]{2}'
TIME_FMT = '%H-%M-%S'
TIMESTAMP_REGEX_STR = f'{DATE_REGEX_STR}_{TIME_REGEX_STR}'
TIMESTAMP_FMT = f'{DATE_FMT}_{TIME_FMT}'


class ClassifierSettings:
	"""Settings associated with a classifiers predictions."""
	def __init__(self, behavior: str, interpolate: int = 5, stitch: int = 5, min_bout: int = 5):
		"""Initializes a settings object.

		Args:
			behavior: string containing the name of the behavior
			interpolate: number of frames where predictions will be interpolated when data is missing
			stitch: number of frames between "behavior" predictions that will be merged
			min_bout: minimum number of frames for "behavior" predictions to remain

		Todo:
			Add back in the functionality to change thresholds (useful for searching for data which has poor/odd probabilities).
			This feature added 2 more parameters:
				threshold_min: low threshold for calling behavior (default 0.5)
				threshold_max: high threshold for calling behavior (default 1.0)
		"""
		self._behavior = behavior
		self._interpolate = interpolate
		self._stitch = stitch
		self._min_bout = min_bout

	@property
	def behavior(self):
		return self._behavior
	
	@property
	def interpolate(self):
		return self._interpolate
	
	@property
	def stitch(self):
		return self._stitch
	
	@property
	def min_bout(self):
		return self._min_bout

	def __str__(self):
		return f'Settings: behavior={self._behavior}, interpolate={self._interpolate}, stitch={self._stitch}, filter={self._min_bout}'

	def __repr__(self):
		return self.__str__()


class FeatureSettings(ClassifierSettings):
	"""Settings associated with a feature-based classifier."""
	def __init__(self, behavior: str, config_file: str, interpolate: int = 5, stitch: int = 5, min_bout: int = 5):
		"""Initializes a feature settings object.

		Args:
			behavior: string containing the name of the behavior
			config_file: configuration file indicating heuristic rules
			interpolate: number of frames where predictions will be interpolated when data is missing
			stitch: number of frames between "behavior" predictions that will be merged
			min_bout: minimum number of frames for "behavior" predictions to remain
		"""
		super().__init__(behavior, interpolate, stitch, min_bout)
		self._config_file = config_file
		self._rules = None

	@property
	def config_file(self):
		return self._config_file

	@property
	def rules(self):
		if self._rules is None:
			raise AttributeError('Rules unassigned. Please parse the config file before accessing the rules.')
		return self._rules

	@rules.setter
	def rules(self, value):
		self._rules = value

	def __str__(self):
		if self._rules is None:
			return f'Settings: behavior={self._behavior}, interpolate={self._interpolate}, stitch={self._stitch}, filter={self._min_bout}, rules: UNSET'
		return f'Settings: behavior={self._behavior}, interpolate={self._interpolate}, stitch={self._stitch}, filter={self._min_bout}, rules: {self._rules}'

	def copy(self):
		"""Returns a deep copy."""
		return deepcopy(self)


class VideoMetadata:
	"""Metadata associated with an experimental video."""
	def __init__(self, file: Path):
		"""Initializes a VideoMetadata object.

		Args:
			file: a file pointing to a video, pose file, or behavior prediction file

		Notes:
			The filename can encode information pertaining to a specific video.
		"""
		self._original_file = Path(file)
		self._folder = self._original_file.parent
		file_str = self._original_file.name
		self._video = self.pose_to_video(file_str)
		time_search = re.search(TIMESTAMP_REGEX_STR, self._video)
		video_start_time = time_search.group() if time_search is not None else '1970-01-01_00-00-00'
		self._time = datetime.strptime(video_start_time, TIMESTAMP_FMT)
		self._experiment = self._video[:time_search.start()] if time_search is not None else self._video
		time_search = re.search(DATE_REGEX_STR, str(self._folder))
		self._date_start = time_search.group() if time_search is not None else video_start_time[:10]

	@property
	def folder(self):
		"""Folder of the original path supplied."""
		return self._folder

	@property
	def time(self):
		"""Time object parsed from the file pattern."""
		return self._time

	@property
	def time_str(self):
		"""Formatted version of the time string."""
		return self._time.strftime('%Y-%m-%s %H:%M:%S')

	@property
	def date_start(self):
		"""Date string parsed from folder pattern."""
		return self._date_start

	@property
	def video(self):
		"""Video basename, which is close to the original Path parsed, just with different suffixes removed."""
		return self._video

	@property
	def experiment(self):
		"""Experiment prefix."""
		return self._experiment

	@staticmethod
	def pose_to_video(pose_file):
		"""Converted pose file to expected video file.

		Args:
			pose_file: pose file to convert
		"""
		return re.sub(f'({POSE_REGEX_STR}|{PREDICTION_REGEX_STR}|\\.avi|\\.mp4)', '', str(pose_file))

	def __str__(self):
		return f'Video: {self._video}, Experiment: {self._experiment}, Time: {self.time_str}, Date: {self._date_start}'

	def __repr__(self):
		return self.__str__()
