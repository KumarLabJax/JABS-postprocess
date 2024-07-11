from __future__ import annotations
import h5py
import json
import pandas as pd
import numpy as np
import re
import os
import enum
from pathlib import Path
from datetime import datetime
from typing import List

BEHAVIOR_CLASSIFY_VERSION = 1
POSE_REGEX_STR = '_pose_est_v[2-6].h5'
PREDICTION_REGEX_STR = '_behavior.h5'
FEATURE_REGEX_STR = 'features.h5'
DATE_REGEX_STR = '[0-9]{4}-[0-9]{2}-[0-9]{2}'
DATE_FMT = '%Y-%m-%d'
TIME_REGEX_STR = '[0-9]{2}-[0-9]{2}-[0-9]{2}'
TIME_FMT = '%H-%M-%S'
TIMESTAMP_REGEX_STR = f'{DATE_REGEX_STR}_{TIME_REGEX_STR}'
TIMESTAMP_FMT = f'{DATE_FMT}_{TIME_FMT}'


class MissingBehaviorException(ValueError):
	"""Custom error for behavior-related missing data."""
	def __init__(self, message):
		"""Default initialization."""
		super().__init__(message)


class MissingFeatureException(ValueError):
	"""Custom error for feature-related missing data."""
	def __init(self, message):
		"""Default initialization."""
		super().__init__(message)


class InvalidTableError(ValueError):
	"""Custom error for invalid columns in a data table."""
	def __init__(self, message):
		"""Default initialization."""
		super().__init__(message)


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


class Relation(enum.IntEnum):
	"""A relation operator."""
	LESS_THAN = 0
	LESS_THAN_EQUAL = 1
	GREATER_THAN = 2
	GREATER_THAN_EQUAL = 3

	@classmethod
	def from_str(cls, s: str):
		"""Creates a Relation from a string.

		Args:
			s: string to try and conform to a relation

		Returns:
			The closest enum to the input

		Raises:
			TypeError if there isn't a closest enum
		"""
		s_lower = s.lower()
		if s_lower in ['<', 'lt', 'less than']:
			return cls.LESS_THAN
		elif s_lower in ['<=', '=<', 'lte', 'less than equal', 'less than or equal']:
			return cls.LESS_THAN_EQUAL
		elif s_lower in ['>', 'gt', 'greater than']:
			return cls.GREATER_THAN
		elif s_lower in ['>=', '=>', 'gte', 'greater than equal', 'greater than or equal']:
			return cls.GREATER_THAN_EQUAL
		raise ValueError(f'Invalid relationship operator, supplied: "{s}"')

	def __str__(self):
		strs = ['<', '<=', '>', '>=']
		return strs[self.value]

	def __repr__(self):
		return self.__str__()


class FeatureRule:
	"""A single rule applied to a feature."""
	def __init__(self, feature_name: str, threshold: float, relation: Relation, feature_module: str = None):
		"""Constructs a feature filter rule.

		Args:
			feature_name: feature to apply the filter on
			threshold: threshold value for a decision boundary
			relation: the relation for positive events of this rule
			feature_module: (optional) module the feature belongs to. Only required if a feature exists in 2 different modules
		"""
		self._feature = feature_name
		self._threshold = threshold
		self._relation = relation if isinstance(relation, Relation) else Relation.from_str(relation)

	@property
	def feature(self):
		return self._feature

	@property
	def threshold(self):
		return self._threshold

	@property
	def relation(self):
		return self._relation

	def __str__(self):
		return f'{self._feature} {self._relation} {self._threshold}'

	def __repr__(self):
		return self.__str__()


class WindowFeatureRule(FeatureRule):
	"""A rule applied to window features."""
	def __init__(self, feature_name: str, window_size: int, window_op: str, threshold: float, relation: Relation, feature_module: str = None):
		"""Constructs a window feature filter rule.

		Args:
			feature_name: feature to apply the filter on
			window_size: 
			threshold: threshold value for a decision boundary
			relation: the relation for positive events of this rule
			feature_module: (optional) module the feature belongs to. Only required if a feature exists in 2 different modules
		"""
		super().__init__(feature_name, threshold, relation, feature_module)

		self._window_size = window_size
		self._window_op = window_op

	@property
	def window_size(self):
		return self._window_size

	@property
	def window_op(self):
		return self._window_op

	def __str__(self):
		return f'{self._window_op} {self._feature} {self._relation} {self._threshold}, window_size = {self._window_size}'

	def __repr__(self):
		return self.__str__()
		

class FeatureSettings(ClassifierSettings):
	"""Settings associated with a feature-based classifier."""
	def __init__(self, behavior: str, rules: List[FeatureRule], interpolate: int = 5, stitch: int = 5, min_bout: int = 5):
		"""Initializes a feature settings object.

		Args:
			behavior: string containing the name of the behavior
			rules: list of rules which are combined with an "and" operation for defining an event classifier
			interpolate: number of frames where predictions will be interpolated when data is missing
			stitch: number of frames between "behavior" predictions that will be merged
			min_bout: minimum number of frames for "behavior" predictions to remain
		"""
		super().__init__(behavior, interpolate, stitch, min_bout)
		if len(rules) == 0:
			raise ValueError('No rules provided for detecting events.')
		self._rules = rules

	@property
	def rules(self):
		return self._rules

	def __str__(self):
		return f'Settings: behavior={self._behavior}, interpolate={self._interpolate}, stitch={self._stitch}, filter={self._min_bout}, rules: {" and ".join([str(x) for x in self._rules])}'


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
		self._video = re.sub(f'({POSE_REGEX_STR}|{PREDICTION_REGEX_STR}|\\.avi|\\.mp4)', '', file_str)
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

	def __str__(self):
		return f'Video: {self._video}, Experiment: {self._experiment}, Time: {self.time_str}, Date: {self._date_start}'

	def __repr__(self):
		return self.__str__()


class Bouts:
	"""Object that handles bout data."""
	def __init__(self, starts, durations, values):
		"""Initializes a bouts object.

		Args:
			starts: start indices of bouts
			durations: durations of bouts
			values: state of bouts
		"""
		assert len(starts) == len(durations)
		assert len(starts) == len(values)
		self._starts = starts
		self._durations = durations
		self._values = values

	@property
	def starts(self):
		return self._starts
	
	@property
	def durations(self):
		return self._durations
	
	@property
	def values(self):
		return self._values

	@classmethod
	def from_value_vector(cls, values):
		"""Creates a Bouts object based on time-state vector.

		Args:
			values: state vector where the index indicates time and value indicates state

		Returns:
			Bouts object based on RLE of values
		"""
		starts, durations, values = cls.rle(values)
		return cls(starts, durations, values)

	@staticmethod
	def rle(inarray):
		"""Run-length encode value data.

		Args:
			inarray: input array of data to RLE

		Returns:
			tuple of (starts, durations, values)
			starts: start indices of events
			durations: duration of events
			values: state of events
		"""
		ia = np.asarray(inarray)
		n = len(ia)
		if n == 0: 
			return (None, None, None)
		else:
			y = ia[1:] != ia[:-1]
			i = np.append(np.where(y), n - 1)
			z = np.diff(np.append(-1, i))
			p = np.cumsum(np.append(0, z))[:-1]
			return (p, z, ia[i])

	def shift_start(self, offset: int):
		"""Shifts the starts for all bouts.

		Args:
			offset: offset in frames to add to all starts
		"""
		self._starts = self._starts + offset

	def delete_short_events(self, max_event_length, remove_values):
		"""Removes states from RLE data based on filters.

		Args:
			max_event_length: maximum event length to remove
			remove_values: state to filter out

		Notes:
			Although this function allows for multiple states to be removed, it may produce unwanted behavior. If multiple short bouts alternate between 2 values contained within remove_values, the entire section will be deleted.
		"""
		gaps_to_remove = np.logical_and(np.isin(self.values, remove_values), self.durations < max_event_length)
		return self._delete_bouts(np.where(gaps_to_remove)[0])

	def _delete_bouts(self, indices_to_remove):
		"""Helper function to delete events from bout data.

		Args:
			indices_to_remove: event indices to delete

		Returns:
			Bouts object that has been modified to interpolate within deleted events

		Notes:
			Interpolation on an odd number will result with the "previous" state getting 1 more frame compared to "next" state
		"""
		new_durations = np.copy(self.durations)
		new_starts = np.copy(self.starts)
		new_values = np.copy(self.values)
		if len(indices_to_remove) > 0:
			# Delete backwards so that we don't need to shift indices
			for cur_gap in np.sort(indices_to_remove)[::-1]:
				# Nothing earlier or later to join together, ignore
				if cur_gap == 0 or cur_gap == len(new_durations) - 1:
					pass
				else:
					# Delete gaps where the borders match
					if new_values[cur_gap - 1] == new_values[cur_gap + 1]:
						# Adjust surrounding data
						cur_duration = np.sum(new_durations[cur_gap - 1:cur_gap + 2])
						new_durations[cur_gap - 1] = cur_duration
						# Since the border bouts merged, delete the gap and the 2nd bout
						new_durations = np.delete(new_durations, [cur_gap, cur_gap + 1])
						new_starts = np.delete(new_starts, [cur_gap, cur_gap + 1])
						new_values = np.delete(new_values, [cur_gap, cur_gap + 1])
					# Delete gaps where the borders don't match by dividing the block in half
					else:
						# Adjust surrounding data
						# To remove rounding issues, round down for left, up for right
						duration_deleted = new_durations[cur_gap]
						# Previous bout gets longer
						new_durations[cur_gap - 1] = new_durations[cur_gap - 1] + int(np.floor(duration_deleted / 2))
						# Next bout also needs start time adjusted
						new_durations[cur_gap + 1] = new_durations[cur_gap + 1] + int(np.ceil(duration_deleted / 2))
						new_starts[cur_gap + 1] = new_starts[cur_gap + 1] - int(np.ceil(duration_deleted / 2))
						# Delete out the gap
						new_durations = np.delete(new_durations, [cur_gap])
						new_starts = np.delete(new_starts, [cur_gap])
						new_values = np.delete(new_values, [cur_gap])
		self._starts = new_starts
		self._durations = new_durations
		self._values = new_values

	def filter_by_settings(self, settings: ClassifierSettings):
		"""Filters bouts by all classifier settings options.

		Args:
			settings: ClassifierSettings defining the event filter criteria

		Notes:
			Order of operations is to interpolate (remove no prediction), merge (remove not-behavior), then filter (remove behavior).
		"""
		if settings.interpolate > 0:
			self.delete_short_events(settings.interpolate, [-1])
		if settings.stitch > 0:
			self.delete_short_events(settings.stitch, [0])
		if settings.min_bout > 0:
			self.delete_short_events(settings.min_bout, [1])


class Table:
	"""Object that handles aggregated data."""
	def __init__(self, settings: ClassifierSettings, data: pd.DataFrame):
		"""Initializes a bout object.

		Args:
			settings: settings used for this data
			data: pandas dataframe containing the data
		"""
		self._settings = settings
		self._data = data
		self._required_columns = ['animal_idx', 'video_name', 'start', 'duration', 'is_behavior']
		self._optional_columns = ['longterm_idx', 'exp_prefix', 'time', 'distance', 'closest_id', 'closest_lixit', 'closest_corner']
		if data is not None:
			self._check_fields()

	@property
	def settings(self):
		return self._settings

	@property
	def data(self):
		"""The underlying pandas table."""
		return self._data
	
	@classmethod
	def combine_data(cls, data_list: List(Table)):
		"""Combines multiple data tables together.

		Args:
			data_list: Time-sorted list of Table objects to merge together

		Returns:
			Table object containing the table data concatenated together

		Note:
			Settings from only the first in list are carried forward
		"""
		first_settings = data_list[0].settings
		all_bout_data = pd.concat([cur_data.data for cur_data in data_list])
		return cls(first_settings, all_bout_data)

	@classmethod
	def from_file(cls, file: Path):
		"""Reads in data from a file.

		Args:
			file: prediciton file to read in

		Returns:
			Table object read from file
		"""
		header_data = pd.read_csv(file, nrows=1)
		behavior_name = header_data['Behavior'][0]
		interpolate = header_data['Interpolate Size'][0]
		stitch = header_data['Stitch Gap'][0]
		filter_setting = header_data['Min Bout Length'][0]
		settings = ClassifierSettings(behavior_name, interpolate, stitch, filter_setting)
		df = pd.read_csv(file, skiprows=2)
		return cls(settings, df)

	@staticmethod
	def shift_merge_data(df_list: List[pd.DataFrame]) -> pd.DataFrame:
		"""Shifts data to combine event data.

		Args:
			df_list: dataframes to concatenate together

		Returns:
			a combined dataframe where event start data has been shifted appropriately
		"""
		all_bout_data = []
		cur_offset = 0
		for cur_bout_data in df_list:
			cur_bout_data['start'] += cur_offset
			cur_offset = cur_bout_data['start'].iloc[-1] + cur_bout_data['duration'].iloc[-1]
			all_bout_data.append(cur_bout_data)

		all_bout_data = pd.concat(all_bout_data)
		return all_bout_data

	def to_file(self, file: Path, overwrite: bool = False):
		"""Writes out data to file.

		Args:
			file: prediction file to write out
			overwrite: bool indicating if there exists a file, should it overwrite?

		Raises:
			FileExistsError if file exists and overwrite is False
		"""
		self._check_fields()
		if os.path.exists(file) and not overwrite:
			raise FileExistsError(f'Out_file {file} exists and overwriting was not selected.')
		header_df = pd.DataFrame({
			# 'Project Folder': [self._settings.project_folder],
			'Behavior': [self._settings.behavior],
			'Interpolate Size': [self._settings.interpolate],
			'Stitch Gap': [self._settings.stitch],
			'Min Bout Length': [self._settings.min_bout],
			# 'Out Bin Size': [self._settings.out_bin_size],
		})
		with open(file, 'w') as f:
			header_df.to_csv(f, header=True, index=False)
			self._data.to_csv(f, header=True, index=False)

	def _check_fields(self, restrict_additional: bool = True):
		"""Checks that columns in data are correct.

		Args:
			restrict_additional: bool indicating that no additional columns exist outside the required and optional

		Raises:
			InvalidTableError if either required columns are missing or additional columns exist when restricted
		"""
		# Skip the checks if there is no data.
		if self._data is None or len(self._data) == 0:
			return

		column_names = set(self._data.columns.to_list())
		required_set = set(self._required_columns)
		if not required_set.issubset(column_names):
			raise InvalidTableError(f'Required column(s) not present: {list(required_set.difference(column_names))}')

		if restrict_additional:
			total_expected_columns = set(self._required_columns + self._optional_columns)
			if not column_names.issubset(total_expected_columns):
				raise InvalidTableError(f'Additional columns present: {list(column_names.difference(total_expected_columns))}')


class BoutTable(Table):
	"""Table specific to bout data."""
	def __init__(self, settings: ClassifierSettings, data: pd.DataFrame):
		"""Initializes a bout object.

		Args:
			settings: settings used for this data
			data: pandas dataframe containing the data
		"""
		super().__init__(settings, data)
		self._required_columns = ['animal_idx', 'video_name', 'start', 'duration', 'is_behavior']
		self._optional_columns = ['longterm_idx', 'exp_prefix', 'time', 'distance', 'closest_id', 'closest_lixit', 'closest_corner']
		self._check_fields()

	def to_summary_table(self, bin_size_minutes: int = 60):
		"""Converts bout information into binned summary table.

		Args:
			bin_size_minutes: size of the bin in minutes

		Returns:
			BinTable representation of this data
		"""
		if 'longterm_idx' not in self.data.keys():
			self.data['longterm_idx'] = self.data['animal_idx'] + 1
		grouped_df = self.data.groupby(['exp_prefix', 'longterm_idx'])
		all_results = []
		for cur_group, cur_data in grouped_df:
			binned_results = self.bouts_to_bins(cur_data, bin_size_minutes)
			binned_results['exp_prefix'], binned_results['longterm_idx'] = cur_group
			all_results.append(binned_results)
		all_results = pd.concat(all_results)

		return BinTable(self._settings, all_results)

	def compare_to(self, other: BoutTable, state: int = 1):
		"""Compares these bouts with other bouts.

		Args:
			other: the other bout table to compare overlaps
			state: state to detect overlaps

		Returns:
			tuple of (intersect, union, iou)
			intersect: intersection of bouts
			union: union of bouts
			iou: intersection over union
		"""
		raise NotImplementedError()

	@staticmethod
	def bouts_to_bins(event_df: pd.DataFrame, bin_size_minutes: int = 60, fps: int = 30):
		"""Converts bout data to bin data.

		Args:
			event_df: data frame to be converted
			bin_size_minutes: duration of the bin
			fps: frames per second to convert frame data into bins

		Returns:
			Binned event data describing the event data.

		Notes:
			Binned data describes event data as summaries. For each state, total time and distance travelled are provided. Additionally, the number of behavior events are counted.
			Events that span multiple bins are split between them based on the percent in each, allowing fractional bout counts.
		""" 
		# Get the range that the experiment spans
		try:
			# TODO: Add support for different sized experiment blocks (re-use block below to make an end time that is adjusted per-video)
			start_time = BoutTable.round_hour(datetime.strptime(min(event_df['time']), '%Y-%m-%d %H:%M:%S'))
			end_time = BoutTable.round_hour(datetime.strptime(max(event_df['time']), '%Y-%m-%d %H:%M:%S'), up=True)
		# Timestamp doesn't exist. Make up some. This assumes only 1 video exists and just makes up timestamps based on the available bout data.
		except:
			start_time_str = '1970-01-01 00:00:00'
			start_time = BoutTable.round_hour(datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S'))
			num_hours = np.max(event_df['start'] + event_df['duration']) / fps / 60 // 60 + 1
			end_time_str = start_time_str
			for _ in np.arange(num_hours):
				end_time = BoutTable.round_hour(datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S'), up=True)
				end_time_str = str(end_time)
			# Add one last hour (can be discarded later)
			end_time = BoutTable.round_hour(datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S'), up=True)
			# Also adjust the df to contain valid times
			event_df['time'] = start_time_str
		# Calculate the framewise time bins that we need to summarize
		time_idx = pd.date_range(start=start_time, end=end_time, freq=str(bin_size_minutes) + 'min')
		event_df['adjusted_start'] = [BoutTable.time_to_frame(x['time'], str(start_time), 30) + x['start'] for row_idx, x in event_df.iterrows()]
		event_df['adjusted_end'] = event_df['adjusted_start'] + event_df['duration']
		event_df['percent_bout'] = 1.0
		# Summarize each time bin
		results_df_list = []
		for t1, t2 in zip(time_idx[:-1], time_idx[1:]):
			start_frame = BoutTable.time_to_frame(str(t1), str(start_time), fps)
			end_frame = BoutTable.time_to_frame(str(t2), str(start_time), fps)
			bins_to_summarize = event_df[np.logical_and(event_df['adjusted_start'] >= start_frame, event_df['adjusted_end'] <= end_frame)]
			cut_start = pd.DataFrame.copy(event_df[np.logical_and(event_df['adjusted_start'] < start_frame, event_df['adjusted_end'] >= start_frame)])
			if len(cut_start) > 0:
				new_duration = cut_start['duration'] - (cut_start['adjusted_start'] - start_frame)
				cut_start['percent_bout'] = new_duration / cut_start['duration']
				cut_start['duration'] = new_duration
				bins_to_summarize = pd.concat([bins_to_summarize, cut_start])
			cut_end = pd.DataFrame.copy(event_df[np.logical_and(event_df['adjusted_start'] < end_frame, event_df['adjusted_end'] >= end_frame)])
			if len(cut_end) > 0:
				new_duration = cut_end['duration'] - (end_frame - cut_end['adjusted_start'])
				cut_end['percent_bout'] = new_duration / cut_end['duration']
				cut_end['duration'] = new_duration
				bins_to_summarize = pd.concat([bins_to_summarize, cut_end])
			# With bins_to_summarize as needed
			# This operation throws a warning which can be ignored, so mute it before throwing...
			pd.options.mode.chained_assignment = None
			if 'distance' in bins_to_summarize.keys():
				bins_to_summarize['calc_dist'] = bins_to_summarize['distance'] * bins_to_summarize['percent_bout']
			else:
				pass
			pd.options.mode.chained_assignment = 'warn'
			results = {}
			results['time'] = [str(t1)]
			results['time_no_pred'] = bins_to_summarize.loc[bins_to_summarize['is_behavior'] == -1, 'duration'].sum()
			results['time_not_behavior'] = bins_to_summarize.loc[bins_to_summarize['is_behavior'] == 0, 'duration'].sum()
			results['time_behavior'] = bins_to_summarize.loc[bins_to_summarize['is_behavior'] == 1, 'duration'].sum()
			results['bout_behavior'] = len(bins_to_summarize.loc[bins_to_summarize['is_behavior'] == 1] )
			if 'distance' in bins_to_summarize.keys():
				results['not_behavior_dist'] = bins_to_summarize.loc[bins_to_summarize['is_behavior'] == 0, 'calc_dist'].sum()
				results['behavior_dist'] = bins_to_summarize.loc[bins_to_summarize['is_behavior'] == 1, 'calc_dist'].sum()
			results_df_list.append(pd.DataFrame(results))

		# Remove an non-informative rows
		results = pd.concat(results_df_list)
		no_frames = (results['time_no_pred'] + results['time_not_behavior'] + results['time_behavior'] == 0).values
		results = results[~no_frames]
		return results.reset_index(drop=True)

	@staticmethod
	def round_hour(t, up: bool = False):
		"""Rounds a time to the hour.

		Args:
			t: time object to round
			up: should time be rounded up (True) or down (False)

		Returns:
			time object rounded to the hour
		"""
		hour_to_round_to = t.hour
		# If we want to round up, just increment the hour
		if up:
			hour_to_round_to += 1
		# Do some remainder/modulo operations to handle day wrapping
		return (t.replace(day=t.day + (hour_to_round_to)//24, second=0, microsecond=0, minute=0, hour=hour_to_round_to%24))

	@staticmethod
	def time_to_frame(t: str, rel_t: str, fps: float):
		"""Converts a time relative to another to a frame index.

		Args:
			t: time of interest
			rel_t: time relative to t (typically recording start time)
			fps: frames per second

		Returns:
			frame count difference for t - rel_t
		"""
		delta = datetime.strptime(t, '%Y-%m-%d %H:%M:%S') - datetime.strptime(rel_t, '%Y-%m-%d %H:%M:%S')
		return np.int64(delta.total_seconds() * fps)


class BinTable(Table):
	"""Object that handles time-binned data."""
	def __init__(self, settings: ClassifierSettings, data: pd.DataFrame):
		"""Initializes a binned object.

		Args:
			settings: settings used for these bins
			data: pandas dataframe containing the binned data
		"""
		# Skip passing the data in because it will fail the column check and instead place it in afterwards
		super().__init__(settings, None)
		self._data = data
		self._required_columns = ['longterm_idx', 'time_no_pred', 'time_not_behavior', 'time_behavior', 'bout_behavior']
		self._optional_columns = ['animal_idx', 'exp_prefix', 'time', 'not_behavior_dist', 'behavior_dist']
		self._check_fields()


class Prediction(BoutTable):
	"""A prediction object that defines how to interact with prediction files."""
	def __init__(self, settings: ClassifierSettings, data: pd.DataFrame, video_metadata: VideoMetadata):
		"""Initializes a prediction object.

		Args:
			settings: settings used for these predictions
			data: tabular data to store
			video_metadata: metadata parsed from originating file

		Notes:
			If the dataframe already contains video metadata fields, they will not be overwritten.
		"""
		# While we can pass the data in here, it's safer not to since it will check against default required fields.
		super().__init__(settings, None)

		self._data = data.copy()
		if 'video_name' not in data.keys():
			self._data['video_name'] = video_metadata.video
		if 'exp_prefix' not in data.keys():
			self._data['exp_prefix'] = video_metadata.experiment
		if 'time' not in data.keys():
			self._data['time'] = video_metadata.time_str

		self._file_meta = video_metadata

	@classmethod
	def from_prediction_file(cls, source_file: Path, settings: ClassifierSettings):
		"""Initialized a bout table from a prediction file.

		Args:
			source_file: the file containing predictions
			settings: settings used for these predictions

		Returns:
			Prediction object containing the parsed predictions
		"""
		bout_df = cls.generate_bout_table(source_file, settings)
		video_metadata = VideoMetadata(source_file)

		return cls(settings, bout_df, video_metadata)

	@classmethod
	def from_feature_file(cls, feature_file: Path, feature_settings: FeatureSettings):
		"""Generates predictions from feature-based classification.

		Args:
			feature_file: cached feature file to classify
			feature_settings: settings for classification

		Returns:
			Prediction object based on feature thresholds in settings
		"""
		feature_obj = JABSFeature(feature_file)

		classification_vectors = []
		for cur_feature_rule in feature_settings.rules:
			if isinstance(cur_feature_rule, WindowFeatureRule):
				feature_vector = feature_obj.get_window_feature(cur_feature_rule.feature, cur_feature_rule.window_size, cur_feature_rule.window_op)
			else:
				feature_vector = feature_obj.get_per_frame_feature(cur_feature_rule.feature)

			if cur_feature_rule.relation == Relation.LESS_THAN:
				call_vector = feature_vector < cur_feature_rule.threshold
			elif cur_feature_rule.relation == Relation.LESS_THAN_EQUAL:
				call_vector = feature_vector <= cur_feature_rule.threshold
			elif cur_feature_rule.relation == Relation.GREATER_THAN:
				call_vector = feature_vector > cur_feature_rule.threshold
			elif cur_feature_rule.relation == Relation.GREATER_THAN_EQUAL:
				call_vector = feature_vector >= cur_feature_rule.threshold
			else:
				raise ValueError(f'Rule relation {cur_feature_rule.relation} not supported (full rule: {cur_feature_rule}).')

			# Convert back to 3-state (1: behavior, 0: not behavior, -1: no feature)
			call_vector = call_vector.astype(np.int8)
			call_vector[np.isnan(feature_vector)] = -1

			classification_vectors.append(call_vector)

		classification_vectors = np.stack(classification_vectors)
		# Note that np.all() casts to bool, so "true" here is all elements are either behavior or no feature
		# any 0 values (not behavior) in a row will always evaluate to "false", yielding the desired "and" operation
		final_calls = np.all(classification_vectors, axis=0).astype(np.int8)
		# Note:
		# We could allow the user to do a allow an "or" operation for missing features
		# However, the logic here is to mark any frame where any of the features were missing as no prediction
		final_calls[np.any(classification_vectors == -1, axis=0)] = -1

		bout_data = Bouts.from_value_vector(final_calls)
		bout_data.filter_by_settings(feature_settings)

		bout_df = pd.DataFrame({
			'animal_idx': feature_obj.animal_idx,
			'start': bout_data.starts,
			'duration': bout_data.durations,
			'is_behavior': bout_data.values,
		})

		return cls(feature_settings, bout_df, feature_obj.video_metadata)

	@classmethod
	def from_no_prediction(cls, settings: ClassifierSettings, num_frames: int, num_animals: int, file: Path):
		"""Initializes a bout table with no predictions made.

		Args:
			settings: classifier settings to carry over to the predictions
			num_frames: number of frames in the video where no predictions were made
			num_animals: number of animals to make no predictions on
			file: filename to try and parse video metadata (video, pose, or prediction). Does not need to exist.

		Returns:
			Prediction object with only 1 bout of "no prediction" for each animal
		"""
		bout_df = cls.generate_default_bouts(num_frames, num_animals)
		video_metadata = VideoMetadata(file)

		return cls(settings, bout_df, video_metadata)

	@classmethod
	def combine_data(cls, data_list: List(Table)):
		"""Combines multiple prediction tables together.

		Args:
			data_list: Time-sorted list of Table objects to merge together

		Returns:
			Table object containing the table data concatenated together

		Note:
			Settings from only the first in list are carried forward
		"""
		first_settings = data_list[0].settings
		all_bout_data = pd.concat([cur_data.data for cur_data in data_list])
		first_video_metadata = data_list[0]._file_meta
		return cls(first_settings, all_bout_data, first_video_metadata)

	@staticmethod
	def generate_bout_table(source_file: Path, settings: ClassifierSettings):
		"""Generates a bout table given classifier settings.

		Args:
			source_file: the file containing predictions
			settings: settings used when generating the bouts

		Returns:
			BoutTable containing the predictions.

		Raises:
			MissingBehaviorException if behavior file exists but contains no behavior predictions.
		"""
		with h5py.File(str(source_file), 'r') as in_f:
			if settings.behavior not in in_f['predictions/'].keys():
				available_keys = in_f['predictions'].keys()
				if len(available_keys) > 0:
					behavior_pred_shape = in_f[f'predictions/{available_keys[0]}/predicted_class'].shape
					return Prediction.generate_default_bouts(behavior_pred_shape[0], behavior_pred_shape[1])
				else:
					raise MissingBehaviorException('Prediction file exists, but no behaviors present to discover shape.')
			class_calls = in_f[f'predictions/{settings.behavior}/predicted_class'][:]

		# Iterate over the animals
		bout_dfs = []
		for idx in np.arange(len(class_calls)):
			bout_data = Bouts.from_value_vector(class_calls[idx])
			bout_data.filter_by_settings(settings)
			new_df = pd.DataFrame({
				'animal_idx': idx,
				'start': bout_data.starts,
				'duration': bout_data.durations,
				'is_behavior': bout_data.values,
			})
			bout_dfs.append(new_df)

		bout_dfs = pd.concat(bout_dfs)
		return bout_dfs

	@staticmethod
	def generate_default_bouts(num_animals: int, num_frames: int):
		"""Generates no predictions for the behavior settings.

		Args:
			num_animals: number of animals to generate no prediction
			num_frames: number of frames to pad for no predictions

		Returns:
			BoutTable containing 1 bout of no prediction for the entire prediction size.
		"""
		bout_dfs = []
		for idx in np.arange(num_animals):
			default_df = pd.DataFrame({
				'animal_idx': [idx],
				'start': [0],
				'duration': [num_frames],
				'is_behavior': [-1],
			})
			bout_dfs.append(default_df)

		bout_dfs = pd.concat(bout_dfs)
		return bout_dfs


class JABSAnnotation(BoutTable):
	"""A ground truth object that defines how to interact with JABS-behavior-classifier annotations."""
	def __init__(self, source_file: Path, settings: ClassifierSettings):
		"""Initializes an annotation object.

		Args:
			source_file: JABS annotation json file
			settings: settings used for these annotations
		"""
		super().__init__(settings, pd.DataFrame())
		self._source_file = source_file
		with open(source_file, 'r') as f:
			data = json.load(f)

		vid_name = data['file']
		df_list = []
		for animal_idx, labels in data['labels']:
			for cur_behavior, label_data in labels.items():
				if cur_behavior == settings.behavior:
					new_events = []
					for cur_event in label_data:
						new_df = pd.DataFrame({
							'animal_idx': [animal_idx],
							'behavior': [cur_behavior],
							'start': [cur_event['start']],
							'duration': [cur_event['end'] - cur_event['start'] + 1],
							'is_behavior': [cur_event['present']]
						})
						new_events.append(new_df)
					if len(new_events) > 0:
						df_list.append(pd.concat(new_events))

		if len(df_list) > 0:
			df_list = pd.concat(df_list)
			df_list['video'] = Path(vid_name).stem
		else:
			df_list = pd.DataFrame({'animal_idx': [], 'behavior': [], 'start': [], 'duration': [], 'is_behavior': [], 'video': []})

		self._data = df_list


class JabsProject:
	"""A collection of experiments."""
	def __init__(self, experiments: List[Experiment]):
		"""Initializes a jabs project object.

		Args:
			experiments: list of experiment objects belonging to this project
		"""
		self._experiments = experiments

	@classmethod
	def from_prediction_folder(cls, project_folder: Path, settings: ClassifierSettings):
		"""Constructor based on a predction folder structure.

		Args:
			project_folder: Prediction project folder. Folder is recursively searched for all pose files.
			settings: classifier settings to pass to experiment.

		Returns:
			JabsProject object containing all the poses in the project folder.
		"""
		video_filenames = []
		matched_exp_idxs = {}
		discovered_pose_files = cls.find_pose_files(project_folder)
		# Match files with same experiment
		for i, cur_pose_file in enumerate(discovered_pose_files):
			next_metadata = VideoMetadata(cur_pose_file)
			video_filenames.append(cur_pose_file)
			updated_exp_ids = matched_exp_idxs.get(next_metadata.experiment, [])
			updated_exp_ids.append(i)
			matched_exp_idxs[next_metadata.experiment] = updated_exp_ids

		# Import experiment data
		experiments = []
		for _, meta_idxs in matched_exp_idxs.items():
			experiment_poses = [x for i, x in enumerate(video_filenames) if i in meta_idxs]
			new_experiment = Experiment.from_pose_files(experiment_poses, settings)
			experiments.append(new_experiment)

		return cls(experiments)

	@classmethod
	def from_feature_folder(cls, project_folder: Path, feature_settings: FeatureSettings, feature_folder: Path = Path('.')):
		"""Constructor based on a feature heuristic.

		Args:
			project_folder: Project folder with cached feature data.
			feature_settings: feature settings for constructing events
			feature_folder: optional folder where features are located

		Returns:
			JabsProject object with experimental results for feature based classification

		Todo:
			Expose feature folder searching...
		"""
		discovered_pose_files = cls.find_pose_files(project_folder)

		experiments = []
		for cur_pose in discovered_pose_files:
			new_experiment = Experiment.from_features(cur_pose, feature_settings, feature_folder)
			experiments.add(new_experiment)

		return cls(experiments)

	@classmethod
	def from_pose_files(cls, poses: List[Path], settings: ClassifierSettings):
		"""Constructor based on a list of pose files.

		Args:
			poses: Pose files that may or may not have prediction files.
			settings: classifier settings to pass to experiment.

		Returns:
			JabsProject object containing each pose file as an experiment (pose + prediction).
		"""
		experiments = []
		for cur_pose in poses:
			try:
				new_experiment = Experiment.from_pose_file(cur_pose, settings)
				experiments.append(new_experiment)
			except MissingBehaviorException:
				pass

		if len(experiments) == 0:
			raise FileNotFoundError('No poses contained behavior prediction files.')

		return cls(experiments)

	@staticmethod
	def find_pose_files(path: Path):
		"""Discovers pose files in a folder.

		Args:
			path: Folder to discover pose files

		Returns:
			List of pose files discovered in the folder
		"""
		return Path(path).glob(f'**/*{POSE_REGEX_STR}')

	@staticmethod
	def find_behaviors(path: Path):
		"""Gets the behavior list for this project folder.

		Args:
			path: Path to search for behaviors

		Returns:
			list of behavior keys in the folder
		"""
		found_poses = JabsProject.find_pose_files(path)
		found_predictions = []
		for cur_pose in found_poses:
			try:
				cur_prediction = Experiment.get_prediction_file(cur_pose)
				found_predictions.append(cur_prediction)
			except MissingBehaviorException:
				pass
		found_behaviors = Experiment.get_behaviors(found_predictions)
		return found_behaviors

	def get_bouts(self):
		"""Generates the bout table for the entire experiment.

		Returns:
			BoutTable containing the behavioral bouts for the entire experiment
		"""
		all_bouts = [cur_experiment.get_behavior_bouts() for cur_experiment in self._experiments]
		all_bouts = Prediction.combine_data(all_bouts)
		return all_bouts


class Experiment:
	"""One or more pose files with associated behavior prediction files."""
	def __init__(self, poses: List[Path], predictions: List[Prediction]):
		"""Initializes an experiment object.

		Args:
			poses: list of pose files
			predictions: list of prediction objects associated with the pose files

		"""
		self._pose_files = poses
		self._predictions = predictions

		# If this is more than 1 video we need to do extra steps
		if len(poses) > 1:
			# TODO:
			# Handle identity
			# Handle time sorting
			pass

	@classmethod
	def from_prediction_files(cls, poses: List[Path], predictions: List[Path], settings: ClassifierSettings):
		"""Initializes an experiment object.

		Args:
			poses: list of pose files
			predictions: list of prediction files. Add None values to this list to include pose files without predictions.
			settings: settings associated with a given behavior

		Raises:
			ValueError if length of arguments does not match
		"""
		if len(poses) != len(predictions):
			raise ValueError(f'Poses {len(poses)} did not match predictions {len(predictions)}.')

		predictions = []
		for pose_file, pred_file in zip(poses, predictions):
			if pred_file is not None and Path(pred_file).exists():
				predictions.append(Prediction.from_prediction_file(pred_file, settings))
			else:
				with h5py.File(str(pose_file), 'r') as in_f:
					keypoint_shape = in_f['poseest/points'].shape
				num_frames = keypoint_shape[0]
				# poseest v2 is single animal and only has a dim of 3
				if len(keypoint_shape) > 3:
					num_animals = keypoint_shape[1]
				else:
					num_animals = 1

				predictions.append(Prediction.from_no_prediction(settings, num_frames=num_frames, num_animals=num_animals))

		return cls(poses, predictions)

	@classmethod
	def from_pose_files(cls, pose_files: List[Path], settings: ClassifierSettings, pattern: str = PREDICTION_REGEX_STR, folder: Path = '.', include_missing: bool = True):
		"""Attempts to find a behavior file given pose files.

		Args:
			pose_files: list of pose files
			settings: classifier settings
			pattern: expected pattern to find the behavior file
			folder: folder where the behavior files are located.
			include_missing: flag to construct an experiment without behavior data (True) or to remove them (False)

		Returns:
			Experiment constructed from all the pose files that have associated behavior files.

		Raises:
			MissingBehaviorException if include_missing is False and there are no behavior files for the provided poses.
		"""
		matched_poses, matched_behaviors = [], []
		for pose_f in pose_files:
			try:
				behavior_f = Experiment.get_prediction_file(pose_f, folder)
				matched_poses.append(pose_f)
				matched_behaviors.append(behavior_f)
			except MissingBehaviorException:
				if include_missing:
					matched_poses.append(pose_f)
					matched_behaviors.append(None)

		if len(matched_poses) == 0:
			raise MissingBehaviorException('No poses were matched to behaviors.')

		return cls.from_prediction_files(matched_poses, matched_behaviors, settings)

	@classmethod
	def from_pose_file(cls, pose_file: Path, settings: ClassifierSettings, pattern: str = PREDICTION_REGEX_STR, folder: Path = None, include_missing: bool = True):
		"""Attempts to find a behavior file given pose file.

		Args:
			pose_file: pose file
			settings: classifier settings
			pattern: expected pattern to find the behavior file
			folder: folder where the behavior files are located.
			include_missing: flag to construct an experiment without behavior data (True) or to raise an error (False)

		Returns:
			Experiment constructed from the pose file.

		Raises:
			MissingBehaviorException if include_missing is False and the behavior file was not found.
		"""
		return cls.from_pose_files([pose_file], settings, pattern, folder, include_missing)

	@classmethod
	def from_features(cls, pose_file: Path, feature_settings: FeatureSettings, feature_folder: Path = Path('.')):
		"""Constructs an experiment from feature-based classifications.

		Args:
			pose_file: pose file for this experiment
			feature_settings: settings describing the feature-based classification
			feature_folder: optional path folder location for the features

		Returns:
			Experiment containing the feature-based classification
		"""
		feature_files = cls.find_features(pose_file, feature_folder)

		predictions = []
		for feature_file in feature_files:
			predictions.append(Prediction.from_feature_file(feature_file, feature_settings))

		return cls([pose_file], predictions)

	@staticmethod
	def find_features(pose: Path, feature_folder: Path = Path('.')):
		"""Gets the feature data filenames for a given pose file.

		Args:
			pose: pose file to identify the feature files
			feature_folder: folder in which features were exported

		Returns:
			List of feature files, one for each animal

		Raises:
			MissingFeatureException if no feature files found
		"""
		# Test both a generic path and a path relative to the pose file
		feature_folder_generic = Path(feature_folder) / Path(pose).stem
		feature_folder_pose_rel = Path(pose).parent / Path(feature_folder) / Path(pose).stem
		if os.path.exists(feature_folder_generic):
			feature_folder = feature_folder_generic
		elif os.path.exists(feature_folder_pose_rel):
			feature_folder = feature_folder_pose_rel
		else:
			raise MissingFeatureException(f'Feature folder not found for {Path(pose).stem} (searching {feature_folder_generic} and {feature_folder_pose_rel}).')

		found_feature_files = list(Path(feature_folder).glob(f'**/*{FEATURE_REGEX_STR}'))
		if len(found_feature_files) == 0:
			raise MissingFeatureException(f'No features present in folder {feature_folder}.')
		return found_feature_files

	@staticmethod
	def get_prediction_file(pose_f: Path, folder: Path = '.'):
		"""Identifies a prediction file given a pose file.

		Args:
			pose_f: pose file to find the prediction file
			folder: folder in which the predictions are located

		Returns:
			prediction filename

		Raises:
			MissingBehaviorException if behavior file was not found
		"""
		behavior_f = Path(folder) / Path(re.sub(POSE_REGEX_STR, PREDICTION_REGEX_STR, str(pose_f)))
		if behavior_f.exists():
			return behavior_f
		# Also check the folder with the pose file...
		behavior_f_2 = Path(pose_f).parent / behavior_f.name
		if (behavior_f_2).exists():
			return behavior_f_2

		raise MissingBehaviorException(f'No behavior prediction file found for {pose_f} (searched {behavior_f} and {behavior_f_2}).')

	@staticmethod
	def get_behaviors(predictions: List[Path]):
		"""Behaviors available given a list of predictions.
		
		Args:
			predictions: list of files to detect behaviors

		"""
		behavior_list = []
		for cur_prediction_file in predictions:
			try:
				with h5py.File(cur_prediction_file, 'r') as f:
					new_behaviors = list(f['predictions'].keys())
					behavior_list = set(list(behavior_list) + new_behaviors)
			# Ignore when a file doesn't exist or 'predictions' aren't present
			except (FileNotFoundError, KeyError):
				pass

		return behavior_list

	def get_behavior_bouts(self):
		"""Generates behavior bout data for a given behavior.

		Returns:
			BoutTable containing the bout prediction data

		Raises:
			MissingBehaviorException if behavior was not predicted for this experiment.
		"""
		all_bout_data = Prediction.combine_data(self._predictions)
		return all_bout_data


class JABSFeature:
	"""Methods to interact with JABS feature data."""
	def __init__(self, feature_file: Path):
		"""Initializes a feature object.

		Args:
			feature_file: file to interact with feature data
		"""
		assert os.path.exists(feature_file)

		# Transforms feature file into a metadata object
		# Note that this expects /path/to/feature/files/[POSE_FILE]/[ANIMAL_IDX]/features.h5 as the structure
		pose_file = str(Path(*Path(feature_file).parts[:-2])) + '.h5'
		self._video_metadata = VideoMetadata(pose_file)
		self._animal_idx = int(Path(feature_file).parts[-2])
		self._file = feature_file
		# _populate_window_ops will populate modules as well, but may fail if window features are not cached...
		# Fall back to only populate per-frame keys
		try:
			self._populate_window_ops()
		except KeyError:
			self._populate_per_frame()

	@property
	def video_metadata(self):
		return self._video_metadata
	
	@property
	def animal_idx(self):
		return self._animal_idx

	@property
	def feature_keys(self):
		"""Dataframe containing the available feature keys."""
		return self._feature_keys.copy()

	def get_per_frame_feature(self, feature_key: str, feature_module: str = None):
		"""Retrieves the stored feature vector with a given key.

		Args:
			feature_key: key of the feature to retrieve
			feature_module: Optional module which the features belong to

		Returns:
			np.ndarray containing the feature data

		Raises:
			ValueError if feature module is not provided and doesn't map uniquely
			KeyError if feature key does not exist
		"""
		if feature_module is None:
			feature_module = self.discover_feature_module(feature_key)

		if not (
			(self._feature_keys['module'] == feature_module)
			& (self._feature_keys['feature'] == feature_key)
		).any():
			raise KeyError('Module and Feature key not available.')

		feature_path = f'features/per_frame/{feature_module} {feature_key}'
		with h5py.File(self._file, 'r') as f:
			feature_data = f[feature_path][:]

		return feature_data

	def get_window_feature(self, feature_key: str, window_size: int, window_op: str, feature_module: str = None):
		"""Retrieves the stored feature vector from a window feature.

		Args:
			feature_key: key of the feature to retrieve
			window_size: window size of the window feature
			window_op: window operation for the feature
			feature_module: Optional module which the features belong to

		Returns:
			np.ndarray containing the feature data

		Raises:
			KeyError if the feature key, window size, or window op are not present
		"""
		if feature_module is None:
			feature_module = self.discover_feature_module(feature_key)

		if 'window_op' not in self._feature_keys:
			raise KeyError(f'Feature file {self._file} does not contain window feature data.')
		if not (
			(self._feature_keys['module'] == feature_module)
			& (self._feature_keys['window_op'] == window_op)
			& (self._feature_keys['feature'] == feature_key)
		).any():
			raise KeyError('Module, Window Op, and Feature key not available.')

		feature_key = f'features/window_features_{window_size}/{feature_module} {window_op} {feature_key}'

		with h5py.File(self._file, 'r') as f:
			feature_data = f[feature_key][:]

		return feature_data

	def discover_feature_module(self, feature_key: str):
		"""Discovers the feature module given a key.

		Args:
			feature_key: feature to identify the module name

		Returns:
			module string of the feature

		Raises:
			ValueError if module is not unique
		"""
		discovered_feature_module = np.unique(self._feature_keys.loc[self._feature_keys['feature'] == feature_key, 'module'])
		if len(discovered_feature_module) != 1:
			raise ValueError(f'Feature {feature_key} does not map uniquely to a feature module. Found modules: {discovered_feature_module}')
		return discovered_feature_module[0]

	def _populate_per_frame(self):
		"""Populates the available module-feature pairs."""
		with h5py.File(self._file, 'r') as f:
			available_features = list(f['features/per_frame'].keys())

		self._feature_keys = pd.DataFrame([x.split(' ', 1) for x in available_features], columns=['module', 'feature'])

	def _populate_window_ops(self):
		"""Populates the available window operation data."""
		with h5py.File(self._file, 'r') as f:
			feature_grps = list(f['features'].keys())
		self._window_sizes = [x.split('_')[2] for x in feature_grps if x.startswith('window_features_')]

		if len(self._window_sizes) > 0:
			with h5py.File(self._file, 'r') as f:
				window_keys = list(f[f'features/window_features_{self._window_sizes[0]}'].keys())
			self._feature_keys = pd.DataFrame([x.split(' ', 2) for x in window_keys], columns=['module', 'window_op', 'feature'])
		else:
			raise KeyError('No window features in this feature file.')
