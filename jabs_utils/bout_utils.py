import numpy as np

# Run length encoding, implemented using numpy
# Accepts a 1d vector
# Returns a tuple containing (starts, durations, values)
def rle(inarray):
	ia = np.asarray(inarray)
	n = len(ia)
	if n == 0: 
		return (None, None, None)
	else:
		y = ia[1:] != ia[:-1]
		i = np.append(np.where(y), n - 1)
		z = np.diff(np.append(-1, i))
		p = np.cumsum(np.append(0, z))[:-1]
		return(p, z, ia[i])

# Removes states of RLE data based on filters
# Returns a new tuple of RLE data
# Note that although this supports removing a list of different values, it may not operate as intended and is safer to sequentially delete ones from the list
# Risky behavior is when multiple short bouts alternate between values that are all going to be removed
# Current behavior is to remove all of those bouts, despite the sum duration being a lot longer that the max_gap_size
# Recommended usage: Only remove one value at a time. For 2 values to remove, this will remove at most 1.5x the max_gap size
def filter_data(starts, durations, values, max_gap_size: int, values_to_remove: list[int] = [0]):
	gaps_to_remove = np.logical_and(np.isin(values, values_to_remove), durations<max_gap_size)
	return delete_bouts_from_rle(starts, durations, values, np.where(gaps_to_remove)[0])

# Helper function to actually delete bouts out of rle data
def delete_bouts_from_rle(starts, durations, values, bouts_to_remove):
	new_durations = np.copy(durations)
	new_starts = np.copy(starts)
	new_values = np.copy(values)
	if len(bouts_to_remove)>0:
		# Delete backwards so that we don't need to shift indices
		for cur_gap in np.sort(bouts_to_remove)[::-1]:
			# Nothing earlier or later to join together, ignore
			if cur_gap == 0 or cur_gap == len(new_durations)-1:
				pass
			else:
				# Delete gaps where the borders match
				if new_values[cur_gap-1] == new_values[cur_gap+1]:
					# Adjust surrounding data
					cur_duration = np.sum(new_durations[cur_gap-1:cur_gap+2])
					new_durations[cur_gap-1] = cur_duration
					# Since the border bouts merged, delete the gap and the 2nd bout
					new_durations = np.delete(new_durations, [cur_gap, cur_gap+1])
					new_starts = np.delete(new_starts, [cur_gap, cur_gap+1])
					new_values = np.delete(new_values, [cur_gap, cur_gap+1])
				# Delete gaps where the borders don't match by dividing the block in half
				else:
					# Adjust surrounding data
					# To remove rounding issues, round down for left, up for right
					duration_deleted = new_durations[cur_gap]
					# Previous bout gets longer
					new_durations[cur_gap-1] = new_durations[cur_gap-1] + int(np.floor(duration_deleted/2))
					# Next bout also needs start time adjusted
					new_durations[cur_gap+1] = new_durations[cur_gap+1] + int(np.ceil(duration_deleted/2))
					new_starts[cur_gap+1] = new_starts[cur_gap+1] - int(np.ceil(duration_deleted/2))
					# Delete out the gap
					new_durations = np.delete(new_durations, [cur_gap])
					new_starts = np.delete(new_starts, [cur_gap])
					new_values = np.delete(new_values, [cur_gap])
	return new_starts, new_durations, new_values

# Returns the distance traveled during bouts
def get_bout_dists(bout_starts, bout_durations, raw_activity):
	# Zero out missing data to not mess up sums
	activity_copy = np.copy(raw_activity)
	activity_copy[activity_copy<0] = 0
	dists = []
	for cur_start, cur_duration in zip(bout_starts, bout_durations):
		dists.append(np.sum(activity_copy[cur_start:cur_start+cur_duration]))
	return np.asarray(dists)

# Flattens a per-animal bout events into an arena events
# If any animal is doing the behavior, the behavior is occurring for the arena. Otherwise, not-behavior.
# Warning: This will fill any "missing animals" as not-behavior
def get_arena_bouts(starts, durations, values):
	bin_vec = np.zeros(np.max(starts+durations), dtype=np.int8)
	for cur_start, cur_dur in zip(starts[values==1], durations[values==1]):
		bin_vec[cur_start:cur_start + cur_dur] = 1
	return rle(bin_vec)

# Filters to_filter_bouts event list by another filter_by_bouts event list
# If multiple control operations are used, then you require either criteria (logical or)
# If you wish to use both criteria (logical and), instead call this function twice.
#
# Control options:
# 1. inverse_discard: reverses the logic for discarding/keeping bout (eg simultaneous becomes mutually exclusive).
# 2. before_tolerance: Requires a filter_by_bout to precede to_filter_bout by at most N frames to be kept. Can be negative to indicate an filter_by ends after to_filter starts.
# 3. after_tolerance: Requires a filter_by_bout to occur after to_filter_bout by at most N frames to be kept. Can be negative to indicate an filter_by starts after to_filter ends.
# 4. intersect: Requires bouts to intersect for a specific quantity of time.
def filter_bouts_by_bouts(to_filter_bouts, filter_by_bouts, inverse_discard: bool=False, before_tolerance: int=None, after_tolerance: int=None, intersect: int=None):
	if before_tolerance is None and after_tolerance is None and intersect is None:
		print('One of before_tolerance, after_tolerance, or intersect must be assigned when calling filter_bouts_by_bouts. Returning unfiltered bouts...')
		return to_filter_bouts
	# Calculate some filtering criteria for potential behavior bouts
	behavior_bout_idxs = np.where(to_filter_bouts[2,:]==1)[0]
	filtering_criteria = np.stack([compare_bout_to_bouts(x, filter_by_bouts[:,filter_by_bouts[2,:]==1]) for x in to_filter_bouts[:,behavior_bout_idxs].T])
	# Figure out which bouts to keep
	# This is indexed from behavior_bout_idxs
	behaviors_to_keep = []
	if before_tolerance is not None:
		if before_tolerance >= 0:
			behaviors_to_keep = np.append(behaviors_to_keep, np.where(filtering_criteria[:,0]<=before_tolerance)[0])
		else:
			behaviors_to_keep = np.append(behaviors_to_keep, np.where(filtering_criteria[:,1]<=-before_tolerance)[0])
	if after_tolerance is not None:
		if after_tolerance >= 0:
			behaviors_to_keep = np.append(behaviors_to_keep, np.where(filtering_criteria[:,3]<=after_tolerance)[0])
		else:
			behaviors_to_keep = np.append(behaviors_to_keep, np.where(filtering_criteria[:,4]<=-after_tolerance)[0])
	if intersect is not None:
		behaviors_to_keep = np.append(behaviors_to_keep, np.where(filtering_criteria[:,3]>=intersect)[0])
	# Get the bouts that we want to remove from the initial list
	if inverse_discard:
		bouts_to_remove = [x for i, x in enumerate(behavior_bout_idxs) if np.isin(i,behaviors_to_keep)]
	else:
		bouts_to_remove = [x for i, x in enumerate(behavior_bout_idxs) if not np.isin(i,behaviors_to_keep)]
	# Actually apply the filtering by deleting bins
	starts, durations, states = delete_bouts_from_rle(to_filter_bouts[0], to_filter_bouts[1], to_filter_bouts[2], bouts_to_remove)
	return (starts, durations, states)

# Helper function for comparing one bout with a list of bouts
# Returns 5 values:
# 1. The closest distance for end before the bout starts
# 2. The closest distance for end after the bout starts
# 3. The maximum overlap
# 4. The closest distance for start before the bout ends
# 5. The closest distance for start after the bout ends
def compare_bout_to_bouts(in_bout, compare_bouts):
	start_times = np.sort(compare_bouts[0])
	end_times = np.sort(compare_bouts[0]+compare_bouts[1])
	# Calculate all "closest" at the same time
	diff_end_start = in_bout[0]-end_times
	ebs = diff_end_start >= 0
	diff_start_end = (in_bout[0]+in_bout[1])-start_times
	sbe = diff_start_end >= 0
	# Pick out the values we want
	closest_ebs = diff_end_start[np.where(ebs)[0][-1]] if np.any(ebs) else diff_end_start[0]
	closest_eas = -diff_end_start[np.where(~ebs)[0][0]] if np.any(~ebs) else -diff_end_start[-1]
	closest_sbe = diff_start_end[np.where(sbe)[0][-1]] if np.any(sbe) else diff_start_end[0]
	closest_sae = -diff_start_end[np.where(~sbe)[0][0]] if np.any(~sbe) else -diff_start_end[-1]
	# Extra calculation for the largest overlap
	overlaps = np.minimum(end_times, in_bout[0]+in_bout[1])-np.maximum(start_times, in_bout[0])
	max_overlap = np.max(np.clip(overlaps, 0, np.inf))
	return (closest_ebs, closest_eas, max_overlap, closest_sbe, closest_sae)
