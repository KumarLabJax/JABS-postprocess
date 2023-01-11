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
	new_durations = np.copy(durations)
	new_starts = np.copy(starts)
	new_values = np.copy(values)
	if np.any(gaps_to_remove):
		# Go through backwards removing gaps
		for cur_gap in np.where(gaps_to_remove)[0][::-1]:
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
