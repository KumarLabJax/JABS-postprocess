import numpy as np

# Calculates the intersection of 2 bouts
# Each bout is a tuple of (start, duration)
def calculate_intersection(gt_bout: tuple[int, int], pr_bout: tuple[int, int]):
	# Detect the larger of the 2 start times
	max_start_time = np.max([gt_bout[0], pr_bout[0]])
	# Detect the smaller of the 2 end times
	gt_bout_end = gt_bout[0]+gt_bout[1]
	pr_bout_end = pr_bout[0]+pr_bout[1]
	min_end_time = np.min([gt_bout_end, pr_bout_end])
	# Detect if the 2 bouts intersected at all
	if max_start_time < min_end_time:
		return min_end_time-max_start_time
	else:
		return 0
	
# Calculates the union of 2 bouts
# Each bout is a tuple of (start, duration)
def calculate_union(gt_bout: tuple[int, int], pr_bout: tuple[int, int]):
	# If the 2 don't intersect, we can just sum up the durations
	if calculate_intersection(gt_bout, pr_bout) == 0:
		return gt_bout[1] + pr_bout[1]
	# They do intersect
	else:
		min_start_time = np.min([gt_bout[0], pr_bout[0]])
		gt_bout_end = gt_bout[0]+gt_bout[1]
		pr_bout_end = pr_bout[0]+pr_bout[1]
		max_end_time = np.max([gt_bout_end, pr_bout_end])
		return max_end_time - min_start_time

# Generates an IoU matrix based on bout arrays
def get_iou_mat(gt_bouts: np.array, pr_bouts: np.array):
	intersection_mat = np.zeros([len(gt_bouts), len(pr_bouts)])
	union_mat = np.zeros([len(gt_bouts), len(pr_bouts)])
	# Calculate each of the intersection/unions
	for gt_idx in np.arange(len(gt_bouts)):
		for pr_idx in np.arange(len(pr_bouts)):
			cur_gt_bout = tuple(gt_bouts[gt_idx,:])
			cur_pr_bout = tuple(pr_bouts[pr_idx,:])
			# Calculate the intersections, unions, and IoUs
			intersection_mat[gt_idx, pr_idx] = calculate_intersection(cur_gt_bout, cur_pr_bout)
			union_mat[gt_idx, pr_idx] = calculate_union(cur_gt_bout, cur_pr_bout)
	iou_mat = np.divide(intersection_mat, union_mat, out=np.zeros_like(intersection_mat), where=union_mat!=0)
	return intersection_mat, union_mat, iou_mat

# Define detection metrics, given a IoU threshold
def calc_temporal_iou_metrics(iou_data, threshold):
	tp_counts = 0
	fn_counts = 0
	fp_counts = 0
	for cur_iou_mat in iou_data:
		tp_counts += np.sum(np.any(cur_iou_mat>threshold, axis=1))
		fn_counts += np.sum(np.all(cur_iou_mat<threshold, axis=1))
		fp_counts += np.sum(np.all(cur_iou_mat<threshold, axis=0))
	precision = tp_counts/(tp_counts + fp_counts)
	recall = tp_counts/(tp_counts + fn_counts)
	f1 = 2*(precision * recall)/(precision + recall)
	return precision, recall, f1