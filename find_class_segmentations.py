import numpy as np
import pandas as pd
#from itertools import chain
import cv2
#import scipy
import h5py
import plotnine as p9
import sys
import os
import math

# Removes 0-padding from a contour
def get_trimmed_contour(padded_contour, default_val=-1):
	mask = np.all(padded_contour==default_val, axis=1)
	trimmed_contour = np.reshape(padded_contour[~mask,:], [-1,2])
	return trimmed_contour.astype(np.int32)

# Helper function to return a contour list
# Returns a stack of length 1 if only 1 contour was stored
# Otherwise, returns the entire stack of contours
def get_contour_stack(contour_mat, default_val=-1):
	# Only one contour was stored per-mouse
	if np.ndim(contour_mat)==2:
		trimmed_contour = get_trimmed_contour(contour_mat, default_val)
		contour_stack = [trimmed_contour]
	# Entire contour list was stored
	elif np.ndim(contour_mat)==3:
		contour_stack = []
		for part_idx in np.arange(np.shape(contour_mat)[0]):
			cur_contour = contour_mat[part_idx]
			if np.all(cur_contour==default_val):
				break
			trimmed_contour = get_trimmed_contour(cur_contour, default_val)
			contour_stack.append(trimmed_contour)
	return contour_stack

# Returns a stack of masks for all valid contours
# Expects one-frame worth of stored contour data
def get_frame_masks(contour_mat, frame_size=[800, 800]):
	frame_stack = []
	for animal_idx in np.arange(np.shape(contour_mat)[0]):
		new_frame = render_blob(contour_mat[animal_idx], frame_size=frame_size)
		frame_stack.append(new_frame.astype(bool))
	if len(frame_stack)>0:
		return np.stack(frame_stack)
	return np.zeros([0, frame_size[0], frame_size[1]])

# Renders a mask for an individual
# contour is expected to be of shape
# One contour stored:
# [max_contour_length, 2]
# List of contours stored prevernal + internal):
# [max_contours, max_contour_length, 2]
def render_blob(contour, frame_size=[800, 800], default_val=-1):
	new_mask = np.zeros(frame_size, dtype=np.uint8)
	contour_stack = get_contour_stack(contour)
	# Note: We need to plot them all at the same time to have opencv properly detect holes
	_ = cv2.drawContours(new_mask, contour_stack, -1, (1), thickness=cv2.FILLED)
	return new_mask.astype(bool)

# Returns a stack of masks for all valid contours
# Expects one-frame worth of stored contour data
def get_frame_outlines(contour_mat, frame_size=[800, 800], thickness=1):
	frame_stack = []
	for animal_idx in np.arange(np.shape(contour_mat)[0]):
		new_frame = render_outline(contour_mat[animal_idx], frame_size=frame_size, thickness=thickness)
		frame_stack.append(new_frame.astype(bool))
	if len(frame_stack)>0:
		return np.stack(frame_stack)
	return np.zeros([0, frame_size[0], frame_size[1]])

# Renders a mask for an individual
# contour is expected to be of shape
# One contour stored:
# [max_contour_length, 2]
# List of contours stored prevernal + internal):
# [max_contours, max_contour_length, 2]
def render_outline(contour, frame_size=[800, 800], thickness=1, default_val=-1):
	new_mask = np.zeros(frame_size, dtype=np.uint8)
	contour_stack = get_contour_stack(contour)
	# Note: We need to plot them all at the same time to have opencv properly detect holes
	_ = cv2.drawContours(new_mask, contour_stack, -1, (1), thickness=thickness)
	return new_mask.astype(bool)

def h5py_file_to_array(path):
	""" Retrieves segmentation data and returns segmentation and id data
	Args:
		path: h5 pose file of interest
	Returns:
		jabs_seg_data: the segmentation data with dimensions: 
		#frames, #max_animals, #max_contours, #max_contour_length, 2
		longterm_seg_id_data: the corrected JABS labels with the with dimensions: 
		#frames, #max_animals
	"""
	file= h5py.File(path,'r')
	jabs_seg_data = np.array(file['poseest']['seg_data'])
	longterm_seg_id_data = np.array(file['poseest']['longterm_seg_id'])
	return jabs_seg_data, longterm_seg_id_data

def make_segmentation_frames(path,output_folder):
	""" Makes black and white images for each frame of the given video and 
	Args:
		path: the h5 file  
		output_folder: where the images will be saved 
	Returns:
		jabs_seg_data
	"""
	if not os.path.exists('/Users/szadys/jabs-postprocess/'+output_folder):
		os.makedirs('/Users/szadys/jabs-postprocess/'+output_folder)
	jabs_seg_data, longterm_seg_id_data = h5py_file_to_array(path)
	for frame in range(len(jabs_seg_data)):
		all_renders = []
		for mouse_id in range (jabs_seg_data.shape[1]):
			all_renders.append(render_blob(jabs_seg_data[frame, mouse_id]))
		zip_list = [sum(zip_list) for zip_list in zip(all_renders)]
		cv2.imwrite(output_folder + 'frame' + str(frame) +'.png', (sum(zip_list)*255).astype(np.uint8))
	return jabs_seg_data

def erode_segmentation(segmentation_folder,video_name):
	"""
	Args:
		segmentation
	Returns:
	"""
	os.makedirs(video_name+'/'+video_name+'_frames/eroded_and_dilated_1x_filter_775')
	for frame_num in range(len(os.listdir(segmentation_folder))):
		img = cv2.imread(segmentation_folder+ 'frame' + str(frame_num) + '.png', cv2.IMREAD_GRAYSCALE)
		assert img is not None, "file could not be read, check with os.path.exists()"
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12)).astype(np.uint8)
		eroded = cv2.erode(img.copy(),kernel, iterations=1)
		dilated = cv2.dilate(eroded.copy(), kernel, iterations=1)
		cv2.imwrite(video_name+'/'+video_name+'_frames/eroded_and_dilated_1x_filter_775/frame'+str(frame_num)+'eroded_and_dilated_1x_filter_775.png',dilated)

def find_largest_contour(path, num_frames):
	"""
	Args:
	Returns:
	"""
	largest_blob = np.zeros(num_frames)
	num_contours = np.zeros(num_frames)
	largest_blob_idx = np.zeros(num_frames)
	heur_contour_pts = []
	for frame_num in range(num_frames):
		img = cv2.imread(path+ 'frame'+ str(frame_num) + 'eroded_and_dilated_1x_filter_775.png', cv2.IMREAD_GRAYSCALE)
		assert img is not None, "file could not be read, check with os.path.exists()"
		ret,thresh = cv2.threshold(img,127,255,0)
		contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		areas = np.zeros(len(contours))
		contour_list = []
		filt_areas = []
		for i in range(len(contours)):
			cnt = contours[i]
			areas[i] = cv2.contourArea(cnt)
			if hierarchy[0][i][3] == -1 and areas[i]>500:
				contour_list.append(cnt)
				filt_areas.append(areas[i])
		if len(areas) > 0:
			largest_blob[frame_num] = max(areas)
		else:
			largest_blob[frame_num] = 0
		num_contours[frame_num] = len(contour_list)
		heur_contour_pts.append(contour_list)
	return largest_blob, largest_blob_idx, num_contours, heur_contour_pts

def plot_largest_contour(largest_blob, video_name):
	"""
	Args:
	Returns:
	"""
	largest_blob = pd.DataFrame(largest_blob)
	(p9.ggplot(data = largest_blob, mapping = p9.aes(y = largest_blob[0], x = range(len(largest_blob))))+
	p9.geom_line()+
	#p9.geom_hline(yintercept=np.percentile(largest_blob,75))+
	p9.geom_hline(yintercept = np.median(largest_blob), color = 'red')
	+ p9.theme(prev = p9.element_prev(size=28, font= 'Times New Roman'))
	+p9.labs(x='Frame',y='Largest Contour Size (in pixels)', title='Largest Contour in Training Data', size = 50)).save(video_name+'/largest_blob_by_frame_'+video_name+'_eroded_and_dilated_1x_filter_775.png',height=6, width=12, dpi=300)

def plot_num_contours(num_contours,video_name):
	"""
	Args:
		num_contours:
		video_name:
	Returns:
	"""
	num_contours = pd.DataFrame(num_contours)
	(p9.ggplot(data = num_contours, mapping = p9.aes(y = num_contours[0], x = range(len(num_contours))))+
	p9.geom_line()+
	p9.scale_y_continuous(limits=(0,7))+
	p9.labs(x='Frame',y='Number of Contours', title='Numbers of Contours in Training Data')).save(video_name+'/num_contours_by_frame_eroded_and_dilated_1x'+video_name+'_filter_775.png',height=6, width=12, dpi=300)


def blob_to_huddling(largest_blob, num_contours, file_name):
	"""
	Args:
		largest_blob:
		num_contours:
		file_name:
	Returns:
		bouts:
	"""
	data = []
	huddle_length = 0
	start = 0
	end = 0
	for huddle in range(1,len(largest_blob)):
		if num_contours[huddle] == 1:
			huddle_length+= 1
		else:
			if largest_blob[huddle] > np.median(largest_blob):
				huddle_length += 1
			else:
				if huddle_length>775:
					end = huddle
					start = huddle-huddle_length
					data.append({'start':start, 'end': end, 'huddling': True, 'vid_name': file_name})
				huddle_length = 0
				start = 0
				end = 0
	if huddle_length>775:
		end = huddle
		start = huddle-huddle_length
		data.append({'start':start, 'end': end, 'huddling': True, 'vid_name': file_name})
	huddle_length+=1
	bouts = pd.DataFrame(data)
	return bouts

def stitch_bouts(bout_file,file_name, num_mouse, num_frames):
	"""
	Args:
		bout_file:
		file_name:
	Returns:
	"""
	bouts = pd.read_csv(bout_file)
	stitched_bouts = []
	behavior_bouts = bouts.loc[bouts['huddling']==True]
	prev_bout = {'start': behavior_bouts.iloc[0]['start'], 'end': behavior_bouts.iloc[0]['end'], 'huddling': behavior_bouts.iloc[0]['huddling'], 'mouse': behavior_bouts.iloc[0]['mouse'], 'vid_name': file_name}
	#print(behavior_bouts)
	for i in range(1, len(behavior_bouts)):
		bout = behavior_bouts.iloc[i]
		if bout['start'] - prev_bout['end'] < 45 and bout['mouse'] == prev_bout['mouse']:
			prev_bout =  {'start': prev_bout['start'], 'end': bout['end'], "huddling": bout["huddling"], 'mouse': prev_bout['mouse'], 'vid_name': file_name}
		else:
			if prev_bout['end'] - prev_bout['start'] > 775:
				if prev_bout['mouse'] < num_mouse:
					stitched_bouts.append(prev_bout) 
			prev_bout =  {'start': bout['start'], 'end': bout['end'], "huddling": bout["huddling"], 'mouse': bout['mouse'], 'vid_name': file_name}
	if prev_bout['mouse'] < num_mouse:
		stitched_bouts.append(prev_bout)
	stitched_bouts = pd.DataFrame(stitched_bouts)
	#print(stitched_bouts)
	binary_bouts = np.zeros((num_mouse, num_frames))
	for i in range(stitched_bouts['mouse'].max()+1):
		mouse = stitched_bouts.loc[stitched_bouts['mouse']==i]
		for j in range(len(mouse)):
			binary_bouts[i,mouse.iloc[j]['start']:mouse.iloc[j]['end']] = np.ones(len(range(mouse.iloc[j]['start'],mouse.iloc[j]['end'])))
	filtered_bouts = []
	binary_stitched_bouts = np.zeros((stitched_bouts['mouse'].max()+1, stitched_bouts['end'].max()))
	for i in range(stitched_bouts['mouse'].max()+1):
		huddle_length = 0
		not_huddle_length= 0
		print(len(binary_bouts[0]))
		for j in range(len(binary_bouts[0])):
			if sum(binary_bouts[:,j]) > 1 and binary_bouts[i,j] == 1: #if the frame is huddling
				if not_huddle_length>0 and huddle_length == 0: #if the false is long enough
					temp = {'start': j-not_huddle_length, 'end': j+1, "huddling": False, 'mouse': i, 'vid_name': file_name}
					filtered_bouts.append(temp)
				huddle_indicator = True
				huddle_length += 1
			else:
				if huddle_length>775: #if the true is long enough
					temp = {'start': j-huddle_length, 'end': j+1, "huddling": True, 'mouse': i, 'vid_name': file_name}
					binary_stitched_bouts[i, j-huddle_length: j+1] = np.ones(huddle_length+1)
					filtered_bouts.append(temp)
					not_huddle_length = 0
					huddle_length = 0
				else:
					if huddle_length != 0:
						#print(j)
						not_huddle_length = huddle_length
					else:
						not_huddle_length += 1
					huddle_length = 0
					huddle_indicator = False
		if huddle_length>0:
			huddle_indicator = True
			filtered_bouts.append({'start': j-huddle_length+1, 'end': j, "huddling": huddle_indicator, 'mouse': i, 'vid_name': file_name})				
			binary_stitched_bouts[i, j-huddle_length: j+1] = np.ones(huddle_length+1)
		else:
			huddle_indicator=False
			filtered_bouts.append({'start': j-not_huddle_length+1, 'end': j, "huddling": huddle_indicator, 'mouse': i, 'vid_name': file_name})				
	filtered_bouts = pd.DataFrame(filtered_bouts)
	print(filtered_bouts)
	bout_length = 0
	print(np.where(np.sum(binary_stitched_bouts, axis=0)==1))
	wrong_start=0
	for i in range(len(binary_stitched_bouts)):
		for j in range(len(binary_stitched_bouts[i])):
			if sum(binary_stitched_bouts[:,j]) == 1 and binary_stitched_bouts[i,j]==1:
				bout_length += 1
				if bout_length == 1:
						wrong_start = j
						print(wrong_start)
			else:
				if bout_length > 10 and wrong_start!=0:
					curr_mouse = filtered_bouts.loc[filtered_bouts['mouse']==i]
					print(curr_mouse)
					print(wrong_start)
					print(bout_length)
					print(i)
					print(filtered_bouts.loc[filtered_bouts['mouse']==i].loc[filtered_bouts['end'] == wrong_start+bout_length, 'end'])
					print(filtered_bouts.loc[filtered_bouts['mouse']==i].loc[filtered_bouts['start'] == wrong_start+bout_length, 'start'])
					if sum(binary_stitched_bouts[:,wrong_start]) == 1 and binary_stitched_bouts[i,wrong_start]==1:
						if not filtered_bouts.loc[filtered_bouts['mouse']==i].loc[filtered_bouts['end'] == wrong_start+bout_length, 'end'].empty and not filtered_bouts.loc[filtered_bouts['mouse']==i].loc[filtered_bouts['start'] == wrong_start+bout_length, 'start'].empty: #post huddling
							print(bout_length)
							idx = curr_mouse.index[curr_mouse['end']==wrong_start+bout_length][0]
							print(idx)
							next_idx = idx + 1
							print(next_idx)
							curr = filtered_bouts.iloc[idx]
							curr['end'] = wrong_start
							next = filtered_bouts.iloc[next_idx]
							next['start'] = wrong_start 
							filtered_bouts.iloc[idx] = curr
							filtered_bouts.iloc[next_idx] = next	
						elif filtered_bouts.loc[filtered_bouts['mouse']==i].loc[filtered_bouts['end'] == wrong_start+bout_length, 'end'].empty: #pre huddling
								print('pre')
								print(filtered_bouts.loc[filtered_bouts['mouse']==i].loc[filtered_bouts['start'] == wrong_start+bout_length, 'start'])
								print(filtered_bouts.index[filtered_bouts['start']==wrong_start])
								idx = curr_mouse.index[curr_mouse['start']==wrong_start].values
								print(idx)
								if len(idx) == 0:
									idx = curr_mouse.index[curr_mouse['start']==wrong_start+1].values
								print(idx)
								prev_idx = idx - 1
								print(prev_idx)
								curr = filtered_bouts.iloc[idx]
								print(curr)
								prev = filtered_bouts.iloc[prev_idx]
								prev['end'] = wrong_start+bout_length
								prev['huddling'] = False
								curr['start'] = wrong_start+bout_length+1
								print(curr)
								print(prev)
								filtered_bouts.iloc[idx] = curr
								filtered_bouts.iloc[prev_idx] = prev
						elif filtered_bouts.loc[filtered_bouts['mouse']==i].loc[filtered_bouts['start'] == wrong_start+bout_length, 'start'].empty:
							print('post')
							print(filtered_bouts.loc[filtered_bouts['mouse']==i].loc[filtered_bouts['end'] == wrong_start+bout_length, 'end'])
							print(filtered_bouts.index[filtered_bouts['end']==wrong_start])
							idx = filtered_bouts.index[filtered_bouts['end']==wrong_start]
							next_idx = curr_mouse.index[curr_mouse['start']==wrong_start][0]+1
							print(next_idx)
							curr = filtered_bouts.iloc[idx]
							curr['end'] = wrong_start
							next = filtered_bouts.iloc[next_idx]
							next['start'] = wrong_start+bout_length+1
							print(curr)
							print(prev)
							filtered_bouts.iloc[idx] = curr
							filtered_bouts.iloc[next_idx] = next		
				bout_length=0
	filtered_bouts.to_csv('gt_stitched_with_identity_csvs/'+file_name+'.csv')
	return filtered_bouts

def huddling_ethogram(bout_data,video_name,frames):
	"""
	Args:
		bout_data:
		video_name:
	Returns:
	"""
	if len(bout_data) == 0:
		return "No huddling Detected"
	(p9.ggplot(pd.DataFrame(bout_data))+
    	p9.geom_rect(p9.aes(xmin='start', xmax='end',ymin='mouse-0.25',ymax='mouse+0.25', fill='huddling'))+
     	p9.scale_fill_brewer(type='qual', palette='Set1')+
		p9.scale_x_continuous(limits=(0,frames))+
		p9.scale_y_continuous(limits=(-0.5,2.5))+
     	p9.labs(x='Frame',y='Mouse Label', title='Class Based Segmentation Huddling Classifier')).save(video_name+'/'+video_name +'_with_identities.png', height=6, width=12, dpi=300)


def multivideo_processing(directory):
	"""
	Args:
		directory: 
	Returns:
	"""
	pose_files = [ f for f in os.listdir(directory) if f.endswith('.h5') & f.startswith('MD')]
	num_frames = 3600
	for vid in range(len(pose_files)):
		video_name = str(pose_files[vid]).split('_')[:3]
		video_name = "_".join(video_name[:3])
		segmentation_path = video_name+'/'+video_name+'_frames/original/'
		eroded_path = video_name+'/'+video_name+'_frames/eroded_and_dilated_1x_filter_775/'
		csv_path = 'heuristic_stitched_csvs/filter_775/' + video_name + '_375_775.csv'
		print(segmentation_path)
		if not os.path.exists('/Users/szadys/jabs-postprocess/'+segmentation_path):
			os.makedirs('/Users/szadys/jabs-postprocess/'+segmentation_path)
			make_segmentation_frames(directory+pose_files[vid],segmentation_path)
		if not os.path.exists('/Users/szadys/jabs-postprocess/'+eroded_path):
			erode_segmentation(segmentation_path, video_name)		 
		if not os.path.exists('/Users/szadys/jabs-postprocess/'+csv_path):
			largest_blob, largest_blob_idx, num_contours, heur_contour_pts = find_largest_contour(eroded_path, num_frames)
			bout_data = blob_to_huddling(largest_blob, num_contours, video_name)
			print(bout_data)
			if len(bout_data) > 1:
				stitch_bouts(bout_data, video_name)
			else:
				#bout_data = pd.DataFrame({'start': 0, 'end': 3599, "huddling": False})
				pd.DataFrame(bout_data).to_csv(csv_path)
				huddling_ethogram(bout_data,video_name)
				plot_largest_contour(largest_blob,video_name)
				plot_num_contours(num_contours,video_name)
	
def applying_identities(path, video_name, num_contours, heur_contour_pts, largest_blob_idx):
	""" 
	Stores all the identities with all the details about centroids and centroid distances,
	essemtially the unrefined but more comprehensive version of simple_identities
	Args:
		path: what folder the video of interest is in
		video_name: the name of the video
		num_contours: the number of contours in the video
		heur_contour_pts: a list of all the points that outline the contour
		largest_blob_idx: the index of the largest blob in the list of blob sizes
	"""
	jabs_seg_data, longterm_seg_id_data = h5py_file_to_array(path)
	identities = []
	for frame in range(len(jabs_seg_data)): #go through each frame
		jabs_mice_num = 0
		temp = []
		for seg in range(len(jabs_seg_data[frame])): # calculate how many mice are in the current frame
			if -1 not in jabs_seg_data[frame, seg][0][0]:
				jabs_mice_num += 1
		id_list = np.zeros(jabs_mice_num)
		for cntr in range(int(num_contours[frame])): #go through each contour in heuristic classifier
			centroids = []
			dists = []
			for jabs_mouse_num in range(jabs_mice_num): #go through each mouse in the h5 file
				moments = cv2.moments(heur_contour_pts[frame][cntr]) # calculating image moments derived from physics
				centroid = [(moments['m10']/moments['m00']),(moments['m01']/moments['m00'])] #calculating the centroid from the heuristic segmentation
				mouse = get_contour_stack(jabs_seg_data[frame, jabs_mouse_num])[0]
				jabs_moments = cv2.moments(mouse) # calculating image moments derived from physics
				jabs_centroid = [(jabs_moments['m10']/jabs_moments['m00']),(jabs_moments['m01']/jabs_moments['m00'])] #calculating the centroid from the JABS data
				centroid_dist = math.dist(centroid, jabs_centroid) #linear distance between the 2 centroids
				centroids.append(jabs_centroid)
				dists.append(centroid_dist)
			mouse_num = dists.index(min(dists))
			if jabs_mouse_num != 4 and longterm_seg_id_data[frame][mouse_num] != 0: #not a false identity (all videos have >= 4 mice) and longterm_seg_id_data starts at 1 and 0 is no mouse detected
				seg_id = longterm_seg_id_data[frame][mouse_num]-1 
			else: # the one video in the GT data set has 4 mice and if the id is 0 they will stay the same
				seg_id = mouse_num
			if num_contours[frame] < jabs_mice_num: # if huddling is occurring
				if cntr != int(largest_blob_idx[frame]): # if the current mouse isn't huddling
					id_list[seg_id-1] = 1
					temp.append({'num mice': jabs_mice_num, 'num contours': num_contours[frame],'frame':frame, 'contour': cntr, 'seg_ids':seg_id, 'jabs_mouse_num': seg_id, 'centroids': centroids, 'centroid': centroid, 'centroid distances': dists})
				else: # current mouse is huddling, will be modified in a few lines
					temp.append({'num mice': jabs_mice_num, 'num contours': num_contours[frame],'frame':frame, 'contour': cntr, 'seg_ids':seg_id, 'jabs_mouse_num': -1, 'centroids': centroids, 'centroid': centroid, 'centroid distances': dists})
			else: # current mouse isn't huddling
				temp.append({'num mice': jabs_mice_num, 'num contours': num_contours[frame],'frame':frame, 'contour': cntr, 'seg_ids':seg_id, 'jabs_mouse_num': seg_id, 'centroids': centroids, 'centroid': centroid, 'centroid distances': dists})
		if num_contours[frame] < jabs_mice_num: #changing the huddling contours to contain more than one mouse
			temp[int(largest_blob_idx[frame])].update({'jabs_mouse_num':np.where(id_list == 0)[0]})
		identities.append(temp)
	print(identities[0])


def simple_identities(path, video_name, num_contours, heur_contour_pts, largest_blob_idx):

	""" 
	Very similar to applying identities, but stores a more simple dataframe in a csv file
	Args:
		path: what folder the video of interest is in
		video_name: the name of the video
		num_contours: the number of contours in the video
		heur_contour_pts: a list of all the points that outline the contour
		largest_blob_idx: the index of the largest blob in the list of blob sizes
	"""	

	jabs_seg_data, longterm_seg_id_data = h5py_file_to_array(path)
	identities = pd.DataFrame()
	huddle_status = False
	biggest = []
	for frame in range(len(jabs_seg_data)): #go through each frame
		jabs_mice_num = 0
		temp = []
		for seg in range(len(jabs_seg_data[frame])): #calculate the number of mice in the frame
			if -1 not in jabs_seg_data[frame, seg][0][0]:
				jabs_mice_num += 1
		id_list = -np.ones(4)
		for cntr in range(int(num_contours[frame])): #go through each contour in heuristic classifier
			centroids = []
			dists = []
			for jabs_mouse_num in range(jabs_mice_num): #go through each mouse in the h5 file
				moments = cv2.moments(heur_contour_pts[frame][cntr])
				centroid = [(moments['m10']/moments['m00']),(moments['m01']/moments['m00'])] #calculates the centroid of the segmentation data
				mouse = get_contour_stack(jabs_seg_data[frame, jabs_mouse_num])[0]
				jabs_moments = cv2.moments(mouse)
				jabs_centroid = [(jabs_moments['m10']/jabs_moments['m00']),(jabs_moments['m01']/jabs_moments['m00'])] #calculates the centroid of the JABS data
				centroid_dist = math.dist(centroid, jabs_centroid)
				centroids.append(jabs_centroid)
				dists.append(centroid_dist)
			mouse_num = dists.index(min(dists)) # finds the smallest distance for the given contour
			if jabs_mouse_num < 4 and longterm_seg_id_data[frame][mouse_num] != 0: #
				seg_id = longterm_seg_id_data[frame][mouse_num]-1
			elif jabs_mice_num > 4: #too many contours, typically noise
				seg_id = 0
			else: # the identity set was correct so it stays the same
				seg_id = mouse_num
			if cntr != int(largest_blob_idx[frame]): #if the current contour is not part of the largest blob 
				id_list[seg_id] = 0
			else: # current contour is the largest 
				id_list[seg_id] = 1
		identities = pd.concat([identities,pd.DataFrame({'frame':frame, 'contour':cntr, 'seg_id': [seg_id], 'mouse_0': id_list[0], 'mouse_1': id_list[1], 'mouse_2': id_list[2], 'mouse_3':id_list[3], 'huddling': huddle_status})], ignore_index=True)
		if num_contours[frame] < jabs_mice_num and not identities.empty: # when the number of number of contours is less than the number of mice (huddling)
			huddle_mice = np.where(id_list[:jabs_mice_num] == -1)[0] #have not been changed from original list
			for i in range(len(huddle_mice)):
				name = 'mouse_'+ str(huddle_mice[i])
				identities.loc[identities['frame']==frame, name] = 1
			identities.loc[identities['frame']==frame, 'huddling'] = True #huddling happens at this frame
		biggest.append(id_list[0])
	print(biggest)
	pd.DataFrame(identities).to_csv(video_name+'/'+video_name+'_seg_ids_simple.csv')

def behavior_with_identity(video_name,behavior_file, behavior, num_mouse):
	behavior_data = pd.read_csv(behavior_file) 
	behavior_length = 0
	not_behavior_length = 0
	bout_data = behavior_data.loc[behavior_data[behavior]==True]
	mouse_data = behavior_data[behavior_data.columns[4:]]
	bout_lengths = []
	temp = []
	for i in range(num_mouse):
		curr_mouse = mouse_data[mouse_data.columns[i]]
		behavior_length = 0
		not_behavior_length = 0
		for j in range(len(mouse_data)):
			if j == len(mouse_data)-1:
				if not_behavior_length > 0:
					huddle_state = False
					temp.append({'start': j-not_behavior_length, 'end': j-1, 'huddling': huddle_state, 'mouse': i})
					not_behavior_length = 0
				if behavior_length > 0:
					huddle_state = True
					temp.append({'start': j-behavior_length, 'end': j-1, 'huddling': huddle_state, 'mouse': i})
					behavior_length = 0
			else:
				if curr_mouse[j] == 1:
					behavior_length += 1
					huddle_state = False
					if not_behavior_length > 0:
						temp.append({'start': j-not_behavior_length, 'end': j-1, 'huddling': huddle_state, 'mouse': i})
						not_behavior_length = 0
				elif curr_mouse[j] == 0:
					not_behavior_length+=1
					huddle_state = True
					if behavior_length > 0:
						temp.append({'start': j-behavior_length, 'end': j-1, 'huddling': huddle_state, 'mouse': i})
						behavior_length = 0
		if curr_mouse[j] == 1:
			behavior_length += 1
			if not_behavior_length > 0:
				temp.append({'start': j-not_behavior_length, 'end': j-1, 'huddling': False, 'mouse': i})
				not_behavior_length = 0
		elif curr_mouse[j] == 0:
			not_behavior_length+=1
			if behavior_length > 0:
				temp.append({'start': j-behavior_length, 'end': j-1, 'huddling': True, 'mouse': i})
				behavior_length = 0	
	pd.DataFrame(temp).to_csv(video_name+'/'+video_name+'.csv')

def main(argv):
	video_name = '005_3_revised'
	directory = '/Users/szadys/Desktop/gt_with_seg_data/'
	path = directory+video_name+'_pose_est_v5.h5' 
	output_folder = video_name + '/' + video_name + '_frames/original/'
	num_frames = 17991
	num_mouse = 4
	#jabs_seg_data = make_segmentation_frames(path, output_folder)
	#erode_segmentation(output_folder, video_name)
	# largest_blob, largest_blob_idx, num_contours, heur_contour_pts = find_largest_contour(video_name+'/'+video_name+'_frames/eroded_and_dilated_1x_filter_775/', num_frames)
	# huddling_identities = simple_identities(path, video_name, num_contours, heur_contour_pts, largest_blob_idx) 
	# behavior_with_identity(video_name,video_name+'/'+video_name+'_seg_ids_simple.csv', 'huddling', num_mouse)
	stitched_bouts = stitch_bouts(video_name+'/'+video_name+'.csv', video_name+'_with_identities', num_mouse, num_frames)
	huddling_ethogram(stitched_bouts, video_name, num_frames)

if __name__ == '__main__':
    main(sys.argv[1:])