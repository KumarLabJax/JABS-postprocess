import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import json
import plotnine as p9
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import seaborn as sns

def compare_classifiers(jabs_bouts, heuristic_bouts, num_frames, num_vids):
    """ Makes a confusion matrix comparing the jabs and heuristic huddling classifiers
    Args:
        jabs_bouts: name of the csv file that contains all of the jabs classifier huddling bouts
        heuristic_bouts: name of the csv file that contains all of the heuristic classifier huddling bouts
        num_frames: length of the videos (same between jabs and heuristic)
        num_vids: number of videos being compared (same between jabs and heuristic)
    """
    jabs_array = np.zeros(num_frames*num_vids)
    heuristic_array = np.zeros(num_frames*num_vids)
    # go through each video
    for vid_num in range(num_vids):
        jabs = pd.read_csv('/Users/szadys/jabs-postprocess/jabs_stitched_csvs/'+jabs_bouts[vid_num])
        heuristic = pd.read_csv('/Users/szadys/jabs-postprocess/heuristic_stitched_csvs/'+heuristic_bouts[vid_num])
        for i in range(num_frames):
            if not heuristic.empty: #some will be empty because there is no huddling behavior
                for j in range(len(heuristic['start'])):
                    if i in range(int(heuristic['start'][j]),int(heuristic['end'][j]+1)):
                        heuristic_array[((vid_num-1)*num_frames)+i] = 1
            if not jabs.empty:  #some will be empty because there is no huddling behavior
                for k in range(len(jabs['start'])):
                    if i in range(int(jabs['start'][k]),int(jabs['end'][k])+1):
                        if jabs['Huddling Behavior'][k] == True:
                            jabs_array[((vid_num-1)*num_frames)+i] = 1
    F1_score = f1_score(jabs_array, heuristic_array)

    cm = confusion_matrix(jabs_array,heuristic_array)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title("F1 Score: " + str(F1_score))
    plt.savefig('total_confusion_matrix_1.png')

def single_vid_compare_classifiers(jabs_bouts, heuristic_bouts, num_frames):
    """ Makes a confusion matrix comparing the jabs and heuristic huddling classifiers for a single video
    Args:
        jabs_bouts: name of the csv file that contains the jabs classifier huddling bouts for the given video
        heuristic_bouts: name of the csv file that contains the heuristic classifier huddling bouts for the given video
        num_frames: length of the videos (same between jabs and heuristic)
    """
    jabs = pd.read_csv(jabs_bouts)
    heuristic = pd.read_csv(heuristic_bouts)
    jabs_array = np.zeros(num_frames)
    heuristic_array = np.zeros(num_frames)
    for i in range(num_frames):
        for j in range(len(heuristic['start'])):
            if i in range(int(heuristic['start'][j]),int(heuristic['end'][j]+1)):
                heuristic_array[i] = 1
        for k in range(len(jabs['start'])):
            if i in range(int(jabs['start'][k]),int(jabs['end'][k])+1):
                if jabs['Huddling Behavior'][k] == True:
                    jabs_array[i] = 1
    F1_score = f1_score(jabs_array, heuristic_array)
    cm = confusion_matrix(jabs_array,heuristic_array)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title("F1 Score: " + str(F1_score))
    plt.savefig('total_confusion_matrix_1.png')

def combine_csv(directory):
    """ Combines the individual video csv files into 1 csv file
    Args:
        directory: path to the individual video csv files 
        classifier_type: jabs or heuristic
    """
    huddling_csvs = [ f for f in os.listdir(directory) if f.endswith('.csv')]
    all_data = pd.concat([pd.read_csv(directory+f, index_col = 0) for f in huddling_csvs]).reset_index(drop=True)
    all_data['jabs'] = True
    #all_data['vid_name'] = ["_".join(i.split('_')[:2]) for i in all_data['vid_name']]
    print(all_data)
    all_data.to_csv('jabs_gt_all_bouts.csv')

def jsons_to_csv(directory, mouse_nums, frame_nums):
    sorted_arr = []
    files = []
    excluded_arr = []
    sorted_directory = sorted(os.listdir(directory))
    print(sorted_directory)
    count = 0
    for file in [f for f in sorted_directory if f.endswith('.json')]:
        count+=1
        files.append(file.split('.')[0])
        with open(directory+file) as json_file:
            data = json.load(json_file)
            #print(data)
            labels = list(data['labels'].keys())
            if len(labels) < mouse_nums[count-1]:
                print(file)
                #print(mouse_nums)
                sorted_labels = sorted(labels)
                sorted_labels = [eval(i) for i in sorted_labels]
                #print(sorted_labels)
                print(mouse_nums[count])
                for i in range(mouse_nums[count]):
                    if i not in sorted_labels:
                        #print(i)
                        #print(file)
                        excluded_data = {'start': 0, 'end':  frame_nums[count]-1,'huddling': False,'mouse': i,'vid_name': "_".join(file.split('_')[:2]), 'jabs': True}
                        excluded_arr.append(excluded_data)

            #print(labels)
            for i in labels:
                mouse = data['labels'][i]
                #print(mouse)
                for j in range(len(list(mouse.values())[0])):
                    formatted_data = {'start': mouse['huddling'][j]['start'], 'end':  mouse['huddling'][j]['end'],'huddling': mouse['huddling'][j]['present'],'mouse': i,'vid_name': "_".join(file.split('_')[:2]), 'jabs': True}
                    #print(formatted_data)
                    sorted_arr.append(formatted_data)
    with_false_arr = []
    #print(sorted_arr[0])
    #print(np.where(np.array(files) == sorted_arr[0]['vid_name'].split('.')[0])[i])
    if sorted_arr[0]['start'] != 0:
        with_false_arr.append({'start': 0, 'end':  sorted_arr[0]['start']-1,'huddling':False,'mouse': sorted_arr[0]['mouse'],'vid_name': sorted_arr[0]['vid_name'], 'jabs': True})
    for i in range(1,len(sorted_arr)):
        #print(sorted_arr[i])
        with_false_arr.append(sorted_arr[i-1])
        if sorted_arr[i]['vid_name'] == sorted_arr[i-1]['vid_name'] and sorted_arr[i]['mouse'] == sorted_arr[i-1]['mouse']:
            # print(i)
            # print(sorted_arr['mouse'][i-1])
            # print(sorted_arr['start'][i-1])
            with_false_arr.append({'start': sorted_arr[i-1]['end']+1, 'end':  sorted_arr[i]['start']-1,'huddling':False,'mouse': sorted_arr[i]['mouse'],'vid_name': sorted_arr[i]['vid_name'], 'jabs': True})
        elif not sorted_arr[i]['mouse'] == sorted_arr[i-1]['mouse']:
            num_frames = np.array(frame_nums)[np.where(np.array(files) == sorted_arr[0]['vid_name'].split('.')[0])[0]]-1
            #print(num_frames)
            #print(frame_nums[i])
            if sorted_arr[i]['start'] != 0:
                with_false_arr.append({'start': 0, 'end':  sorted_arr[i]['start']-1,'huddling':False,'mouse': sorted_arr[i]['mouse'],'vid_name': sorted_arr[i]['vid_name'], 'jabs': True})
            if sorted_arr[i]['end'] != frame_nums[count-1]:
                #print(frame_nums[count-1])
                #print(sorted_arr[i-1]['vid_name'])
                with_false_arr.append({'start': sorted_arr[i-1]['end']+1, 'end':  frame_nums[count-1],'huddling':False,'mouse': sorted_arr[i-1]['mouse'],'vid_name': sorted_arr[i-1]['vid_name'], 'jabs': True})
        #print(with_false_arr)
    with_false_arr.append(sorted_arr[i])
    total = (pd.concat([pd.DataFrame(with_false_arr),(pd.DataFrame(excluded_arr))])).sort_values(by=['vid_name','mouse']).reset_index(drop=True)
    #print(excluded_arr[0])
    #print(total)
    total.to_csv('gt_original.csv')

# def add_false(true_csv, mouse_num, frame_num, vid_list):
#     # for entirely true csvs filling in false bouts
#     sorted_arr = []
#     excluded_arr = []
#     count = 0
#     for i in range(mouse_num):
#     with open(directory+file) as json_file:
#         data = json.load(json_file)
#         labels = list(data['labels'].keys())
#         if len(labels) < mouse_num:
#                 print(file)
#                 #print(mouse_nums)
#                 sorted_labels = sorted(labels)
#                 sorted_labels = [eval(i) for i in sorted_labels]
#                 for i in range(mouse_num):
#                     if i not in sorted_labels:
#                         #print(i)
#                         #print(file)
#                         excluded_data = {'start': 0, 'end':  frame_num,'huddling': False,'mouse': i,'vid_name': "_".join(file.split('_')[:2]), 'jabs': True}
#                         excluded_arr.append(excluded_data)

#             #print(labels)
#         for i in labels:
#                 mouse = data['labels'][i]
#                 #print(mouse)
#                 for j in range(len(list(mouse.values())[0])):
#                     formatted_data = {'start': mouse['huddling'][j]['start'], 'end':  mouse['huddling'][j]['end'],'huddling': mouse['huddling'][j]['present'],'mouse': i,'vid_name': "_".join(file.split('_')[:2]), 'jabs': True}
#                     #print(formatted_data)
#                     sorted_arr.append(formatted_data)
#     with_false_arr = []
#     #print(sorted_arr[0])
#     #print(np.where(np.array(files) == sorted_arr[0]['vid_name'].split('.')[0])[i])
#     if sorted_arr[0]['start'] != 0:
#         with_false_arr.append({'start': 0, 'end':  sorted_arr[0]['start']-1,'huddling':False,'mouse': sorted_arr[0]['mouse'],'vid_name': sorted_arr[0]['vid_name'], 'jabs': True})
#     for i in range(1,len(sorted_arr)):
#         #print(sorted_arr[i])
#         with_false_arr.append(sorted_arr[i-1])
#         if sorted_arr[i]['vid_name'] == sorted_arr[i-1]['vid_name'] and sorted_arr[i]['mouse'] == sorted_arr[i-1]['mouse']:
#             # print(i)
#             # print(sorted_arr['mouse'][i-1])
#             # print(sorted_arr['start'][i-1])
#             with_false_arr.append({'start': sorted_arr[i-1]['end']+1, 'end':  sorted_arr[i]['start']-1,'huddling':False,'mouse': sorted_arr[i]['mouse'],'vid_name': sorted_arr[i]['vid_name'], 'jabs': True})
#         elif not sorted_arr[i]['mouse'] == sorted_arr[i-1]['mouse']:
#             num_frames = np.array(frame_nums)[np.where(np.array(files) == sorted_arr[0]['vid_name'].split('.')[0])[0]]-1
#             #print(num_frames)
#             #print(frame_nums[i])
#             if sorted_arr[i]['start'] != 0:
#                 with_false_arr.append({'start': 0, 'end':  sorted_arr[i]['start']-1,'huddling':False,'mouse': sorted_arr[i]['mouse'],'vid_name': sorted_arr[i]['vid_name'], 'jabs': True})
#             if sorted_arr[i]['end'] != frame_nums[count-1]:
#                 #print(frame_nums[count-1])
#                 #print(sorted_arr[i-1]['vid_name'])
#                 with_false_arr.append({'start': sorted_arr[i-1]['end']+1, 'end':  frame_nums[count-1],'huddling':False,'mouse': sorted_arr[i-1]['mouse'],'vid_name': sorted_arr[i-1]['vid_name'], 'jabs': True})
#         #print(with_false_arr)
#     with_false_arr.append(sorted_arr[i])
#     total = (pd.concat([pd.DataFrame(with_false_arr),(pd.DataFrame(excluded_arr))])).sort_values(by=['vid_name','mouse']).reset_index(drop=True)
#     #print(excluded_arr[0])
#     #print(total)
#     total.to_csv('gt_original.csv')

def full_dataset_ethogram(jabs,heuristic, max_frames):
    jabs_data = pd.read_csv(jabs, index_col=0)
    heur_data = pd.read_csv(heuristic, index_col=0)
    total_dataset = pd.concat([jabs_data, heur_data], ignore_index=True).reset_index(drop=True)
    print(total_dataset)
    vid_names = ['_'.join(i.split('_')) for i in total_dataset['vid_name']]
    factor_mouse = pd.factorize(vid_names)
    print(len(factor_mouse[0]))
    total_dataset['yax'] = factor_mouse[0]
    total_dataset.to_csv('total.csv')

    print(total_dataset)
    (p9.ggplot(total_dataset)+
	p9.geom_rect(p9.aes(xmin='start', xmax='end', ymin='yax + jabs/2', ymax='yax + jabs/2 + 0.5', fill='huddling'))+
	p9.theme_bw()+
	p9.facet_wrap('~mouse')+
	p9.scale_y_continuous(breaks=np.arange(len(factor_mouse[1]))+0.5, labels=factor_mouse[1])+
	p9.scale_fill_brewer(type='qual', palette='Set1')+
	p9.geom_hline(p9.aes(yintercept='y'), pd.DataFrame({'y':np.arange(len(factor_mouse[1])+1)}))+
	p9.labs(x='Frame', y='Video', title = "Comparison between JABS Ground Truth and Second Iteration Classifier", fill='huddling')
    ).save('total_gt_ethogram.png', height=6, width=12, dpi=300)

def gt_classifier_eval(path):
    data = pd.read_csv(path, index_col=0).reset_index(drop=True)
    heur_total = []
    jabs_total = []
    for i in range(max(data['yax'])+1):
        vid = data[data['yax'] == i].reset_index(drop=True)
        heur_vid_binary = np.zeros((max(vid['mouse'])+1)*(max(vid['end'])+1))
        jabs_vid_binary = np.zeros((max(vid['mouse'])+1)*(max(vid['end'])+1))
        print(max(vid['mouse'])+1)
        print(max(vid['end'])+1)
        print(heur_vid_binary.shape)
        jabs = vid[vid['jabs'] == True].reset_index(drop=True)
        heur = vid[vid['jabs'] == False].reset_index(drop=True)
        for mouse in range(max(vid['mouse'])+1):
            #print(mouse)
            jabs_mouse = jabs[jabs['mouse']==mouse]
            jabs_huddling_mouse = jabs_mouse[jabs_mouse['huddling']==True]
            heur_mouse = heur[heur['mouse']==mouse]
            heur_huddling_mouse = heur_mouse[heur_mouse['huddling']==True]
            for frame in range(max(vid['end'])+1):
                if frame in jabs_huddling_mouse['start'].values and not jabs_huddling_mouse.empty:
                    jabs_bouts = jabs_huddling_mouse[jabs_mouse['start']==frame]
                    jabs_vid_binary[(mouse*(max(vid['end'])+1) + jabs_bouts['start'].values[0]):(mouse*(max(vid['end'])+1) + jabs_bouts['end'].values[0])] = 1
                if frame in heur_huddling_mouse['start'].values and not heur_huddling_mouse.index.empty:
                    heur_bouts = heur_huddling_mouse[heur_huddling_mouse['start']==frame]
                    heur_vid_binary[(mouse*(max(vid['end'])+1) + heur_bouts['start'].values[0]):(mouse*(max(vid['end'])+1) + heur_bouts['end'].values[0])] = 1
        print(heur_vid_binary)
        heur_total.extend(list(heur_vid_binary))
        jabs_total.extend(list(jabs_vid_binary))
    print(heur_total)
    accuracy = accuracy_score(jabs_total, heur_total)
    print(accuracy)
        #print(np.where(np.sum(heur_vid_binary, axis=1)==1))
        #F1_score = f1_score(jabs_vid_binary,heur_vid_binary, average='micro')
        #print(F1_score)
        #heur_total.extend(heur_vid_binary)
        #jabs_total.extend(jabs_vid_binary)
    F1_score = f1_score(jabs_total,heur_total)
    print(F1_score)
    cm = confusion_matrix(jabs_total,heur_total)
    #disp = ConfusionMatrixDisplay(cm)
    #disp.plot()
    print(cm)
    sns.heatmap(cm/np.sum(cm), annot=True, 
            fmt='.2%')
    plt.title("F1 Score: " + str(F1_score))
    plt.savefig('total_confusion_gt.png')

def main(argv):
    heur = '/Users/szadys/jabs-postprocess/training_and_testing_data/heuristic_all_bouts.csv'
    jabs = '/Users/szadys/jabs-postprocess/jabs_gt_with_false.csv'
    mouse_nums = [4,3,3,3,3,3,3,3,3,3,3]
    frame_nums = [18000,18000,18000,18000,18000,18000,18000,18000,18000,18000,17990]
    #full_dataset_ethogram(heur, jabs, 18000)
    gt_classifier_eval('total.csv')
    #combine_csv('/Users/szadys/jabs-postprocess/training_and_testing_data/jabs_stitched_csvs/filter_775/')
    #jsons_to_csv('/Users/szadys/Desktop/gt_annotations/', mouse_nums, frame_nums)
    # jabs_bouts = 'jabs_gt_with_false.csv'
    # heuristic_bouts = 'heuristic_all_bouts.csv'
    # num_frames = 3600
    # num_vids = len(heur)
    # compare_classifiers(jabs, heur, num_frames, num_vids)
    # single_vid_compare_classifiers('/Users/szadys/jabs-postprocess/jabs_stitched_csvs/stitched_MDB0025_2021-01-10_13-00-00_huddling.csv','/Users/szadys/jabs-postprocess/heuristic_stitched_csvs/MDB0025_2021-01-10_13-00-00.csv', num_frames)

if __name__ == '__main__':
    main(sys.argv[1:])
