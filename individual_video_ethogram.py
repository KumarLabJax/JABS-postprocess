import plotnine as p9
import pandas as pd
import sys
import os

# This file takes JABS pose annotation and extracts the labels, stitches bouts, creates behavior ethograms,
# and converts data into a csv of behavior bouts

# Run in a conda environment in order to have plotnine if running locally

def get_data(path):
    """Extracts the labeled behavior for each mouse from an annotation json
    Args:
        path: location of the json file found in the rotta/annotations folder of the project
    Returns: 
        bout_values: a list of the annotations, which contains nested dictionaries of the 
        behavior and the annotations for each mouse
    """
    file = open(path)
    data = pd.read_json(file).sort_index()
    bout_data = pd.DataFrame(data.values)
    bout_values = bout_data[2].values
    return bout_values

def json_to_dataframe(path, behavior, behavior_label):
    """Formats the json data into a dataframe of the needed data to make an ethogram
    Args:
        path: location of the json file found in the rotta/annotations folder of the project
        behavior: the behavior annotations of interest that matches the format of the json
        behavior_label: the formatted behavior name
    Returns: 
        bouts: a dataframe with dimensions [num_annotations x 4] with columns: 
        - start (frame # of beginning of bout)
        - end (frame # of end of bout) 
        - behavior_label (boolean of behavior or not behavior)
        - mouse (mouse label #)
    """
    bout_values = get_data(path)
    bouts = pd.DataFrame()
    data = []
    for i in range(len(bout_values)):
        for j in range(len(bout_values[i][behavior])):
            start_val = bout_values[i][behavior][j]['start']
            end_val = bout_values[i][behavior][j]['end']
            behavior_bool = bout_values[i][behavior][j]['present']
            bout = {'start': start_val, 'end':end_val, behavior_label:behavior_bool, "mouse":i}
            data.append(bout)
            bouts = pd.DataFrame(data)
    return bouts

def single_video_ethogram(path,behavior,video_name,behavior_label):
    """ Creates an ethogram for every mouse in the given video of the behavior of interest
    Args:
        path: location of the json file found in the rotta/annotations folder of the project
        behavior: the behavior annotations of interest that matches the format of the json
        video_name: the video of interest
        behavior_label: the formatted behavior name
    """
    bout_data = json_to_dataframe(path,behavior,behavior_label)
    (p9.ggplot(bout_data)+
     p9.geom_rect(p9.aes(xmin='start', xmax='end',ymin='mouse-0.25',ymax='mouse+0.25', fill=behavior_label))+
     p9.scale_fill_brewer(type='qual', palette='Set1')+
     p9.labs(x='Frame',y='Mouse Label', title='Sparsely Labeled Training Data for Huddling Classifier')).save("ethogram_"+video_name+".png", height=6, width=12, dpi=300)

def filtered_bout_ethogram(stitched_bouts, video_name, behavior_label):
    """ Creates an ethogram of the stitched behavior bouts for each mouse
    Args:
        stitched_bouts: a data frame that contains all of the bouts of the behavior of interest
        video_name: the info about video of interest (ex. MDX0606_2021-10-01_13-00-00) 
        behavior_label: the formatted behavior name
    """
    (p9.ggplot(stitched_bouts)+
     p9.geom_rect(p9.aes(xmin='start', xmax='end',ymin='mouse-0.25',ymax='mouse+0.25', fill=behavior_label))+
     p9.scale_fill_brewer(type='qual', palette='Set1')+
     p9.scale_x_continuous(limits=(0,3600))+
     p9.labs(x='Frame',y='Mouse Label', title='Stitched Training Data for Huddling Classifier')).save("stitched_ethogram_"+video_name+".png", height=6, width=12, dpi=300)

def no_identity_ethogram_filtered_ethogram(stitched_bouts, video_name, behavior_label):
    """ Creates an ethogram of the stitched behavior bouts for the total behavior regardless of identity
    Args:
        stitched_bouts: a data frame that contains all of the bouts of the behavior of interest
        video_name: the info about video of interest (ex. MDX0606_2021-10-01_13-00-00) 
        behavior_label: the formatted behavior name
    """
    (p9.ggplot(stitched_bouts)+
     p9.geom_rect(p9.aes(xmin='start', xmax='end',ymin='0.75',ymax='1.25', fill=behavior_label))+
     p9.scale_fill_brewer(type='qual', palette='Set1')+
     p9.scale_y_continuous(limits=(0,2))+
     p9.scale_x_continuous(limits=(0,3600))+
     p9.labs(x='Frame',y='Mouse Label', fill=behavior_label, title='Sparsely Labeled Training Data for Huddling Classifier')).save(video_name+"/no_identity_stitched_ethogram_"+video_name+"_775.png", height=6, width=12, dpi=300)


def behavior_bouts_to_csv(behavior_bouts, file_name, behavior_label):
    """ Writes a csv file that filters and stitches all the existing behavior bouts
    Args:
        behavior_bouts: a dataframe 
        file_name: the name of the output csv file
        behavior_label: the formatted behavior label
    Returns:
        stitched_bouts: a data frame that contains all of the bouts of the behavior of interest
    """
    stitched_bouts = []
    prev_bout = {'start': behavior_bouts.iloc[0]['start'], 'end': behavior_bouts.iloc[0]['end'], behavior_label: behavior_bouts.iloc[0][behavior_label], 'mouse':behavior_bouts.iloc[0]['mouse'], "vid_name":"_".join(file_name.split("_")[4:7])}
    for i in range(1,len(behavior_bouts)):
        bout = behavior_bouts.iloc[i] 
        if prev_bout['mouse'] == bout['mouse'] and prev_bout[behavior_label] == bout[behavior_label]:
            if bout['start'] - prev_bout['end'] < 375:
                prev_bout =  {'start': prev_bout['start'], 'end': bout['end'], behavior_label: prev_bout[behavior_label], "mouse":prev_bout['mouse'], "vid_name":"_".join(file_name.split("_")[4:7])}
            else:
                if prev_bout['end']-prev_bout['start']>775:
                    print(prev_bout['end']-prev_bout['start'])
                    stitched_bouts.append(prev_bout)
                prev_bout = {'start': bout['start'], 'end': bout['end'], behavior_label: bout[behavior_label], "mouse":bout['mouse'], "vid_name":"_".join(file_name.split("_")[4:7])}
        else:
            if prev_bout['end']-prev_bout['start']>775:
                #print(prev_bout['end']-prev_bout['start'])
                stitched_bouts.append(prev_bout)
            prev_bout = {'start': bout['start'], 'end': bout['end'], behavior_label: bout[behavior_label], "mouse":bout['mouse'], "vid_name":"_".join(file_name.split("_")[4:7])}
    if prev_bout['end']-prev_bout['start']>775:
        stitched_bouts.append(prev_bout)
    print(stitched_bouts)
    pd.DataFrame(stitched_bouts).to_csv(file_name)
    return pd.DataFrame(stitched_bouts)

def main(argv):
    behavior = 'huddling'
    behavior_label = 'Huddling Behavior'
    path = '/Users/szadys/Desktop/MiceVideos/rotta/annotations/'
    pose_files = [ f for f in os.listdir(path) if f.endswith('.json')]
    for vid in pose_files:
        video_path = path + vid
        bout_data = json_to_dataframe(video_path,behavior,behavior_label)
        behavior_bouts = pd.DataFrame(bout_data)
        if behavior_bouts.empty == False:
            stitched_bouts = behavior_bouts_to_csv(behavior_bouts, 'jabs_stitched_csvs/filter_775/stitched_' + str(vid.split('.')[0]) + '_' + behavior + '_775.csv', behavior_label)
            if stitched_bouts.empty == False:
                no_identity_ethogram_filtered_ethogram(stitched_bouts,str(vid.split('.')[0]), behavior_label)

    #single_video_ethogram(path+'MDB0025_2021-01-10_13-00-00.json', 'huddling', 'MDB0025_2021-01-10_13-00-00', )

if __name__ == '__main__':
    main(sys.argv[1:])