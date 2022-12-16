# Installation

This code contains a [singularity definition file](vm/JABS-postprocess.def) for assistance with installing the python environment. This environment supports both generating behavior table files and plotting the data in python.

Example building of the singularity image:

```
cd vm
singularity build --fakeroot ../../JABS-Postprocessing.sif JABS-postprocess.def
```

Optionally, you can also just use `pip3 install -r vm/requirements.txt` with a python environment. Only python3.9 has been tested.

# Generating Behavior Tables

## Multi-Animal Multi-Day Table Generation

```
python3 generate_behavior_tables.py --project_folder /path/to/project/folder --out_prefix results
```

This will generate 2 behavior table files per behavior detected in the project folder. Optionally, you can include `--behavior BehaviorName` to only generate a behavior table for that one behavior.

## Single Animal OFA Table Generation

This feature is not yet implemented.

## Notes on Filtering

We apply filtering in 3 sequential stages using optional parameters.
All filtering is applied on bout-level data targeted at removing bouts in 1 of the 3 states (missing prediction, not behavior, behavior). When a block gets deleted, the borders are compared. When the borders match, the bout of the surrounding predictions get merged. When they borders mismatch, 50% of the deleted block gets assigned to each bordering. If the deleted duration is not divisible by 2, the later bout receives 1 more frame.

* `--max_interpolate_size` : The maximum size to delete missing predictions. Missing predictions below this size are assigned 50% to each of its bordering prediction blocks.
* `--stitch_gap` : The maximum size to delete not behavior predictions. This essentially attempts to merge together multiple behavior bouts.
* `--min_bout_length` : The maximum size to delete behavior predictions. Any behavior bout shorter than this length gets deleted.

The order of deletions is:

1. Missing Predictions
2. Not Behavior
3. Behavior

## Data table extensions

Lots of the functions used in generating these behavior tables were designed for potential re-use. Check out the functions inside [jabs_utils](jabs_utils/) if you wish to possibly extend the functionality of the generate behavior scripts.

# Data table format

There are two behavior tables generated. Both contain a header line to store parameters used while calling the script.

## Header Data

Stores data pertaining to the script call that globally addresses all data within the table.

* `Project Folder` : The folder the script searched for data
* `Behavior` : The behavior the script parsed
* `Interpolate Size` : The number of frames where missing data gets interpolated
* `Stitch Gap` : The number of frames in which predicted bouts were merged together
* `Min Bout Length` : The number of frames in which short bouts were omitted (filtered out)
* `Out Bin Size` : Time duration (minutes) used in binning the results

## Bout Table

The bout table contains a compressed RLE encoded format for each bout (post-filtering)

* `animal_idx` : Animal index in the pose file (typically not used)
* `longterm_idx` : Identity of the mouse in the experiment
* `exp_prefix` : Detected experiment ID
* `time` : Formatted time string in "%Y-%m-%d %H:%M:%S" of the time this bout was extracted from
* `video_name` : Name of the video this bout was extracted from
* `start` : Start in frame of the bout
* `duration` : Duration of the bout in frames
* `is_behavior` : State of the described bout
    * `-1` : The mouse did not have a pose to create a prediction
    * `0` : Not behavior prediction
    * `1` : Behavior prediction

## Binned Table

The binned table contains summaries of the results in time-bins.

Summaries included:

* `longterm_idx` : Identity of the mouse in the experiment
* `exp_prefix` : Detected experiment ID
* `level_0` : Formatted time string in "%Y-%m-%d %H:%M:%S" of the time bin
* `time_no_pred` : Count of frames where mouse did not have a predicted pose (missing data)
* `time_not_behavior` : Count of frames where mouse is not performing the behavior
* `time_behavior` : Count of frames where mouse is performing the behavior
* `bout_behavior` : Number of bouts where the mouse is performing the behavior
    * Bouts are counted by the proportion in which bouts are contained in a time bin
    * If a bout spans multiple time bins, it will be divided into both via the proportion of time
    * Sum of bouts across bins produces the correct total count
    * Note that bouts cannot span between video files

# Example Plotting Code

Since the data is in a "long" format, it is generally straight forward to generate plots using ggplot in R or plotnine in python.

Some example code for generating plots is located in [test_plot.py](test_plot.py).

Additionally, there are a variety of helper functions located in [analysis_utils](analysis_utils/) for reading, manipulating, and generating plots of data using the data tables produced.