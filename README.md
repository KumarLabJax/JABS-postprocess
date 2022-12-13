# Installation

This code contains a singularity definition file for assistance with installing the python environment.

# Running

Example Call:

```
python3 generate_behavior_tables.py --project_folder /path/to/project/folder --out_prefix results
```

This will generate 2 behavior table files per behavior detected in the project folder. Optionally, you can include `--behavior BehaviorName` to only generate a behavior table for that one behavior.

# Data table format

There are two behavior tables generated. Both contain a header line to store parameters used while calling the script.

## Header Data

Stores data pertaining to the script call that globally addresses all data within the table.

* `Project Folder` : The folder the script searched for data
* `Behavior` : The behavior the script parsed
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
  * Bouts are counted by the proportion in which bouts are contained in a time bin. If a bout spans multiple time bins, it will be divided into both via the proportion of time.
  * Note that bouts cannot span between video files

# Example Plotting Code

Since the data is in a "long" format, it is generally straight forward to generate plots using ggplot in R or plotnine in python.

Some example code for generating plots is located in [test_plot.py](test_plot.py).