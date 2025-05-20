# Installation

## Singularity Container

This code contains a [singularity definition file](vm/JABS-postprocess.def) for assistance with installing the python environment. This environment supports both generating behavior table files and plotting the data in python.

Example building of the singularity image:

```
cd vm
singularity build --fakeroot ../../JABS-Postprocessing.sif JABS-postprocess.def
```

### `uv` Based Image
The `uv` based singularity image should be built from the repository root.
```
singularity build JABS-postprocess-uv.sif vm/JABS-postprocess-uv.def
```

## Virtual Environment

This project uses [`uv`](https://docs.astral.sh/uv/) for managing dependencies. You can use it to set up your virutal environment.

From the root of the repository, run:
```
uv sync
```

### Pip Based Virtual Environment
If you must use pip, you can create a virtual environment by running:

```python3 -m venv postprocess_venv
source postprocess_venv/bin/activate
pip3 install -r vm/requirements.txt
```

Only python3.10 has been tested.

# Generating Behavior Tables

## Classifier-based Table Generation

**Note**: When using a uv based environment, add `uv run` before each python command.

```
python3 generate_behavior_tables.py \
    --project_folder /path/to/project/folder/ \
    --out_prefix results \
    --behavior Behavior_1 \
    --behavior Behavior_2
```

This will generate 2 behavior table files per behavior detected in the project folder. You must include `--behavior BehaviorName` to generate a behavior table for each behavior. If you are unsure which behavior are available in a given project folder, you can check by intentionally guessing incorrectly.

To see all options with a short description, run:

```
python3 generate_behavior_tables.py --help
```

## JABS-feature-based Table Generation

```
python3 heuristic_classify.py --behavior Heuristic_Behavior --project_folder /path/to/project/folder/ --feature_folder /path/to/project/features/ --feature_key 'BASE_NECK speed' --relation '<' --threshold '1'
```

This will generate 2 behavior table files based on the threshold applied to the feature. Additional `--feature_key <key> --relation <relation> --threshold <threshold>` can be used in succession to indicate all conditions at the same time (e.g. `feature_1 < threshold_1 AND feature_2 > threshold_2`).

To see all options with a short description, run:
```
python3 heuristic_classify.py --help
```

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

Some features are optional, because calculating them can be expensive. These options are noted with an asterisk (\*). While default behavior is to include them, they are not guaranteed.

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

* `animal_idx` : Animal index in the pose file (typically not used, see pose file documentation for indexing rules)
* `longterm_idx` : Identity of the mouse in the experiment (-1 reserved for unlinked animals, animals in experiment are index 0+)
* `exp_prefix` : Detected experiment ID
* `time` : Formatted time string in "%Y-%m-%d %H:%M:%S" of the time this bout was extracted from
* `video_name` : Name of the video this bout was extracted from
* `start` : Start in frame of the bout
* `duration` : Duration of the bout in frames
* `is_behavior` : State of the described bout
    * `-1` : The mouse did not have a pose to create a prediction
    * `0` : Not behavior prediction
    * `1` : Behavior prediction
* `distance`\* : Distance traveled during bout

## Binned Table

The binned table contains summaries of the results in time-bins.

Summaries included:

* `longterm_idx` : Identity of the mouse in the experiment (-1 reserved for unlinked animals, animals in experiment are index 0+)
* `exp_prefix` : Detected experiment ID
* `time` : Formatted time string in "%Y-%m-%d %H:%M:%S" of the time bin
* `time_no_pred` : Count of frames where mouse did not have a predicted pose (missing data)
* `time_not_behavior` : Count of frames where mouse is not performing the behavior
* `time_behavior` : Count of frames where mouse is performing the behavior
* `bout_behavior` : Number of bouts where the mouse is performing the behavior
    * Bouts are counted by the proportion in which bouts are contained in a time bin
    * If a bout spans multiple time bins, it will be divided into both via the proportion of time
    * Sum of bouts across bins produces the correct total count
    * Note that bouts cannot span between video files
* `not_behavior_dist`\* : Total distance traveled during not behavior bouts
* `behavior_dist`\* : Total distance traveled during behavior bouts

# Example Plotting Code

Since the data is in a "long" format, it is generally straight forward to generate plots using ggplot in R or plotnine in python.

Some example code for generating plots is located in [test_plot.py](test_plot.py).

Additionally, there are a variety of helper functions located in [analysis_utils](analysis_utils/) for reading, manipulating, and generating plots of data using the data tables produced.

# Dense Ground Truth Performance Scripts

These scripts are still in the prototyping phase, but example methods of comparing predictions with a JABS annotated set of videos are available in [compare_gt.py](compare_gt.py)

# Video Clip Extraction

[create_video_snippets.py](create_video_snippets.py) will create video snippets based on an input video. Optionally, can render behavior predictions (either globally or per-mouse).

[sample_uncertain_vids.py](sample_uncertain_vids.py) utilizes a project folder and predictions. This script requires the complete project used by JABS alongside the predictions generated.

For both of these scripts, check the `--help` function for available filters.
