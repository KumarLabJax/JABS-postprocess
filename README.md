# Inputs

1. Folder with all classifier predictions
2. Behavior

# Outputs

* Long format, 2 files
* Header at the top of each containing options during generation
* * Options:
* * * Filters (stitch + ignore)
* * * Bin size
* * * Identity merge tolerance
* * * Frames per bin
* Data will have header fields (input from Jaycee and Gautam on exact fields)
* * Must contain the following:
* * * [Folder, Experiment, Time Bin, AnimalID, Behavior, [Results (to be determined)]]
* Handling missing data
* * Pad with NAs for the maximum animal count per experiment
* * Number of animals in an experiment
* * * mean of predictions/hr -> cast to int
* * * optional input (max animals)
* * * require minimum number of frames to be "real"
* Handling incomplete data (missing poses/predictions on frames)
* * Field for number of frames alive

# Code Location
* Contains the "generate table" and plotting code starting location
