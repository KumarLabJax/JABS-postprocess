# JABS Postprocess

JABS-postprocess is a comprehensive Python toolkit for analyzing behavioral data from 
the [JABS (JAX Animal Behavior System)](https://github.com/KumarLabJax/JABS-behavior-classifier) 
computer vision pipeline. Transform video pose estimations and behavior predictions 
into publication-ready tables and visualizations.

## Quick Start

### Prerequisites

- Python 3.10 or higher
- Access to JABS project data (pose and behavior prediction files)

### Installation

#### Option 1: From PyPI 

##### Using `venv` and `pip`
```bash

# Create and activate virtual environment
python3 -m venv jabs_postprocess_env
source jabs_postprocess_env/bin/activate  # On Windows: jabs_postprocess_env\Scripts\activate

# Install JABS-postprocess
pip install jabs-postprocess
```

##### Using `uv` (Recommended)
```bash

# Create project directory and install with uv
mkdir my_jabs_analysis && cd my_jabs_analysis
uv add jabs-postprocess
```

```bash

# Or install globally with uv
uv tool install jabs-postprocess
```

You can also use poetry, or any other Python package manager of your choice.

#### Option 2: With Containerization
##### Using Docker
```bash

docker pull aberger4/jabs-postprocess:latest
docker run -it --rm aberger4/jabs-postprocess:latest jabs-postprocess --help
```

#### Using Singularity
```bash

# Clone source code
git clone https://github.com/KumarLabJax/JABS-postprocess.git

# Build container
singularity build --fakeroot jabs-postprocess.sif vm/jabs-postprocess.def

# Run commands through container
singularity run jabs-postprocess.sif jabs-postprocess --help
```

#### Option 3: From source
```bash
# Clone source code
git clone https://github.com/KumarLabJax/JABS-postprocess.git

cd JABS-postprocess

uv sync
```


### Verify Installation
```bash

# Check if installation worked (adjust command based on your installation method)
jabs-postprocess --help                    # If installed with pip/uv tool
uv run jabs-postprocess --help             # If using uv project
poetry run jabs-postprocess --help         # If using poetry

# Should display available commands:
# - generate-tables
# - heuristic-classify  
# - create-snippet
# - evaluate-ground-truth
# - merge-tables
# - add-bout-statistics
```

## Key Concepts

Understanding these core concepts will help you effectively use JABS-postprocess:

### Data Organization
- **Project Folder**: Contains all data files for an experiment
- **Pose Files**: `*_pose_est_v*.h5` - Body part coordinates from computer vision
- **Behavior Files**: `*_behavior.h5` - ML classifier predictions for behaviors  
- **Feature Files**: `features.h5` - Optional engineered features (speed, distances, etc.)

### Behavior Analysis Pipeline
```
Raw Video → JABS Pose Detection → JABS Behavior Classification → JABS-Postprocess Tables → Analysis/Visualization
```

### Bout-Based Analysis
- **Bout**: Continuous sequence of the same behavior state
- **Behavior States**: 
  - `1`: Behavior detected
  - `0`: Not performing behavior
  - `-1`: Missing pose data (no prediction possible)

### Filtering Parameters
Applied sequentially to clean up predictions:

1. **Interpolate Size** (`--interpolate_size`): Fill gaps in missing data ≤ N frames
2. **Stitch Gap** (`--stitch_gap`): Merge behavior bouts separated by ≤ N frames  
3. **Min Bout Length** (`--min_bout_length`): Remove behavior bouts shorter than N frames

Example: With `--interpolate_size 5 --stitch_gap 10 --min_bout_length 30`:
```
Raw:      [Behavior-5frames] [Missing-3frames] [NotBehavior-8frames] [Behavior-15frames]
Step 1:   [Behavior-5frames] [Behavior-3frames] [NotBehavior-8frames] [Behavior-15frames]  # Interpolate
Step 2:   [Behavior-26frames]                                                              # Stitch (5+3+8+10=26)
Step 3:   []                                                                               # Filter out (26 < 30)
```

### Output Tables

**Bout Table** (`*_bouts.csv`): Raw bout-level data
- Each row = one behavioral bout
- Columns: animal_id, start_frame, duration, behavior_state, etc.

**Summary Table** (`*_summaries.csv`): Time-binned summaries  
- Each row = one time bin (default: 60 minutes)
- Columns: time_behavior, bout_count, distances_traveled, etc.

## Usage Examples

### Example 1: Basic Behavior Table Generation

Generate tables for grooming and locomotion behaviors from classifier predictions:

```bash

# Navigate to your data directory
cd /path/to/your/experiment

# Generate behavior tables for multiple behaviors
jabs-postprocess generate-tables \
    --project_folder ./my_experiment \
    --behavior grooming \
    --behavior locomotion \
    --out_prefix experiment_results \
    --out_bin_size 60 \
    --interpolate_size 5 \
    --stitch_gap 15 \
    --min_bout_length 30
```

**Expected Output:**
```
Generated tables for grooming:
  Bout table: experiment_results_grooming_bouts.csv
  Summary table: experiment_results_grooming_summaries.csv
  ✓ Includes bout statistics
  
Generated tables for locomotion:
  Bout table: experiment_results_locomotion_bouts.csv  
  Summary table: experiment_results_locomotion_summaries.csv
  ✓ Includes bout statistics
```

**What this does:**
- Processes all pose/behavior files in `./my_experiment`
- Creates 4 CSV files (2 per behavior)
- Fills missing data gaps ≤ 5 frames
- Merges behavior bouts separated by ≤ 15 frames
- Removes behavior bouts shorter than 30 frames
- Bins data into 60-minute summaries
- Adds statistics like bout count, average duration, etc.

### Example 2: Heuristic Classification Workflow

Use pose-based features to classify freezing behavior (alternative to ML classifiers):

```bash

# First, make sure you have exported features from JABS
# Then run heuristic classification

jabs-postprocess heuristic-classify \
    --project_folder ./fear_conditioning_experiment \
    --behavior_config freeze.yaml \
    --feature_folder features \
    --out_prefix freeze_analysis \
    --out_bin_size 30 \
    --min_bout_length 90
```

**Freeze behavior definition** (from `freeze.yaml`):
```yaml
# Mouse immobile for at least 3 seconds
definition:
  all:
   - less than:
     - features/per_frame/point_speeds BASE_NECK speed
     - 2.0  # pixels/frame
   - less than:
     - features/per_frame/point_speeds NOSE speed  
     - 2.0
   - less than:
     - features/per_frame/point_speeds BASE_TAIL speed
     - 2.0
```

**Expected Output:**
```
freeze_analysis_freeze_bouts.csv       # Individual freezing bouts
freeze_analysis_freeze_summaries.csv   # 30-minute time bins
```

### Example 3: Video Snippet Creation with Behavior Overlay

Extract video clips showing specific behaviors with pose and prediction overlays:

```bash
# Create 30-second clip starting at 5 minutes with behavior overlay
jabs-postprocess create-snippet \
    --input_video ./experiment_videos/mouse_session_2023-06-15_14-30-00.mp4 \
    --output_video ./clips/grooming_example.mp4 \
    --start 300 \
    --duration 30 \
    --time_units second \
    --pose_file ./experiment_data/mouse_session_2023-06-15_14-30-00_pose_est_v5.h5 \
    --behavior_file ./experiment_data/mouse_session_2023-06-15_14-30-00_behavior.h5 \
    --render_pose \
    --overwrite
```

**What this creates:**
- 30-second video clip (5:00-5:30)
- Pose keypoints overlaid on video
- Behavior predictions shown as colored regions
- Useful for validating classifier performance or creating figures

**Advanced Example - Extract Multiple High-Confidence Behavior Bouts:**
```bash
# First generate behavior tables to identify good examples
jabs-postprocess generate-tables \
    --project_folder ./social_experiment \
    --behavior approach \
    --out_prefix social_analysis

# Then examine the bout table to find interesting time periods
# Look for approach bouts with duration > 60 frames (2 seconds at 30fps)

# Extract specific bout (manually identified from bout table)
jabs-postprocess create-snippet \
    --input_video ./social_experiment/videos/pair_A_2023-08-10_16-45-22.mp4 \
    --output_video ./analysis_clips/approach_bout_example.mp4 \
    --start 1847 \
    --duration 180 \
    --time_units frame \
    --pose_file ./social_experiment/pair_A_2023-08-10_16-45-22_pose_est_v5.h5 \
    --behavior_file ./social_experiment/pair_A_2023-08-10_16-45-22_behavior.h5 \
    --render_pose
```

## Working with Generated Data

### Loading Data in Python

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load bout-level data
bouts_df = pd.read_csv('experiment_results_grooming_bouts.csv', skiprows=2)
print(f"Found {len(bouts_df)} grooming bouts")

# Load summary data  
summary_df = pd.read_csv('experiment_results_grooming_summaries.csv', skiprows=2)

# Basic analysis
print(f"Total grooming time: {bouts_df['duration'].sum()/30:.1f} seconds")  # Assuming 30 fps
print(f"Average bout duration: {bouts_df['duration'].mean():.1f} frames")

# Plot timeline
plt.figure(figsize=(12, 6))
plt.scatter(pd.to_datetime(summary_df['time']), summary_df['bout_behavior'])
plt.xlabel('Time')
plt.ylabel('Grooming Bouts per Hour')
plt.title('Grooming Behavior Over Time')
plt.show()
```

### Using Built-in Analysis Tools

```python
# Use JABS-postprocess plotting utilities
from jabs_postprocess.analysis_utils.plots import generate_time_vs_feature_plot
from jabs_postprocess.analysis_utils.parse_table import read_ltm_summary_table

# Read data with metadata
header_data, df = read_ltm_summary_table('experiment_results_grooming_summaries.csv')

# Create time-series plots
plot = generate_time_vs_feature_plot(
    df, 
    x_col='relative_exp_time',
    y_col='bout_behavior', 
    title=f"{header_data['Behavior'][0]} Analysis"
)
plot.draw().show()
```

## Available Heuristic Classifiers

Pre-built behavior classifiers using pose features:

| Behavior | Description | Key Features | Min Bout |
|----------|-------------|--------------|----------|
| **Locomotion** | Movement > 5 cm/s | `centroid_velocity_mag > 5.0` | 15 frames |
| **Freeze** | Immobility for ≥3 seconds | `neck_speed < 2.0 AND nose_speed < 2.0 AND tail_speed < 2.0` | 90 frames |
| **Wall Facing** | Oriented toward wall | Head direction + distance to wall | 5 frames |
| **Corner** | In corner region | Position in corner zones | 5 frames |
| **Periphery** | Along arena edges | Distance from center | 5 frames |

### Creating Custom Heuristic Classifiers

```yaml
# Example: custom_rearing.yaml
behavior: Rearing

interpolate: 5
stitch: 10  
min_bout: 45  # 1.5 seconds at 30fps

definition:
  all:
    - greater than:
      - features/per_frame/point_heights NOSE height
      - 50.0  # Nose elevated above baseline
    - less than:
      - features/per_frame/centroid_velocity_mag centroid_velocity_mag  
      - 3.0   # Minimal movement during rearing
```

```bash
# Use your custom classifier
jabs-postprocess heuristic-classify \
    --project_folder ./open_field_test \
    --behavior_config ./custom_rearing.yaml \
    --feature_folder features \
    --out_prefix rearing_analysis
```

## Common Issues and Solutions

### 1. "No behavior files found in project folder"

**Problem**: The generate-tables command can't find behavior prediction files.

**Solutions**:
```bash
# Check your project structure
ls -la /path/to/project/
# Look for files ending in _behavior.h5

# Common issues:
# - Wrong folder path
# - Files named differently
# - Files in subdirectories

# If files are in subdirectories, try:
find /path/to/project/ -name "*_behavior.h5" -type f
```

**Expected Structure**:
```
project_folder/
├── mouse_session_2023-06-15_14-30-00_pose_est_v5.h5
├── mouse_session_2023-06-15_14-30-00_behavior.h5
├── mouse_session_2023-06-15_15-30-00_pose_est_v5.h5  
├── mouse_session_2023-06-15_15-30-00_behavior.h5
└── ...
```

### 2. "Unknown behavior: grooming"

**Problem**: The behavior name doesn't match what's in the behavior files.

**Solutions**:
```bash
# Intentionally use wrong behavior to see available options
jabs-postprocess generate-tables \
    --project_folder ./my_project \
    --behavior wrong_name \
    --out_prefix test

# Error message will show available behaviors like:
# Available behaviors: ['approach', 'locomotion', 'rearing', 'grooming']

# Use exact spelling
jabs-postprocess generate-tables \
    --project_folder ./my_project \
    --behavior approach \
    --out_prefix results
```

### 3. "FileExistsError: Output file already exists"

**Problem**: Output files from previous runs exist.

**Solutions**:
```bash
# Option 1: Use --overwrite flag
jabs-postprocess generate-tables \
    --project_folder ./my_project \
    --behavior grooming \
    --out_prefix results \
    --overwrite

# Option 2: Change output prefix
jabs-postprocess generate-tables \
    --project_folder ./my_project \
    --behavior grooming \
    --out_prefix results_v2

# Option 3: Remove old files
rm results_grooming_*.csv
```

### 4. "Empty behavior table generated"

**Problem**: No behaviors detected after filtering.

**Solutions**:
```bash
# Try more permissive filtering
jabs-postprocess generate-tables \
    --project_folder ./my_project \
    --behavior grooming \
    --out_prefix debug \
    --interpolate_size 10 \    # Increase from default 5
    --stitch_gap 30 \          # Increase from default 5  
    --min_bout_length 5        # Decrease from default 5

# Check the raw predictions first
python -c "
import h5py
with h5py.File('path_to_behavior.h5', 'r') as f:
    predictions = f['preds'][:]
    print(f'Behavior predictions shape: {predictions.shape}')
    print(f'Unique values: {np.unique(predictions)}')
    print(f'Behavior frames: {np.sum(predictions > 0.5)}')
"
```

### 5. "KeyError: 'preds' when reading behavior file"

**Problem**: Behavior file structure is unexpected.

**Solutions**:
```bash
# Examine file structure
python -c "
import h5py
with h5py.File('problematic_file.h5', 'r') as f:
    print('Available keys:', list(f.keys()))
    for key in f.keys():
        print(f'{key}: {f[key].shape if hasattr(f[key], \"shape\") else \"group\"}')"
```

### 6. "Pose file and behavior file frame count mismatch"

**Problem**: Different number of frames between pose and behavior data.

**Solutions**:
- Check if files are from the same recording session
- Verify timestamps match exactly
- Some videos may have been truncated during processing

```bash
# Compare frame counts
python -c "
import h5py
with h5py.File('pose_file.h5', 'r') as f:
    pose_frames = f['poseest'][()].shape[0]
with h5py.File('behavior_file.h5', 'r') as f:  
    behavior_frames = f['preds'][()].shape[0]
print(f'Pose frames: {pose_frames}, Behavior frames: {behavior_frames}')
"
```

### 7. "No features found" for heuristic classification

**Problem**: Feature files missing or incomplete.

**Solutions**:
```bash
# Check if features were exported from JABS
ls -la features/
# Should see: features.h5, centroid_velocity.csv, point_speeds.csv, etc.

# Re-export features from JABS if missing
# See JABS documentation for feature export

# Check specific feature availability
python -c "
import h5py
with h5py.File('features/features.h5', 'r') as f:
    print('Available feature groups:', list(f.keys()))
    if 'per_frame' in f:
        print('Per-frame features:', list(f['per_frame'].keys()))
"
```

## Example Data and Configuration Files

### Sample Heuristic Configurations
- [Locomotion classifier](src/jabs_postprocess/heuristic_classifiers/locomotion.yaml) - Movement detection
- [Freeze classifier](src/jabs_postprocess/heuristic_classifiers/freeze.yaml) - Immobility detection  
- [Wall facing classifier](src/jabs_postprocess/heuristic_classifiers/wall_facing.yaml) - Spatial orientation
- [Corner classifier](src/jabs_postprocess/heuristic_classifiers/corner.yaml) - Corner preference
- [Periphery classifier](src/jabs_postprocess/heuristic_classifiers/periphery.yaml) - Thigmotaxis

### Example Analysis Scripts
- [Basic plotting examples](examples/test_plot.py) - Time-series visualization
- [ID matching examples](examples/test_id_matching.py) - Multi-animal tracking
- [Video sampling](examples/sample_uncertain_vids.py) - Quality control workflows

### Data Structure Examples

**Project Folder Layout**:
```
my_experiment/
├── videos/                          # Original videos (optional)
│   ├── mouse_A_2023-06-15_14-30-00.mp4
│   └── mouse_A_2023-06-15_15-30-00.mp4
├── mouse_A_2023-06-15_14-30-00_pose_est_v5.h5      # Pose data
├── mouse_A_2023-06-15_14-30-00_behavior.h5         # Behavior predictions
├── mouse_A_2023-06-15_15-30-00_pose_est_v5.h5
├── mouse_A_2023-06-15_15-30-00_behavior.h5
└── features/                        # Feature data (optional)
    ├── features.h5
    ├── centroid_velocity.csv
    └── point_speeds.csv
```

## Advanced Usage

### Batch Processing Multiple Experiments

```bash
#!/bin/bash
# Process multiple experiment folders
experiments=("experiment_1" "experiment_2" "experiment_3")
behaviors=("approach" "grooming" "locomotion")

for exp in "${experiments[@]}"; do
  for behavior in "${behaviors[@]}"; do
    echo "Processing $exp - $behavior"
    jabs-postprocess generate-tables \
      --project_folder "./$exp" \
      --behavior "$behavior" \
      --out_prefix "${exp}_${behavior}" \
      --overwrite
  done
done
```

### Merging Results Across Experiments

```bash
# Merge grooming results from multiple experiments
jabs-postprocess merge-multiple-tables \
    --table_folder ./results \
    --behaviors grooming \
    --table_pattern "*grooming_bouts.csv" \
    --output_prefix combined_grooming \
    --overwrite
```

### Performance Evaluation Against Ground Truth

```bash
# Evaluate classifier performance
jabs-postprocess evaluate-ground-truth \
    --behavior grooming \
    --ground_truth_folder ./manually_annotated_data \
    --prediction_folder ./classifier_predictions \
    --results_folder ./evaluation_results
```

### Development Installation

For development or accessing the latest features, you can install from source:

```bash
# Clone the repository
git clone https://github.com/KumarLabJax/JABS-postprocess.git
cd JABS-postprocess

# Option 1: UV development install
uv sync
uv run jabs-postprocess --help

# Option 2: Pip development install
python3 -m venv jabs_dev_env
source jabs_dev_env/bin/activate
pip install -e .

# Option 3: Poetry development install  
poetry install
poetry run jabs-postprocess --help
```

## Getting Help

- **Documentation**: See inline help with `jabs-postprocess COMMAND --help` (or use `uv run`/`poetry run` prefix based on your installation)
- **Issues**: Report problems at [GitHub Issues](https://github.com/KumarLabJax/JABS-postprocess/issues)
- **Examples**: Check the documentation and examples for working code patterns

## Citation

If you use JABS-postprocess in your research, please cite:

```bibtex
@software{jabs_postprocess,
  title = {JABS-postprocess: Behavioral Analysis Toolkit},
  author = {Kumar Lab, The Jackson Laboratory},
  url = {https://github.com/KumarLabJax/JABS-postprocess},
  year = {2024}
}
```
