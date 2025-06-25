# Plotting Instructions

This folder contains scripts for generating plots from behavioral summary data.

## How to Run

To generate strain-level plots, you will need the following information:

1. **Behavior Name**
   - The name of the behavior you want to analyze (e.g., `Feeding`, `Drinking`, etc.).

2. **Path to a Summary CSV File**
   - This file is generated using the `generate_behavior_tables` script.
   - It contains the summary statistics for the behavior of interest.

3. **Metadata File from Tom**
   - This is an Excel file containing experiment metadata (e.g., animal IDs, strain, sex, room, etc.).
   - There are older versions of this file around the HPC. You can use the default one, but if you are analyzing experiments more recent than the date of this file (2023-09-07), you will not find them here.

## Example Command

```bash
python3 plots/generate-strain-plots.py \
  --behavior Feeding \
  --results_file /path/to/behavior_Feeding_summaries.csv \
  --jmcrs_data /path/to/metadata_file.xlsx \
  --remove_experiments MDB0003,MDX0008 \
  --output_dir /path/to/output/plots/
```

- `--behavior`: Name of the behavior (string)
- `--results_file`: Path to the summary CSV file
- `--jmcrs_data`: Path to the metadata Excel file
- `--remove_experiments`: (Optional) Comma-separated list of experiment IDs to exclude
- `--output_dir`: Output directory for all plot files (will be created if it does not already exist)

## Notes
- Make sure you have the correct Python environment with all dependencies installed (see the main project README for details).
- If you encounter issues with missing experiments in the metadata, check if your metadata file is up to date.
