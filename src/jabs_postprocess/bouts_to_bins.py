import sys
import re
import argparse
from pathlib import Path

from jabs_postprocess.utils.project_utils import BoutTable


def transform_bouts_to_bins(bout_file: Path, bin_size: int = 60):
	"""Transform a bout file into a summary table.

	Args:
		bout_file: Path to the bout file to transform
		bin_size: Size of the bins to use in the summary table
	"""
	bout_file = Path(bout_file)
	experiment_bout_data = BoutTable.from_file(bout_file)
	experiment_bin_data = experiment_bout_data.to_summary_table(bin_size)
	parts = bout_file.name.rsplit('_bouts', 1)
	filename = '_summaries'.join(parts)
	out_fname = bout_file.parent / filename
	experiment_bin_data.to_file(out_fname)
