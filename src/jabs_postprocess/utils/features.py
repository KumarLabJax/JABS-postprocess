import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from jabs_postprocess.utils.metadata import VideoMetadata


class JABSFeature:
    """Methods to interact with JABS feature data."""

    def __init__(self, feature_file: Path):
        """Initializes a feature object.

        Args:
                feature_file: file to interact with feature data
        """
        assert os.path.exists(feature_file)

        # Transforms feature file into a metadata object
        # Note that this expects /path/to/feature/files/[POSE_FILE]/[ANIMAL_IDX]/features.h5 as the structure
        pose_file = str(Path(*Path(feature_file).parts[:-2])) + ".h5"
        self._video_metadata = VideoMetadata(pose_file)
        # TODO: this is an unhandled string to int casting
        self._animal_idx = int(Path(feature_file).parts[-2])
        self._file = feature_file
        # Populate keys generates a pandas table to better index which features are available
        self._populate_keys()

    @property
    def video_metadata(self):
        return self._video_metadata

    @property
    def animal_idx(self):
        return self._animal_idx

    @property
    def feature_keys(self):
        """Dataframe containing the available feature keys."""
        return self._feature_keys.copy()

    def get_key_data(self, key: str):
        """Retrieves raw data contained within a feature file.

        Args:
                key: fully defined key to extract from the feature file

        Returns:
                np.ndarray containing the requested data

        TODO:
                Have this function check for valid keys before crashing
        """
        found_data = False
        with h5py.File(self._file, "r") as f:
            if key in f:
                retrieved_data = f[key][...]
                found_data = True

        # Try and search for a simplified key
        if not found_data:
            raise ValueError(f"Full keys only supported currently. Given {key}")

        return retrieved_data

    def get_window_feature(
        self,
        feature_key: str,
        window_size: int,
        window_op: str,
        feature_module: str = None,
    ):
        """Retrieves the stored feature vector from a window feature.

        Args:
                feature_key: key of the feature to retrieve
                window_size: window size of the window feature
                window_op: window operation for the feature
                feature_module: Optional module which the features belong to

        Returns:
                np.ndarray containing the feature data

        Raises:
                KeyError if the feature key, window size, or window op are not present
        """
        if feature_module is None:
            feature_module = self.discover_feature_module(feature_key)

        if "window_op" not in self._feature_keys:
            raise KeyError(
                f"Feature file {self._file} does not contain window feature data."
            )
        if not (
            (self._feature_keys["module"] == feature_module)
            & (self._feature_keys["window_op"] == window_op)
            & (self._feature_keys["feature"] == feature_key)
        ).any():
            raise KeyError("Module, Window Op, and Feature key not available.")

        feature_key = f"features/window_features_{window_size}/{feature_module} {window_op} {feature_key}"

        with h5py.File(self._file, "r") as f:
            feature_data = f[feature_key][:]

        return feature_data

    def discover_feature_module(self, feature_key: str):
        """Discovers the feature module given a key.

        Args:
                feature_key: feature to identify the module name

        Returns:
                module string of the feature

        Raises:
                ValueError if module is not unique
        """
        discovered_feature_module = np.unique(
            self._feature_keys.loc[
                self._feature_keys["feature"] == feature_key, "module"
            ]
        )
        if len(discovered_feature_module) != 1:
            raise ValueError(
                f"Feature {feature_key} does not map uniquely to a feature module. Found modules: {discovered_feature_module}"
            )
        return discovered_feature_module[0]

    def _populate_keys(self):
        """Populates the available module-feature pairs."""
        with h5py.File(self._file, "r") as f:
            feature_grps = list(f["features"].keys())
            available_features = list(f["features/per_frame"].keys())

        base_features = pd.DataFrame(
            [["features/per_frame/" + x] + x.split(" ", 1) for x in available_features],
            columns=["key", "module", "feature"],
        )

        self._window_sizes = [
            x.split("_")[2] for x in feature_grps if x.startswith("window_features_")
        ]
        discovered_window_keys = []
        if len(self._window_sizes) > 0:
            with h5py.File(self._file, "r") as f:
                window_keys = list(
                    f[f"features/window_features_{self._window_sizes[0]}"].keys()
                )
            for cur_window in self._window_sizes:
                next_window_keys = pd.DataFrame(
                    [
                        [f"features/window_features_{cur_window}/{x}"]
                        + x.split(" ", 1)
                        + [cur_window]
                        for x in window_keys
                    ],
                    columns=["key", "module", "feature", "window_size"],
                )
                discovered_window_keys.append(next_window_keys)

        self._feature_keys = pd.concat([base_features] + discovered_window_keys)
