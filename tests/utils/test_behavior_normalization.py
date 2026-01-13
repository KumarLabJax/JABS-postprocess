"""Tests for behavior name normalization and resolution."""

import json
from pathlib import Path

import h5py
import numpy as np

from jabs_postprocess.utils.project_utils import (
    BoutTable,
    ClassifierSettings,
    Prediction,
    normalize_behavior_name,
    resolve_behavior_key,
)


def test_normalize_behavior_name_matches_pipeline_rules():
    assert normalize_behavior_name("rearing (unsupported)") == "rearing_unsupported"
    assert normalize_behavior_name("Rearing_unsupported") == "rearing_unsupported"


def test_resolve_behavior_key_normalized_match():
    available = ["rearing (unsupported)", "grooming"]
    assert (
        resolve_behavior_key("Rearing_unsupported", available)
        == "rearing (unsupported)"
    )


def test_resolve_behavior_key_ambiguous_match():
    available = ["A B", "A_B"]
    assert resolve_behavior_key("a b", available) is None


def test_from_jabs_annotation_file_uses_normalized_behavior(tmp_path: Path):
    annotation = {
        "file": "video1.avi",
        "labels": {
            "0": {
                "rearing (unsupported)": [
                    {"start": 0, "end": 2, "present": 1}
                ]
            }
        },
    }
    annotation_path = tmp_path / "annotation.json"
    annotation_path.write_text(json.dumps(annotation))

    table = BoutTable.from_jabs_annotation_file(
        annotation_path, "Rearing_unsupported"
    )
    df = table.data

    assert len(df) == 1
    row = df.iloc[0]
    assert row["start"] == 0
    assert row["duration"] == 3
    assert row["is_behavior"] == 1
    assert row["video_name"] == "video1"


def test_generate_bout_table_uses_normalized_behavior(tmp_path: Path):
    pred_path = tmp_path / "predictions.h5"
    with h5py.File(pred_path, "w") as h5f:
        pred_group = h5f.create_group("predictions")
        behavior_group = pred_group.create_group("rearing (unsupported)")
        behavior_group.create_dataset(
            "predicted_class", data=np.array([[0, 1, 1, 0]], dtype=np.int8)
        )

    settings = ClassifierSettings("Rearing_unsupported", 0, 0, 0)
    df = Prediction.generate_bout_table(pred_path, settings)

    match = df[
        (df["start"] == 1) & (df["duration"] == 2) & (df["is_behavior"] == 1)
    ]
    assert len(match) == 1
