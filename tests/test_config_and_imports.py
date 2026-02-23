"""Validate project config and core imports (no GPU required)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_mcp_json_exists_and_valid():
    config_path = PROJECT_ROOT / "mcp.json"
    assert config_path.exists(), "mcp.json not found"
    with config_path.open() as f:
        config = json.load(f)
    assert "paths" in config
    assert "train_data" in config["paths"]
    assert "validation_data" in config["paths"]
    assert "test_data" in config["paths"]
    assert "teacher_model_path" in config["paths"]
    assert "student_model_path" in config["paths"]
    assert "teacher_model" in config
    assert "student_model" in config
    assert "training_params" in config
    assert "distillation_params" in config
    assert "kaggle" not in config


def test_datasets_registry_matches_project_format():
    """Registered datasets must yield (text, label) compatible with our CSVs."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    try:
        from dataset_adapters import discover_datasets_matching_format, DATASET_REGISTRY
    except ModuleNotFoundError as e:
        pytest.skip("dataset_adapters deps not installed (e.g. datasets): {}".format(e))
    names = discover_datasets_matching_format()
    assert len(names) >= 1
    for name in names:
        assert name in DATASET_REGISTRY
        meta = DATASET_REGISTRY[name]
        assert "text_columns" in meta
        assert "splits" in meta


def test_data_csv_schema_if_present():
    """If train.csv exists, it must have columns text and label."""
    train_csv = PROJECT_ROOT / "data" / "train.csv"
    if not train_csv.exists():
        pytest.skip("data/train.csv not found (run prepare_data first)")
    import pandas as pd
    df = pd.read_csv(train_csv, nrows=10)
    assert "text" in df.columns and "label" in df.columns
    assert df["label"].dtype in (int, "int64", "int32")


@pytest.mark.skipif(
    __import__("sys").version_info < (3, 10),
    reason="Project requires Python 3.10+",
)
def test_distillation_utils_import():
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    try:
        from distillation_utils import compute_classification_metrics, DistillationTrainer
    except ImportError as e:
        pytest.skip("Core deps not installed (e.g. torch): {}".format(e))
    assert callable(compute_classification_metrics)
    assert DistillationTrainer is not None
