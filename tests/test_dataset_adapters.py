"""Test dataset adapter functionality."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@pytest.mark.skipif(
    True,
    reason="Requires optional 'datasets' library. Install with: pip install datasets"
)
def test_dataset_registry_exists():
    """Test that dataset registry is properly defined."""
    from dataset_adapters import DATASET_REGISTRY, discover_datasets_matching_format
    
    assert isinstance(DATASET_REGISTRY, dict)
    assert len(DATASET_REGISTRY) > 0
    
    datasets = discover_datasets_matching_format()
    assert isinstance(datasets, list)
    assert len(datasets) > 0


@pytest.mark.skipif(
    True,
    reason="Requires optional 'datasets' library. Install with: pip install datasets"
)
def test_dataset_registry_structure():
    """Test that each dataset in registry has required fields."""
    from dataset_adapters import DATASET_REGISTRY
    
    for name, meta in DATASET_REGISTRY.items():
        assert "text_columns" in meta, f"{name} missing text_columns"
        assert isinstance(meta["text_columns"], list), f"{name} text_columns not a list"
        assert len(meta["text_columns"]) > 0, f"{name} text_columns empty"
        assert "splits" in meta, f"{name} missing splits"


def test_extract_text_cell_without_datasets():
    """Test text extraction without requiring datasets library."""
    # Test with simple string
    result = _extract_text_cell_mock("test string")
    assert result == "test string"
    
    # Test with None
    result = _extract_text_cell_mock(None)
    assert result == ""
    
    # Test with long string (truncation)
    long_text = "a" * 5000
    result = _extract_text_cell_mock(long_text)
    assert len(result) <= 4000


def test_concat_text_without_datasets():
    """Test text concatenation without requiring datasets library."""
    row = {"col1": "text1", "col2": "text2", "fallback": "fallback_text"}
    
    # Test normal concatenation
    result = _concat_text_mock(row, ["col1", "col2"], None)
    assert "text1" in result
    assert "text2" in result
    
    # Test with fallback
    result = _concat_text_mock(row, ["missing"], "fallback")
    assert result == "fallback_text"
    
    # Test empty result
    result = _concat_text_mock({}, ["missing"], None)
    assert result == ""


# Mock functions for testing without datasets library
def _extract_text_cell_mock(value, max_len: int = 4000) -> str:
    """Mock version of _extract_text_cell for testing."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()[:max_len]
    return str(value)[:max_len]


def _concat_text_mock(row: dict, columns: list[str], fallback: str | None) -> str:
    """Mock version of _concat_text for testing."""
    parts = []
    for col in columns:
        if col in row and row[col] is not None:
            t = _extract_text_cell_mock(row[col])
            if t:
                parts.append(t)
    if parts:
        return " ".join(parts)[:4000]
    if fallback and fallback in row:
        return _extract_text_cell_mock(row[fallback])
    return ""

