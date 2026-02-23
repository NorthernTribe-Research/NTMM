"""
Adapters to load HuggingFace datasets and return a DataFrame with columns: text, label.
Each adapter yields (text, label) rows; label is int in [0, num_classes).
"""
from __future__ import annotations

import ast
import re
from typing import Any

import pandas as pd
from datasets import load_dataset


# Known medical datasets that match (text, label) format or can be normalized.
# Keys are HF dataset names; value: text_column, label_column (or None = derive from hash/source).
DATASET_REGISTRY = {
    "NorthernTribe-Research/comprehensive-healthbench-v2": {
        "text_columns": ["instruction"],
        "fallback_text": "response",
        "label_column": "source",
        "splits": ["train"],
    },
    "TimSchopf/medical_abstracts": {
        "text_columns": ["medical_abstract"],
        "label_column": "condition_label",
        "splits": ["train", "test"],
    },
    "BI55/MedText": {
        "text_columns": ["Prompt"],
        "fallback_text": "Completion",
        "label_column": None,
        "splits": ["train"],
    },
    "eswardivi/medical_qa": {
        "text_columns": ["instruction", "input"],
        "label_column": None,
        "splits": ["train"],
    },
    "chenhaodev/medmcqa_instruct": {
        "text_columns": ["instruction"],
        "label_column": None,
        "splits": ["train"],
    },
}


def _extract_text_cell(value: Any, max_len: int = 4000) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return ""
        if s.startswith("[") and "content" in s:
            try:
                parts = ast.literal_eval(s)
                if isinstance(parts, list) and len(parts) > 0:
                    if isinstance(parts[0], dict) and "content" in parts[0]:
                        return (parts[0].get("content") or "").strip()[:max_len]
                    if isinstance(parts[0], str):
                        return parts[0].strip()[:max_len]
            except (ValueError, SyntaxError):
                pass
            m = re.search(r"['\"]content['\"]\s*:\s*['\"]([^\"]*)['\"]", s)
            if m:
                return (m.group(1).strip() or s)[:max_len]
        return s[:max_len]
    if isinstance(value, list) and len(value) > 0:
        first = value[0]
        if isinstance(first, dict) and "content" in first:
            return (first.get("content") or "")[:max_len]
        if isinstance(first, str):
            return first[:max_len]
    return str(value)[:max_len]


def _concat_text(row: dict, columns: list[str], fallback: str | None) -> str:
    parts = []
    for col in columns:
        if col in row and row[col] is not None:
            t = _extract_text_cell(row[col])
            if t:
                parts.append(t)
    if parts:
        return " ".join(parts)[:4000]
    if fallback and fallback in row:
        return _extract_text_cell(row[fallback])
    return ""


def load_and_normalize(
    hf_name: str,
    num_classes: int,
    max_samples: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Load a HuggingFace dataset by name, normalize to (text, label), return DataFrame.
    """
    if hf_name not in DATASET_REGISTRY:
        raise ValueError(
            "Unknown dataset '{}'. Supported: {}".format(
                hf_name, list(DATASET_REGISTRY.keys())
            )
        )
    meta = DATASET_REGISTRY[hf_name]
    text_cols = meta["text_columns"]
    label_col = meta.get("label_column")
    fallback = meta.get("fallback_text")
    splits = meta.get("splits", ["train"])

    rows: list[dict[str, Any]] = []
    for split in splits:
        try:
            ds = load_dataset(hf_name, split=split)
        except Exception as e:
            raise RuntimeError("Failed to load {} split '{}': {}".format(hf_name, split, e))

        if label_col and label_col in ds.column_names:
            unique_labels = sorted(ds.unique(label_col))
            n = max(1, min(num_classes, len(unique_labels)))
            label_to_idx = {v: i % n for i, v in enumerate(unique_labels)}
        else:
            label_to_idx = {}

        for i in range(len(ds)):
            row = ds[i]
            text = _concat_text(row, text_cols, fallback)
            if not text:
                continue
            if label_col and label_col in row:
                raw = row[label_col]
                if isinstance(raw, (int, float)):
                    label = int(raw) % num_classes
                else:
                    label = label_to_idx.get(raw, hash(text) % num_classes)
            else:
                label = hash(text) % num_classes
            rows.append({"text": text[:4000], "label": label})

    if not rows:
        raise RuntimeError("No rows produced for {}".format(hf_name))
    df = pd.DataFrame(rows)
    if max_samples is not None and max_samples > 0 and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=seed, replace=False).reset_index(drop=True)
    return df


def discover_datasets_matching_format() -> list[str]:
    """Return list of registered dataset names that yield (text, label)."""
    return list(DATASET_REGISTRY.keys())
