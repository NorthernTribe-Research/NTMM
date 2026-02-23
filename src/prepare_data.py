"""
Download one or more medical datasets, normalize to (text, label), merge and split
into train/validation/test CSVs. Output: data/train.csv, data/validation.csv, data/test.csv
with columns: text, label.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(__file__))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare train/validation/test CSVs from one or more HF datasets."
    )
    parser.add_argument("--config", default="mcp.json", help="Path to config JSON.")
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Override: use this single HF dataset name (ignores config datasets list).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Per-dataset cap (overrides config when set).",
    )
    parser.add_argument(
        "--skip-if-exists",
        action="store_true",
        help="Skip if train/validation/test CSVs already exist.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def load_config(config_arg: str) -> dict:
    config_path = config_arg
    if not os.path.isabs(config_path):
        config_path = os.path.join(ROOT, config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    args = parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    num_classes = config["teacher_model"]["num_classes"]
    data_dir = os.path.join(ROOT, paths["data_dir"])
    train_path = os.path.join(ROOT, paths["train_data"])
    val_path = os.path.join(ROOT, paths["validation_data"])
    test_path = os.path.join(ROOT, paths["test_data"])
    val_ratio = config.get("dataset", {}).get("validation_ratio", 0.05)
    test_ratio = config.get("dataset", {}).get("test_ratio", 0.05)

    if args.skip_if_exists and all(
        os.path.exists(p) for p in (train_path, val_path, test_path)
    ):
        print("Data files already exist; skipping.")
        for p in (train_path, val_path, test_path):
            print("  {}".format(p))
        return

    import pandas as pd
    from dataset_adapters import load_and_normalize

    os.makedirs(data_dir, exist_ok=True)

    # Multi-dataset: config has "datasets" list
    datasets_cfg = config.get("datasets")
    if args.dataset_name:
        # Single dataset override from CLI
        dataset_list = [{"hf_name": args.dataset_name, "max_samples": args.max_samples}]
    elif isinstance(datasets_cfg, list) and len(datasets_cfg) > 0:
        dataset_list = []
        for d in datasets_cfg:
            entry = {"hf_name": d["hf_name"]}
            entry["max_samples"] = args.max_samples if args.max_samples is not None else d.get("max_samples")
            dataset_list.append(entry)
    else:
        # Backward compat: single "dataset" block
        ds = config.get("dataset", {})
        hf_name = ds.get("hf_name", "NorthernTribe-Research/comprehensive-healthbench-v2")
        max_s = args.max_samples if args.max_samples is not None else ds.get("max_samples")
        dataset_list = [{"hf_name": hf_name, "max_samples": max_s}]

    frames = []
    for entry in dataset_list:
        hf_name = entry["hf_name"]
        max_samples = entry.get("max_samples")
        print("Loading {} (max_samples={})...".format(hf_name, max_samples))
        df = load_and_normalize(
            hf_name,
            num_classes=num_classes,
            max_samples=max_samples,
            seed=args.seed,
        )
        print("  -> {} rows".format(len(df)))
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["text"], keep="first")
    merged = merged.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    n = len(merged)
    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))
    n_train = n - n_test - n_val
    if n_train < 1:
        n_train, n_val, n_test = n - 2, 1, 1

    train_df = merged.iloc[:n_train]
    val_df = merged.iloc[n_train : n_train + n_val]
    test_df = merged.iloc[n_train + n_val :]

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("Merged total: {} rows (after dedup)".format(len(merged)))
    print("Saved train {} -> {}".format(len(train_df), train_path))
    print("Saved validation {} -> {}".format(len(val_df), val_path))
    print("Saved test {} -> {}".format(len(test_df), test_path))
    print("Train label distribution:\n{}".format(train_df["label"].value_counts().sort_index()))


if __name__ == "__main__":
    main()
