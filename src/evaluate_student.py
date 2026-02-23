from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate distilled student model.")
    parser.add_argument("--config", default="mcp.json", help="Path to config JSON.")
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--output-report", default=None, help="Optional JSON report path.")
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.is_absolute():
        path = project_root() / config_path
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def tokenize_dataset(frame, tokenizer, max_length: int):
    from datasets import Dataset

    dataset = Dataset.from_pandas(frame, preserve_index=False)
    dataset = dataset.rename_column("label", "labels")

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    dataset = dataset.map(tokenize_fn, batched=True)
    keep_cols = {"input_ids", "attention_mask", "token_type_ids", "labels"}
    remove_cols = [col for col in dataset.column_names if col not in keep_cols]
    if remove_cols:
        dataset = dataset.remove_columns(remove_cols)
    return dataset


def main() -> None:
    args = parse_args()

    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    config = load_config(args.config)

    paths = config["paths"]
    student_cfg = config["student_model"]
    root = project_root()

    student_path = root / paths["student_model_path"]
    if not student_path.exists():
        raise FileNotFoundError(
            "Student model path not found: {}. Run distillation first.".format(student_path)
        )

    test_path = root / paths["test_data"]
    if not test_path.exists():
        raise FileNotFoundError("Test split not found: {}".format(test_path))

    print("Loading NTMM student model from {}".format(student_path))
    tokenizer = AutoTokenizer.from_pretrained(str(student_path), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(student_path))

    test_df = pd.read_csv(test_path)
    required = {"text", "label"}
    missing = required - set(test_df.columns)
    if missing:
        raise ValueError("Test split missing columns {} at {}".format(missing, test_path))

    if args.max_test_samples is not None:
        test_df = test_df.head(args.max_test_samples)

    test_ds = tokenize_dataset(test_df, tokenizer, student_cfg["max_sequence_length"])

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(student_path / "eval_tmp"),
            per_device_eval_batch_size=config["training_params"].get("eval_batch_size", 16),
            report_to=[],
        ),
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    print("Running evaluation on {} rows.".format(len(test_ds)))
    prediction_output = trainer.predict(test_ds)
    logits = prediction_output.predictions
    labels = prediction_output.label_ids
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    f1_weighted = f1_score(labels, predictions, average="weighted")
    report_text = classification_report(labels, predictions, zero_division=0)
    report_dict = classification_report(labels, predictions, output_dict=True, zero_division=0)

    print("Accuracy: {:.4f}".format(accuracy))
    print("Weighted F1: {:.4f}".format(f1_weighted))
    print("\n--- Classification Report ---")
    print(report_text)

    report_path = args.output_report
    if report_path is None:
        report_path = student_path / "evaluation_report.json"
    else:
        report_path = Path(report_path)
        if not report_path.is_absolute():
            report_path = root / report_path

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "accuracy": accuracy,
                "f1_weighted": f1_weighted,
                "classification_report": report_dict,
            },
            handle,
            indent=2,
        )
    print("Saved report to {}".format(report_path))


if __name__ == "__main__":
    main()
