from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train teacher model for medical classification.")
    parser.add_argument("--config", default="mcp.json", help="Path to config JSON.")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-validation-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.is_absolute():
        path = project_root() / config_path
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_split(csv_path: Path, split_name: str):
    import pandas as pd

    if not csv_path.exists():
        raise FileNotFoundError("{} split not found: {}".format(split_name, csv_path))
    frame = pd.read_csv(csv_path)
    required = {"text", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("{} split missing columns {} at {}".format(split_name, missing, csv_path))
    return frame


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

    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    from distillation_utils import compute_classification_metrics

    config = load_config(args.config)

    paths = config["paths"]
    teacher_cfg = config["teacher_model"]
    train_cfg = config["training_params"]

    set_seed(args.seed)
    root = project_root()

    train_path = root / paths["train_data"]
    val_path = root / paths["validation_data"]

    train_df = read_split(train_path, "teacher train")
    val_df = read_split(val_path, "teacher validation")

    if args.max_train_samples is not None:
        train_df = train_df.head(args.max_train_samples)
    if args.max_validation_samples is not None:
        val_df = val_df.head(args.max_validation_samples)

    print("Loading tokenizer/model: {}".format(teacher_cfg["name"]))
    tokenizer = AutoTokenizer.from_pretrained(teacher_cfg["name"], use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        teacher_cfg["name"],
        num_labels=teacher_cfg["num_classes"],
    )

    train_ds = tokenize_dataset(train_df, tokenizer, teacher_cfg["max_sequence_length"])
    val_ds = tokenize_dataset(val_df, tokenizer, teacher_cfg["max_sequence_length"])

    output_dir = root / paths["teacher_model_path"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # State-of-the-art training configuration
    total_steps = (len(train_ds) // train_cfg["teacher_batch_size"]) * train_cfg["teacher_epochs"]
    warmup_steps = int(0.1 * total_steps)  # 10% warmup
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg["teacher_epochs"],
        per_device_train_batch_size=train_cfg["teacher_batch_size"],
        per_device_eval_batch_size=train_cfg.get("eval_batch_size", train_cfg["teacher_batch_size"]),
        learning_rate=train_cfg["teacher_learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 0.01),
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        eval_strategy="no",
        save_strategy="no",
        logging_steps=train_cfg.get("logging_steps", 20),
        report_to=[],
        fp16=torch.cuda.is_available(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", False),
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        label_smoothing_factor=train_cfg.get("label_smoothing", 0.1),
        remove_unused_columns=True,
        dataloader_num_workers=train_cfg.get("num_workers", 0),
        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_classification_metrics,
    )

    print("Starting teacher training on {} rows.".format(len(train_ds)))
    trainer.train()
    metrics = trainer.evaluate()

    print("Saving teacher model to {}".format(output_dir))
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print("Saved teacher metrics to {}".format(metrics_path))


if __name__ == "__main__":
    main()
