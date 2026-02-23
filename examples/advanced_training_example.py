"""
NTMM Advanced Training Example

This example demonstrates how to use state-of-the-art training features:
- Early stopping
- Learning rate finder
- Medical text augmentation
- Training monitoring
- Model optimization

Copyright (c) 2026 NorthernTribe Research
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def example_early_stopping():
    """Example: Using early stopping to prevent overfitting."""
    print("\n=== Example 1: Early Stopping ===\n")

    from advanced_training import get_advanced_training_callbacks

    # Get callbacks with early stopping enabled
    callbacks = get_advanced_training_callbacks(
        enable_early_stopping=True,
        early_stopping_patience=3,
        log_file="training_logs/training_history.json",
    )

    print("Early stopping callback created!")
    print("- Will stop training if validation loss doesn't improve for 3 evaluations")
    print("- Training history will be saved to training_logs/training_history.json")
    print("\nTo use with Trainer:")
    print("  trainer = Trainer(..., callbacks=callbacks)")


def example_learning_rate_finder():
    """Example: Finding optimal learning rate."""
    print("\n=== Example 2: Learning Rate Finder ===\n")

    print("Learning Rate Finder helps you find the optimal learning rate automatically.")
    print("\nUsage:")
    print("""
from advanced_training import LearningRateFinder

# Initialize
lr_finder = LearningRateFinder(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    device="cuda",
    start_lr=1e-7,
    end_lr=10,
    num_iter=100
)

# Run range test
history = lr_finder.range_test(train_loader)

# Get suggestion
optimal_lr = lr_finder.suggest_lr()

# Plot results
lr_finder.plot()
    """)

    print("The LR finder will:")
    print("1. Gradually increase learning rate from 1e-7 to 10")
    print("2. Track loss at each learning rate")
    print("3. Suggest optimal LR based on steepest gradient")
    print("4. Generate a plot showing LR vs Loss")


def example_text_augmentation():
    """Example: Medical text augmentation."""
    print("\n=== Example 3: Medical Text Augmentation ===\n")

    from advanced_training import MedicalTextAugmenter

    augmenter = MedicalTextAugmenter(augmentation_prob=0.15)

    # Example medical texts
    texts = [
        "Patient presents with fever and cough.",
        "Blood pressure is elevated at 150/95 mmHg.",
        "Doctor prescribed medication for the disease.",
    ]

    print("Original texts:")
    for i, text in enumerate(texts, 1):
        print(f"  {i}. {text}")

    print("\nAugmented texts (may vary due to randomness):")
    for i, text in enumerate(texts, 1):
        augmented = augmenter.augment(text)
        print(f"  {i}. {augmented}")

    print("\nAugmentation techniques:")
    print("- Synonym replacement (patient → individual, doctor → physician)")
    print("- Random insertion (adds medical fillers like 'reportedly', 'notably')")
    print("- Random swap (swaps adjacent words while preserving meaning)")


def example_model_optimization():
    """Example: Model optimization (quantization and ONNX)."""
    print("\n=== Example 4: Model Optimization ===\n")

    print("NTMM provides tools to optimize your trained models for deployment:\n")

    print("1. INT8 Quantization (4x size reduction):")
    print("   python src/optimize_model.py --optimization quantize")
    print("   - Reduces model size from 500MB to ~125MB")
    print("   - Speeds up inference by ~1.9x")
    print("   - Minimal accuracy loss (1-2%)")

    print("\n2. ONNX Export (cross-platform deployment):")
    print("   python src/optimize_model.py --optimization onnx")
    print("   - Exports to ONNX format for optimized inference")
    print("   - 2-3x faster inference with ONNX Runtime")
    print("   - Compatible with multiple frameworks and hardware")

    print("\n3. Benchmark Performance:")
    print("   python src/optimize_model.py --optimization benchmark --benchmark-samples 1000")
    print("   - Measures inference latency and throughput")
    print("   - Tests on CPU and GPU")
    print("   - Saves results to JSON for analysis")

    print("\n4. All-in-one Optimization:")
    print("   python src/optimize_model.py --optimization both")
    print("   - Runs both quantization and ONNX export")


def example_complete_training_pipeline():
    """Example: Complete training pipeline with all features."""
    print("\n=== Example 5: Complete Training Pipeline ===\n")

    print("Here's how to combine all advanced features in your training:")
    print("""
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from advanced_training import get_advanced_training_callbacks
from distillation_utils import DistillationTrainer

# 1. Load models
teacher = AutoModelForSequenceClassification.from_pretrained("saved_models/teacher")
student = AutoModelForSequenceClassification.from_pretrained("Qwen/Qwen2-0.5B")

# 2. Configure state-of-the-art training
training_args = TrainingArguments(
    output_dir="saved_models/ntmm-student",
    num_train_epochs=10,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    warmup_steps=100,                    # 10% warmup
    lr_scheduler_type="cosine",          # Cosine decay
    gradient_accumulation_steps=4,       # Effective batch = 64
    label_smoothing_factor=0.1,          # Prevent overconfidence
    max_grad_norm=1.0,                   # Gradient clipping
    fp16=torch.cuda.is_available(),      # Mixed precision
    bf16=torch.cuda.is_bf16_supported(), # BF16 if available
    eval_strategy="epoch",               # Evaluate each epoch
    save_strategy="epoch",               # Save each epoch
    load_best_model_at_end=True,         # Load best model
    metric_for_best_model="eval_loss",   # Optimize for loss
)

# 3. Get advanced callbacks
callbacks = get_advanced_training_callbacks(
    enable_early_stopping=True,
    early_stopping_patience=3,
    log_file="training_logs/history.json"
)

# 4. Create trainer with multi-objective distillation
trainer = DistillationTrainer(
    model=student,
    teacher_model=teacher,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    callbacks=callbacks,
    temperature=3.0,
    alpha=0.5,
    use_cosine_loss=True,  # Enable cosine similarity
    use_mse_loss=False,    # Optional MSE loss
)

# 5. Train
trainer.train()

# 6. Optimize for deployment
# Run: python src/optimize_model.py --optimization both
    """)


def main():
    """Run all examples."""
    print("=" * 70)
    print("NTMM Advanced Training Features - Examples")
    print("=" * 70)

    example_early_stopping()
    example_learning_rate_finder()
    example_text_augmentation()
    example_model_optimization()
    example_complete_training_pipeline()

    print("\n" + "=" * 70)
    print("For more information, see:")
    print("- docs/MODEL_ARCHITECTURE.md - Complete architecture documentation")
    print("- docs/FAQ.md - Frequently asked questions")
    print("- QUICKSTART.md - 15-minute tutorial")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
