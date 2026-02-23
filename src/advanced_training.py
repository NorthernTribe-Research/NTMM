"""
NTMM Advanced Training Utilities - State-of-the-Art Training Enhancements

This module provides advanced training features:
- Early stopping based on validation metrics
- Learning rate finder for optimal LR selection
- Advanced data augmentation for medical text
- Training visualization and logging

Copyright (c) 2026 NorthernTribe Research
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class EarlyStoppingCallback(TrainerCallback):
    """
    Early stopping callback to prevent overfitting.
    
    Stops training when validation metric stops improving for a specified number of evaluations.
    """

    def __init__(
        self,
        early_stopping_patience: int = 3,
        early_stopping_threshold: float = 0.0,
        metric_for_best_model: str = "eval_loss",
        greater_is_better: bool = False,
    ):
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.metric_for_best_model = metric_for_best_model
        self.greater_is_better = greater_is_better
        self.patience_counter = 0
        self.best_metric = None

    def check_metric_value(self, metric_value: float) -> bool:
        """Check if current metric is better than best metric."""
        if self.best_metric is None:
            self.best_metric = metric_value
            return True

        if self.greater_is_better:
            is_improvement = metric_value > (self.best_metric + self.early_stopping_threshold)
        else:
            is_improvement = metric_value < (self.best_metric - self.early_stopping_threshold)

        if is_improvement:
            self.best_metric = metric_value
            self.patience_counter = 0
            return True
        else:
            self.patience_counter += 1
            return False

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Dict[str, float],
        **kwargs,
    ):
        """Called after evaluation."""
        metric_value = metrics.get(self.metric_for_best_model)
        if metric_value is None:
            return control

        is_improvement = self.check_metric_value(metric_value)

        if not is_improvement:
            if self.patience_counter >= self.early_stopping_patience:
                print(
                    f"\nEarly stopping triggered! No improvement in {self.metric_for_best_model} "
                    f"for {self.early_stopping_patience} evaluations."
                )
                control.should_training_stop = True

        return control


class LearningRateFinder:
    """
    Learning rate finder using the method from Leslie Smith's paper.
    
    Gradually increases learning rate and tracks loss to find optimal LR range.
    """

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device: str = "cpu",
        start_lr: float = 1e-7,
        end_lr: float = 10,
        num_iter: int = 100,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.num_iter = num_iter

        self.history = {"lr": [], "loss": []}
        self.best_loss = None

    def range_test(self, train_loader):
        """Run learning rate range test."""
        import torch

        self.model.train()
        lr_schedule = np.logspace(
            np.log10(self.start_lr), np.log10(self.end_lr), self.num_iter
        )

        iterator = iter(train_loader)
        for iteration, lr in enumerate(lr_schedule):
            # Update learning rate
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # Get batch
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)

            # Forward pass
            inputs = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(**inputs, labels=labels)
            loss = outputs.loss

            # Check for divergence
            if self.best_loss is None:
                self.best_loss = loss.item()
            elif loss.item() > 4 * self.best_loss:
                print(f"Stopping early at iteration {iteration}, loss diverged")
                break

            if loss.item() < self.best_loss:
                self.best_loss = loss.item()

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Record
            self.history["lr"].append(lr)
            self.history["loss"].append(loss.item())

            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.num_iter}, LR: {lr:.2e}, Loss: {loss.item():.4f}")

        return self.history

    def plot(self, skip_start: int = 10, skip_end: int = 5):
        """Plot learning rate vs loss."""
        try:
            import matplotlib.pyplot as plt

            lrs = self.history["lr"][skip_start:-skip_end]
            losses = self.history["loss"][skip_start:-skip_end]

            plt.figure(figsize=(10, 6))
            plt.plot(lrs, losses)
            plt.xscale("log")
            plt.xlabel("Learning Rate")
            plt.ylabel("Loss")
            plt.title("Learning Rate Finder")
            plt.grid(True)
            plt.savefig("lr_finder_plot.png")
            print("Learning rate plot saved to lr_finder_plot.png")
        except ImportError:
            print("matplotlib not installed, skipping plot")

    def suggest_lr(self, skip_start: int = 10, skip_end: int = 5) -> float:
        """Suggest optimal learning rate based on steepest gradient."""
        lrs = np.array(self.history["lr"][skip_start:-skip_end])
        losses = np.array(self.history["loss"][skip_start:-skip_end])

        # Find steepest gradient
        gradients = np.gradient(losses)
        min_gradient_idx = np.argmin(gradients)

        suggested_lr = lrs[min_gradient_idx]
        print(f"\nSuggested learning rate: {suggested_lr:.2e}")
        print(f"Recommended range: {suggested_lr / 10:.2e} to {suggested_lr:.2e}")

        return suggested_lr


class MedicalTextAugmenter:
    """
    Advanced text augmentation for medical domain.
    
    Applies domain-aware augmentation techniques while preserving medical meaning.
    """

    def __init__(self, augmentation_prob: float = 0.15):
        self.augmentation_prob = augmentation_prob

    def augment(self, text: str) -> str:
        """Apply random augmentation to text."""
        import random

        if random.random() > self.augmentation_prob:
            return text

        # Choose augmentation technique
        techniques = [
            self._synonym_replacement,
            self._random_insertion,
            self._random_swap,
        ]

        augmentation_fn = random.choice(techniques)
        return augmentation_fn(text)

    def _synonym_replacement(self, text: str) -> str:
        """Replace words with medical synonyms."""
        # Simple synonym mapping for common medical terms
        synonyms = {
            "patient": ["individual", "subject", "case"],
            "doctor": ["physician", "clinician", "practitioner"],
            "medicine": ["medication", "drug", "pharmaceutical"],
            "disease": ["condition", "disorder", "illness"],
            "treatment": ["therapy", "intervention", "management"],
        }

        words = text.split()
        for i, word in enumerate(words):
            word_lower = word.lower().strip(".,!?")
            if word_lower in synonyms and np.random.random() < 0.3:
                words[i] = np.random.choice(synonyms[word_lower])

        return " ".join(words)

    def _random_insertion(self, text: str) -> str:
        """Insert common medical filler words."""
        fillers = ["reportedly", "apparently", "notably", "significantly"]
        words = text.split()

        if len(words) > 3:
            insert_pos = np.random.randint(1, len(words))
            words.insert(insert_pos, np.random.choice(fillers))

        return " ".join(words)

    def _random_swap(self, text: str) -> str:
        """Randomly swap adjacent words (preserving medical meaning)."""
        words = text.split()

        if len(words) > 3:
            idx = np.random.randint(0, len(words) - 1)
            words[idx], words[idx + 1] = words[idx + 1], words[idx]

        return " ".join(words)


class TrainingMonitor(TrainerCallback):
    """
    Monitor training progress and log detailed metrics.
    """

    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file
        self.training_history = []

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Dict[str, float] = None,
        **kwargs,
    ):
        """Called when logging."""
        if logs:
            self.training_history.append(
                {"step": state.global_step, "epoch": state.epoch, **logs}
            )

            if self.log_file:
                import json
                from pathlib import Path

                log_path = Path(self.log_file)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("w") as f:
                    json.dump(self.training_history, f, indent=2)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Called at the end of training."""
        print(f"\nTraining completed! Total steps: {state.global_step}")
        if self.training_history:
            print(f"Training history saved with {len(self.training_history)} entries")


def get_advanced_training_callbacks(
    enable_early_stopping: bool = True,
    early_stopping_patience: int = 3,
    log_file: Optional[str] = None,
) -> list:
    """
    Get list of advanced training callbacks.
    
    Args:
        enable_early_stopping: Whether to enable early stopping
        early_stopping_patience: Number of evaluations to wait before stopping
        log_file: Path to save training logs
        
    Returns:
        List of callback instances
    """
    callbacks = []

    if enable_early_stopping:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
            )
        )

    callbacks.append(TrainingMonitor(log_file=log_file))

    return callbacks
