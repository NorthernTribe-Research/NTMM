"""Test model card generation."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def test_model_card_generation():
    """Test that model card can be generated with valid config."""
    from model_card_template import generate_model_card
    
    config = {
        "student_model": {"name": "Qwen/Qwen2-0.5B"},
        "teacher_model": {"name": "Qwen/Qwen2-1.5B"},
        "training_params": {
            "teacher_epochs": 3,
            "teacher_batch_size": 8,
            "teacher_learning_rate": 2e-5,
            "student_epochs": 10,
            "student_batch_size": 16,
            "student_learning_rate": 2e-5,
        },
        "distillation_params": {"temperature": 3.0, "alpha": 0.5},
    }
    
    metrics = {"accuracy": 0.85, "f1_weighted": 0.83}
    datasets = ["dataset1", "dataset2"]
    
    card = generate_model_card(Path("test"), config, metrics, datasets)
    
    assert "NTMM Student Model" in card
    assert "NorthernTribe Research" in card
    assert "Qwen/Qwen2-0.5B" in card
    assert "Qwen/Qwen2-1.5B" in card
    assert "0.8500" in card  # accuracy formatted
    assert "dataset1" in card
    assert "mit" in card.lower()  # License check (case-insensitive)


def test_model_card_without_metrics():
    """Test model card generation without metrics."""
    from model_card_template import generate_model_card
    
    config = {
        "student_model": {"name": "test-model"},
        "teacher_model": {"name": "test-teacher"},
        "training_params": {},
        "distillation_params": {},
    }
    
    card = generate_model_card(Path("test"), config, None, None)
    
    assert "NTMM Student Model" in card
    assert "NorthernTribe Research" in card
