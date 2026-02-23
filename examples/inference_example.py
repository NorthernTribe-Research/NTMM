"""
Example: Using a trained NTMM student model for inference.

This script demonstrates how to load and use an NTMM model for medical text classification.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="NTMM inference example")
    parser.add_argument(
        "--model-path",
        default="saved_models/ntmm-student",
        help="Path to trained NTMM model",
    )
    parser.add_argument(
        "--text",
        default="Patient presents with fever, cough, and difficulty breathing.",
        help="Medical text to classify",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum sequence length",
    )
    return parser.parse_args()


def load_model(model_path: str):
    """Load NTMM model and tokenizer."""
    print(f"Loading NTMM model from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded on {device}")
    return model, tokenizer, device


def predict(model, tokenizer, device, text: str, max_length: int = 256):
    """Run inference on medical text."""
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    )
    
    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        prediction = logits.argmax(-1).item()
        confidence = probabilities[0, prediction].item()
    
    return prediction, confidence, probabilities[0].cpu().numpy()


def main():
    args = parse_args()
    
    # Check if model exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train a model first using: ./run_all_steps.sh")
        return
    
    # Load model
    model, tokenizer, device = load_model(args.model_path)
    
    # Run inference
    print(f"\nInput text: {args.text}")
    print("\nRunning inference...")
    
    prediction, confidence, probabilities = predict(
        model, tokenizer, device, args.text, args.max_length
    )
    
    # Display results
    print(f"\nPrediction: Class {prediction}")
    print(f"Confidence: {confidence:.4f}")
    print("\nClass probabilities:")
    for i, prob in enumerate(probabilities):
        print(f"  Class {i}: {prob:.4f}")


if __name__ == "__main__":
    main()
