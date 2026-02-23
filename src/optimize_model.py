"""
NTMM Model Optimization - State-of-the-Art Inference Optimization

This module provides advanced model optimization techniques:
- INT8 quantization for 4x size reduction
- ONNX export for cross-platform deployment
- Dynamic quantization for CPU inference
- Benchmark utilities for performance testing

Copyright (c) 2026 NorthernTribe Research
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize NTMM student model for deployment.")
    parser.add_argument("--config", default="mcp.json", help="Path to config JSON.")
    parser.add_argument(
        "--optimization",
        choices=["quantize", "onnx", "both", "benchmark"],
        default="both",
        help="Optimization type to apply.",
    )
    parser.add_argument("--quantization-type", choices=["dynamic", "static"], default="dynamic")
    parser.add_argument("--benchmark-samples", type=int, default=100)
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.is_absolute():
        path = project_root() / config_path
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def quantize_model(model_path: Path, output_path: Path, quantization_type: str = "dynamic") -> None:
    """Apply INT8 quantization to reduce model size and improve inference speed."""
    import torch
    from transformers import AutoModelForSequenceClassification

    print(f"Loading model from {model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))

    if quantization_type == "dynamic":
        print("Applying dynamic INT8 quantization...")
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
    else:
        print("Static quantization not yet implemented. Using dynamic quantization.")
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )

    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving quantized model to {output_path}")
    quantized_model.save_pretrained(str(output_path))

    # Copy tokenizer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    tokenizer.save_pretrained(str(output_path))

    # Calculate size reduction
    original_size = sum(f.stat().st_size for f in model_path.rglob("*.bin") if f.is_file())
    quantized_size = sum(f.stat().st_size for f in output_path.rglob("*.bin") if f.is_file())
    reduction = (1 - quantized_size / original_size) * 100

    print(f"Original size: {original_size / 1024 / 1024:.2f} MB")
    print(f"Quantized size: {quantized_size / 1024 / 1024:.2f} MB")
    print(f"Size reduction: {reduction:.1f}%")


def export_to_onnx(model_path: Path, output_path: Path) -> None:
    """Export model to ONNX format for optimized cross-platform inference."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print(f"Loading model from {model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    model.eval()

    # Create dummy input
    dummy_text = "This is a sample medical text for ONNX export."
    inputs = tokenizer(dummy_text, return_tensors="pt", padding=True, truncation=True)

    output_path.mkdir(parents=True, exist_ok=True)
    onnx_path = output_path / "model.onnx"

    print(f"Exporting to ONNX: {onnx_path}")

    # Dynamic axes for variable batch size and sequence length
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size"},
    }

    with torch.no_grad():
        torch.onnx.export(
            model,
            (inputs["input_ids"], inputs["attention_mask"]),
            str(onnx_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=14,
            do_constant_folding=True,
        )

    # Save tokenizer
    tokenizer.save_pretrained(str(output_path))

    print(f"ONNX model exported successfully to {onnx_path}")
    print(f"Model size: {onnx_path.stat().st_size / 1024 / 1024:.2f} MB")


def benchmark_model(model_path: Path, num_samples: int = 100) -> dict:
    """Benchmark model inference performance."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

    print(f"Benchmarking model from {model_path}")
    print(f"Running {num_samples} inference samples...")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))

    # Sample medical texts for benchmarking
    sample_texts = [
        "Patient presents with fever and cough.",
        "Blood pressure is elevated at 150/95 mmHg.",
        "MRI shows no abnormalities in brain structure.",
        "Laboratory results indicate elevated white blood cell count.",
        "Patient reports chronic lower back pain for 6 months.",
    ] * (num_samples // 5 + 1)
    sample_texts = sample_texts[:num_samples]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # Warmup
    print("Warming up...")
    for _ in range(10):
        inputs = tokenizer(sample_texts[0], return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            _ = model(**inputs)

    # Benchmark
    print("Running benchmark...")
    start_time = time.time()
    for text in sample_texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            _ = model(**inputs)
    end_time = time.time()

    total_time = end_time - start_time
    avg_time = total_time / num_samples
    throughput = num_samples / total_time

    results = {
        "device": device,
        "num_samples": num_samples,
        "total_time_seconds": round(total_time, 3),
        "avg_time_ms": round(avg_time * 1000, 2),
        "throughput_samples_per_second": round(throughput, 2),
    }

    print("\n--- Benchmark Results ---")
    print(f"Device: {results['device']}")
    print(f"Total time: {results['total_time_seconds']}s")
    print(f"Average time per sample: {results['avg_time_ms']}ms")
    print(f"Throughput: {results['throughput_samples_per_second']} samples/second")

    return results


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    root = project_root()
    student_path = root / config["paths"]["student_model_path"]

    if not student_path.exists():
        raise FileNotFoundError(
            f"Student model not found at {student_path}. Run distillation first."
        )

    if args.optimization in ["quantize", "both"]:
        quantized_path = root / "saved_models" / "ntmm-student-quantized"
        quantize_model(student_path, quantized_path, args.quantization_type)

    if args.optimization in ["onnx", "both"]:
        onnx_path = root / "saved_models" / "ntmm-student-onnx"
        export_to_onnx(student_path, onnx_path)

    if args.optimization == "benchmark":
        results = benchmark_model(student_path, args.benchmark_samples)

        # Save benchmark results
        benchmark_path = student_path / "benchmark_results.json"
        with benchmark_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nBenchmark results saved to {benchmark_path}")

    print("\nOptimization complete!")


if __name__ == "__main__":
    main()
