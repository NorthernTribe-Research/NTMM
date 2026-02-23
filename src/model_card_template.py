"""Generate model card for NTMM student models."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path


def generate_model_card(
    model_path: Path,
    config: dict,
    metrics: dict | None = None,
    datasets_used: list[str] | None = None,
) -> str:
    """Generate a model card for the NTMM student model."""
    
    student_cfg = config.get("student_model", {})
    teacher_cfg = config.get("teacher_model", {})
    distill_cfg = config.get("distillation_params", {})
    train_cfg = config.get("training_params", {})
    
    datasets_section = ""
    if datasets_used:
        datasets_section = "\n".join(f"- {ds}" for ds in datasets_used)
    else:
        datasets_section = "- Multiple medical datasets (see config)"
    
    metrics_section = ""
    if metrics:
        metrics_section = f"""
## Performance

| Metric | Value |
|--------|-------|
| Accuracy | {metrics.get('accuracy', 'N/A'):.4f} |
| Weighted F1 | {metrics.get('f1_weighted', 'N/A'):.4f} |
"""
    
    card = f"""---
language: en
license: mit
tags:
- medical
- knowledge-distillation
- ntmm
- northerntribe
- text-classification
library_name: transformers
pipeline_tag: text-classification
---

# NTMM Student Model

**NorthernTribe Medical Models (NTMM)** - Distilled medical reasoning model

## Model Description

This is an NTMM student model created through knowledge distillation from a larger teacher model. 
The model is designed for efficient medical text classification and reasoning tasks.

- **Developed by:** NorthernTribe Research
- **Model type:** Text Classification (Distilled)
- **Base model:** {student_cfg.get('name', 'N/A')}
- **Teacher model:** {teacher_cfg.get('name', 'N/A')}
- **Language:** English
- **License:** MIT

## Training Details

### Teacher Model
- Model: {teacher_cfg.get('name', 'N/A')}
- Epochs: {train_cfg.get('teacher_epochs', 'N/A')}
- Batch size: {train_cfg.get('teacher_batch_size', 'N/A')}
- Learning rate: {train_cfg.get('teacher_learning_rate', 'N/A')}

### Student Model (Distillation)
- Model: {student_cfg.get('name', 'N/A')}
- Epochs: {train_cfg.get('student_epochs', 'N/A')}
- Batch size: {train_cfg.get('student_batch_size', 'N/A')}
- Learning rate: {train_cfg.get('student_learning_rate', 'N/A')}
- Temperature: {distill_cfg.get('temperature', 'N/A')}
- Alpha (distillation weight): {distill_cfg.get('alpha', 'N/A')}

### Datasets
{datasets_section}
{metrics_section}

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "path/to/ntmm-student"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

text = "Patient presents with fever and cough..."
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
outputs = model(**inputs)
prediction = outputs.logits.argmax(-1).item()
```

## Limitations

- Trained on medical datasets; performance may vary on out-of-domain text
- Not a replacement for professional medical advice
- Should be validated before clinical deployment

## Citation

```bibtex
@software{{ntmm2025,
  title = {{NorthernTribe Medical Models (NTMM)}},
  author = {{NorthernTribe Research}},
  year = {{2025}},
  url = {{https://github.com/NorthernTribe-Research/NTMM}}
}}
```

## Contact

NorthernTribe Research

---
*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    return card


def save_model_card(model_path: Path, config: dict, metrics: dict | None = None) -> None:
    """Save model card to the model directory."""
    datasets_used = []
    if "datasets" in config and isinstance(config["datasets"], list):
        datasets_used = [d.get("hf_name", "") for d in config["datasets"] if d.get("hf_name")]
    elif "dataset" in config and "hf_name" in config["dataset"]:
        datasets_used = [config["dataset"]["hf_name"]]
    
    card_content = generate_model_card(model_path, config, metrics, datasets_used)
    card_path = model_path / "README.md"
    
    with card_path.open("w", encoding="utf-8") as f:
        f.write(card_content)
    
    print(f"Model card saved to {card_path}")
