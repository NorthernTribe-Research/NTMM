# NTMM Examples

This directory contains example scripts demonstrating how to use NTMM models.

## Inference Example

Run inference with a trained NTMM model:

```bash
python examples/inference_example.py \
    --model-path saved_models/ntmm-student \
    --text "Patient presents with fever and cough."
```

### Custom Text

```bash
python examples/inference_example.py \
    --text "Your medical text here..."
```

### Using a Different Model

```bash
python examples/inference_example.py \
    --model-path path/to/your/model \
    --text "Medical text..."
```

## Batch Inference

For processing multiple texts, you can modify the inference script or use the model directly:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_path = "saved_models/ntmm-student"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

texts = [
    "Patient has fever and cough.",
    "No symptoms reported.",
    "Severe headache and nausea."
]

inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=256)
with torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits.argmax(-1)

for text, pred in zip(texts, predictions):
    print(f"{text} -> Class {pred.item()}")
```

## Integration with Hugging Face Hub

After training, you can upload your NTMM model to Hugging Face:

```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli upload NorthernTribe-Research/ntmm-student-v1 saved_models/ntmm-student/
```

Then use it anywhere:

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="NorthernTribe-Research/ntmm-student-v1"
)

result = classifier("Patient presents with fever.")
print(result)
```
