# NTMM Deployment Guide

This guide covers deploying NTMM models to various platforms.

## Table of Contents
- [Hugging Face Hub](#hugging-face-hub)
- [Docker Deployment](#docker-deployment)
- [REST API with FastAPI](#rest-api-with-fastapi)
- [AWS SageMaker](#aws-sagemaker)
- [Production Considerations](#production-considerations)

## Hugging Face Hub

### Upload Model

```bash
# Install Hugging Face CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Upload model
huggingface-cli upload NorthernTribe-Research/ntmm-student-v1 saved_models/ntmm-student/
```

### Use Uploaded Model

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="NorthernTribe-Research/ntmm-student-v1"
)

result = classifier("Patient presents with fever and cough.")
print(result)
```

## Docker Deployment

### Create Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and code
COPY saved_models/ntmm-student /app/model
COPY examples/inference_example.py /app/

# Expose port for API
EXPOSE 8000

# Run inference server
CMD ["python", "inference_example.py"]
```

### Build and Run

```bash
docker build -t ntmm:latest .
docker run -p 8000:8000 ntmm:latest
```

## REST API with FastAPI

### Create API Server

```python
# api_server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI(title="NTMM API", version="0.1.0")

# Load model at startup
model_path = "saved_models/ntmm-student"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

class PredictionRequest(BaseModel):
    text: str
    max_length: int = 256

class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    probabilities: list[float]

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        inputs = tokenizer(
            request.text,
            return_tensors="pt",
            truncation=True,
            max_length=request.max_length,
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            pred = logits.argmax(-1).item()
            conf = probs[0, pred].item()
        
        return PredictionResponse(
            prediction=pred,
            confidence=conf,
            probabilities=probs[0].cpu().tolist()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

### Run API Server

```bash
pip install fastapi uvicorn
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

### Test API

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Patient has fever and cough."}'
```

## AWS SageMaker

### Prepare Model for SageMaker

```python
# sagemaker_deploy.py
import tarfile
from pathlib import Path

model_path = Path("saved_models/ntmm-student")
output_file = "model.tar.gz"

with tarfile.open(output_file, "w:gz") as tar:
    tar.add(model_path, arcname=".")

print(f"Model packaged: {output_file}")
```

### Deploy to SageMaker

```python
import boto3
import sagemaker
from sagemaker.huggingface import HuggingFaceModel

role = sagemaker.get_execution_role()

# Upload model to S3
s3_client = boto3.client('s3')
bucket = 'your-bucket-name'
s3_client.upload_file('model.tar.gz', bucket, 'ntmm/model.tar.gz')

# Create HuggingFace model
huggingface_model = HuggingFaceModel(
    model_data=f's3://{bucket}/ntmm/model.tar.gz',
    role=role,
    transformers_version='4.46',
    pytorch_version='2.2',
    py_version='py310',
)

# Deploy
predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type='ml.g4dn.xlarge'
)

# Predict
result = predictor.predict({
    'inputs': 'Patient presents with fever.'
})
print(result)
```

## Production Considerations

### Performance Optimization

1. **Batch Processing**
```python
# Process multiple texts at once
texts = ["text1", "text2", "text3"]
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
outputs = model(**inputs)
```

2. **Model Quantization**
```python
# Reduce model size with quantization
from transformers import AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained(
    "saved_models/ntmm-student",
    torch_dtype=torch.float16  # Use half precision
)
```

3. **ONNX Export**
```python
# Export to ONNX for faster inference
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained("saved_models/ntmm-student")
tokenizer = AutoTokenizer.from_pretrained("saved_models/ntmm-student")

dummy_input = tokenizer("sample text", return_tensors="pt")
torch.onnx.export(
    model,
    (dummy_input["input_ids"], dummy_input["attention_mask"]),
    "model.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "logits": {0: "batch"}
    }
)
```

### Monitoring

1. **Log predictions and confidence scores**
2. **Track inference latency**
3. **Monitor model drift**
4. **Set up alerts for low confidence predictions**

### Security

1. **Input validation**: Sanitize and validate all inputs
2. **Rate limiting**: Prevent abuse with rate limits
3. **Authentication**: Require API keys or OAuth
4. **HTTPS**: Always use encrypted connections

### Scaling

1. **Horizontal scaling**: Deploy multiple instances behind a load balancer
2. **Auto-scaling**: Scale based on request volume
3. **Caching**: Cache frequent predictions
4. **Async processing**: Use message queues for batch jobs

### Compliance

1. **Data privacy**: Ensure HIPAA/GDPR compliance
2. **Audit logging**: Log all predictions for compliance
3. **Model versioning**: Track which model version made each prediction
4. **Validation**: Regular validation against gold standard datasets

## Example Production Stack

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
┌──────▼──────┐
│ Load Balancer│
└──────┬──────┘
       │
   ┌───┴───┐
   │       │
┌──▼──┐ ┌──▼──┐
│API 1│ │API 2│  (FastAPI + NTMM)
└──┬──┘ └──┬──┘
   │       │
   └───┬───┘
       │
┌──────▼──────┐
│  Monitoring │  (Prometheus, Grafana)
└─────────────┘
```

## Support

For deployment issues, see [FAQ.md](FAQ.md) or open an issue on GitHub.
