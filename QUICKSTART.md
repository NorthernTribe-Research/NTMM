# NTMM Quick Start Guide

Get your first NTMM model running in under 15 minutes!

## Prerequisites

- Linux or macOS (Windows with WSL2 works too)
- Python 3.10 or higher
- 10GB free disk space
- Internet connection (for downloading models and datasets)
- GPU recommended but not required

## Step 1: Clone and Setup (2 minutes)

```bash
# Clone the repository
git clone https://github.com/NorthernTribe-Research/NTMM.git
cd NTMM

# Run automated setup
chmod +x setup.sh
./setup.sh
```

This will:
- Create a virtual environment
- Install all dependencies
- Run tests to verify everything works

## Step 2: Quick Test Run (5-15 minutes)

```bash
# Activate virtual environment
source .venv/bin/activate

# Run quick mode (small dataset for testing)
./run_all_steps.sh quick
```

This will:
1. Download and prepare medical datasets (64 train, 16 val, 32 test samples)
2. Train a teacher model (Qwen2-1.5B)
3. Distill to NTMM student model (Qwen2-0.5B)
4. Evaluate and generate metrics

## Step 3: Check Your NTMM Model

```bash
# View model card
cat saved_models/ntmm-student/README.md

# Check metrics
cat saved_models/ntmm-student/evaluation_report.json

# List model files
ls -lh saved_models/ntmm-student/
```

## Step 4: Test Inference

```bash
# Run inference example
python examples/inference_example.py \
    --text "Patient presents with fever and cough."
```

Expected output:
```
Loading NTMM model from saved_models/ntmm-student...
Model loaded on cuda

Input text: Patient presents with fever and cough.

Running inference...

Prediction: Class 2
Confidence: 0.8543

Class probabilities:
  Class 0: 0.0234
  Class 1: 0.0891
  Class 2: 0.8543
  Class 3: 0.0221
  Class 4: 0.0111
```

## Step 5: Customize (Optional)

### Use Your Own Data

Place CSV files with `text` and `label` columns in `data/`:
```bash
# Your data format
text,label
"Patient has fever",0
"No symptoms",1
```

### Adjust Configuration

Edit `mcp.json`:
```json
{
  "teacher_model": {
    "name": "Qwen/Qwen2-1.5B",
    "num_classes": 5  // Change to your number of classes
  },
  "training_params": {
    "teacher_epochs": 3,  // More epochs = better fit
    "student_epochs": 10
  }
}
```

### Run Full Training

```bash
# Full dataset (takes 1-4 hours depending on hardware)
./run_all_steps.sh
```

## Step 6: Deploy Your Model

### Option A: Hugging Face Hub

```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli upload NorthernTribe-Research/ntmm-v1 saved_models/ntmm-student/
```

### Option B: REST API

```bash
# Install FastAPI
pip install fastapi uvicorn

# Create simple API (see docs/DEPLOYMENT.md for full example)
# Then run:
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

### Option C: Docker

```bash
# Build image
docker build -t ntmm:latest .

# Run container
docker run -p 8000:8000 ntmm:latest
```

## Common Issues

### "CUDA out of memory"
Reduce batch size in `mcp.json`:
```json
"training_params": {
    "teacher_batch_size": 4,
    "student_batch_size": 8
}
```

### "torch not found"
Install PyTorch:
```bash
# For CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121

# For CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### "datasets not found"
Install missing dependencies:
```bash
pip install -e ".[dev]"
```

## Next Steps

1. **Read the docs**: Check out [docs/FAQ.md](docs/FAQ.md) for detailed information
2. **Explore examples**: See [examples/README.md](examples/README.md) for more use cases
3. **Deploy to production**: Follow [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)
4. **Customize**: Adjust hyperparameters and datasets for your use case
5. **Contribute**: See [CONTRIBUTING.md](CONTRIBUTING.md) to help improve NTMM

## Getting Help

- **FAQ**: [docs/FAQ.md](docs/FAQ.md)
- **Issues**: Open an issue on GitHub
- **Documentation**: [README.md](README.md)

## What You've Accomplished

✅ Set up NTMM development environment
✅ Trained your first NTMM model
✅ Generated a branded model card
✅ Tested inference
✅ Ready to deploy or customize

**Congratulations!** You now have a working NTMM model owned by NorthernTribe Research.

---

**Time to production**: ~15 minutes for quick test, 2-4 hours for full training

**Model ownership**: All NTMM models are owned by NorthernTribe Research and licensed under MIT
