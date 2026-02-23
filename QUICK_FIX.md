# Quick Fix Guide

## Issue: Missing Dependencies

If you see errors like:
- `ModuleNotFoundError: No module named 'datasets'`
- `ImportError: Using the Trainer with PyTorch requires accelerate>=1.1.0`

### Solution

Run this command to install missing dependencies:

```bash
# Activate virtual environment
source .venv/bin/activate

# Install missing packages
pip install accelerate>=1.1.0 datasets>=2.14.0

# Or reinstall all requirements
pip install -r requirements.txt --upgrade
```

### Verify Installation

```bash
# Test imports
python -c "import accelerate; import datasets; print('✅ All dependencies installed')"

# Run tests
pytest tests/ -v
```

### Run Training

```bash
# Quick test (15 minutes)
./run_all_steps.sh quick

# Full training (1-4 hours)
./run_all_steps.sh
```

## Common Issues

### 1. Virtual Environment Not Activated

**Symptom**: Commands fail or use wrong Python version

**Fix**:
```bash
source .venv/bin/activate
```

### 2. Old Dependencies

**Symptom**: Version conflicts or import errors

**Fix**:
```bash
pip install -r requirements.txt --upgrade --force-reinstall
```

### 3. CUDA/GPU Issues

**Symptom**: Training is slow or GPU not detected

**Fix**:
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# If False, install CUDA-enabled PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 4. Out of Memory

**Symptom**: `RuntimeError: CUDA out of memory`

**Fix**: Edit `mcp.json` to reduce batch sizes:
```json
{
  "training_params": {
    "teacher_batch_size": 4,
    "student_batch_size": 8
  }
}
```

## Complete Fresh Install

If all else fails, start fresh:

```bash
# Remove virtual environment
rm -rf .venv

# Run setup script
./setup.sh

# Verify
pytest tests/ -v
```

## Need Help?

- Check [docs/FAQ.md](docs/FAQ.md) for more solutions
- Open an issue: https://github.com/NorthernTribe-Research/NTMM/issues
- See [QUICKSTART.md](QUICKSTART.md) for detailed setup guide

---

**Quick Command Summary**:
```bash
source .venv/bin/activate
pip install -r requirements.txt --upgrade
pytest tests/ -v
./run_all_steps.sh quick
```
