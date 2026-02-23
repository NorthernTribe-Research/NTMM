#!/usr/bin/env bash
# Run the full pipeline step by step. Use from project root:
#   chmod +x run_all_steps.sh && ./run_all_steps.sh
# Optional: ./run_all_steps.sh quick   (small samples for testing)
# Requires: Python with torch (e.g. pip install -r requirements.txt, or use system python3)

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
# Prefer python3 if it has torch; else .venv
if python3 -c "import torch" 2>/dev/null; then
  PY="python3"
else
  PY=".venv/bin/python"
fi
QUICK="${1:-}"
if [ -n "$QUICK" ]; then
  EXTRA_TRAIN="--max-train-samples 64 --max-validation-samples 16"
  EXTRA_EVAL="--max-test-samples 32"
  echo "Quick mode: small samples (64 train, 16 val, 32 test)"
else
  EXTRA_TRAIN=""
  EXTRA_EVAL=""
fi

echo "=============================================="
echo "Step 0: Check dependencies"
echo "=============================================="
if ! $PY -c "import torch" 2>/dev/null; then
  echo "ERROR: torch not found. Run: pip install -r requirements.txt"
  echo "       (Or for CPU-only: pip install torch --index-url https://download.pytorch.org/whl/cpu)"
  exit 1
fi
echo "OK: torch available"

echo ""
echo "=============================================="
echo "Step 1: Prepare data (train/validation/test CSV)"
echo "=============================================="
if [ -f "data/train.csv" ] && [ -f "data/validation.csv" ] && [ -f "data/test.csv" ]; then
  echo "Data already exists. Skip with: --skip-if-exists or delete data/*.csv to regenerate."
  $PY src/prepare_data.py --config mcp.json --skip-if-exists
else
  $PY src/prepare_data.py --config mcp.json
fi
echo "Step 1 done."

echo ""
echo "=============================================="
echo "Step 2: Train teacher model"
echo "=============================================="
$PY src/train_teacher.py --config mcp.json $EXTRA_TRAIN
echo "Step 2 done."

echo ""
echo "=============================================="
echo "Step 3: Distill student model"
echo "=============================================="
$PY src/distil_student.py --config mcp.json $EXTRA_TRAIN
echo "Step 3 done."

echo ""
echo "=============================================="
echo "Step 4: Evaluate student model"
echo "=============================================="
$PY src/evaluate_student.py --config mcp.json $EXTRA_EVAL
echo "Step 4 done."

echo ""
echo "=============================================="
echo "All steps completed."
echo "=============================================="
