#!/usr/bin/env bash
# NTMM Setup Script
# Sets up the development environment for NorthernTribe Medical Models

set -e

echo "=========================================="
echo "NTMM Setup - NorthernTribe Medical Models"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
    
    if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 10 ]; then
        PYTHON_CMD="python3"
        echo "✓ Python $PYTHON_VERSION found"
    else
        echo "✗ Python 3.10+ required, found $PYTHON_VERSION"
        exit 1
    fi
else
    echo "✗ Python 3 not found. Please install Python 3.10 or higher."
    exit 1
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ -d ".venv" ]; then
    echo "Virtual environment already exists. Skipping creation."
else
    $PYTHON_CMD -m venv .venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
echo "This may take a few minutes..."
pip install -e ".[dev]"
echo "✓ Dependencies installed"

# Create necessary directories
echo ""
echo "Creating project directories..."
mkdir -p data
mkdir -p saved_models
echo "✓ Directories created"

# Make scripts executable
echo ""
echo "Making scripts executable..."
chmod +x run_all_steps.sh
echo "✓ Scripts are executable"

# Run tests
echo ""
echo "Running tests to verify installation..."
pytest tests/ -v

echo ""
echo "=========================================="
echo "✓ Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Activate the virtual environment: source .venv/bin/activate"
echo "  2. Run the pipeline: ./run_all_steps.sh quick"
echo "  3. Check the output in saved_models/ntmm-student/"
echo ""
echo "For more information, see README.md"
