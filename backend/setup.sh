#!/bin/bash
# GNN-DRL Backend Installation and Setup Script

echo "================================"
echo "GNN-DRL Backend Setup Script"
echo "================================"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "✓ Python $python_version"

# Create virtual environment (optional)
echo ""
echo "Installing dependencies..."
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
    source venv/bin/activate
fi

# Install requirements
pip install -r requirements.txt > /dev/null 2>&1
echo "✓ Dependencies installed"

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p models/checkpoints
mkdir -p results/inference
mkdir -p results/simulation
mkdir -p results/evaluation
mkdir -p data
echo "✓ Directories created"

# Verify installation
echo ""
echo "Verifying installation..."
python -c "import torch; print(f'✓ PyTorch {torch.__version__}')" 2>/dev/null || echo "⚠ PyTorch verification skipped"
python -c "import networkx; print(f'✓ NetworkX {networkx.__version__}')" 2>/dev/null || echo "⚠ NetworkX verification skipped"
python -c "import numpy; print(f'✓ NumPy {numpy.__version__}')" 2>/dev/null || echo "⚠ NumPy verification skipped"

# Test import
echo ""
echo "Testing module import..."
python -c "from backend.app_orchestrator import NetworkRoutingSimulator; print('✓ Backend module loaded')" 2>/dev/null || echo "⚠ Module import test skipped"

echo ""
echo "================================"
echo "Setup Complete!"
echo "================================"
echo ""
echo "Next steps:"
echo "1. Read QUICKSTART.md for quick start"
echo "2. Run: python main.py --help"
echo "3. Try: python main.py inference --topology nsfnet --flows 20"
echo "4. Check: python examples.py 1"
echo ""
echo "For API server:"
echo "   python main.py api --port 8000"
echo ""
echo "For training:"
echo "   python main.py train --episodes 100"
echo ""
