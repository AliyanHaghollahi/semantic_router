#!/bin/bash
# setup.sh — One-shot setup for the edge device (your laptop)
# Run: bash setup.sh

set -e

echo "════════════════════════════════════════════════════════"
echo "  Semantic Query Routing System — Setup"
echo "  Edge device: laptop (simulation + Ollama)"
echo "════════════════════════════════════════════════════════"

# Python deps
echo ""
echo "[1/5] Installing Python dependencies..."
pip install -r requirements.txt

# spaCy model
echo ""
echo "[2/5] Downloading spaCy English model..."
python -m spacy download en_core_web_sm

# Create directories
echo ""
echo "[3/5] Creating data directories..."
mkdir -p data dataset models/classifier

# Copy config
if [ ! -f config.yaml ]; then
    cp config.example.yaml config.yaml
    echo "  ✓ config.yaml created (edit fog_server_url if using a real fog server)"
else
    echo "  config.yaml already exists — skipping"
fi

# Seed stores
echo ""
echo "[4/5] Seeding edge context store and training data..."
python scripts/seed_edge_db.py

# Train classifier
echo ""
echo "[5/5] Training classifier on seed data..."
python -c "
import sys
sys.path.insert(0, '.')
from router.classifier import QueryClassifier
clf = QueryClassifier()
clf.load_or_train(dataset_path='dataset/training_data.json')
print('  ✓ Classifier trained and saved.')
"

echo ""
echo "════════════════════════════════════════════════════════"
echo "  ✓ Setup complete!"
echo ""
echo "  Next steps:"
echo ""
echo "  1. (Optional) Install Ollama for real LLM inference:"
echo "     https://ollama.ai"
echo "     ollama pull llama3.2:3b"
echo ""
echo "  2. (Optional) Configure fog server:"
echo "     Edit config.yaml → fog_server_url"
echo "     On fog server: python fog/server.py"
echo "     On fog server: ollama pull llama3.2-vision:11b"
echo ""
echo "  3. Run the simulation:"
echo "     python run_simulation.py          # interactive"
echo "     python run_simulation.py --batch  # all demo queries"
echo "     python -m pytest tests/ -v       # test suite"
echo "════════════════════════════════════════════════════════"
