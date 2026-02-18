#!/usr/bin/env bash
# =============================================================
# GGF LLM Systems Case — Evaluation Runner (Linux/macOS)
# =============================================================
# Usage: ./eval/run_eval.sh
# =============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=========================================="
echo " GGF LLM Systems Case — Evaluation Runner"
echo "=========================================="
echo ""

# Check .env
if [ ! -f "$REPO_ROOT/.env" ]; then
    echo "[WARN] No .env file found. Copying .env.example..."
    cp "$REPO_ROOT/.env.example" "$REPO_ROOT/.env"
    echo "[WARN] Please edit .env with your actual API key before running."
    exit 1
fi

# Source .env
set -a
source "$REPO_ROOT/.env"
set +a

# Check API key
if [ -z "${OPENAI_API_KEY:-}" ] || [ "$OPENAI_API_KEY" = "YOUR_KEY_HERE" ]; then
    echo "[ERROR] OPENAI_API_KEY is not set. Please update .env"
    exit 1
fi

# Check Node
if ! command -v node &> /dev/null; then
    echo "[ERROR] Node.js is required but not found."
    exit 1
fi
echo "[OK] Node.js $(node --version)"

# Check Python
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "[ERROR] Python 3.11+ is required but not found."
    exit 1
fi
echo "[OK] Python $($PYTHON_CMD --version)"

# Install Node dependencies
echo ""
echo "[STEP 1] Installing Node dependencies..."
cd "$REPO_ROOT/ggf-mini-game"
npm install --silent

# Build baseline
echo ""
echo "[STEP 2] Building baseline..."
npm run build
echo "[OK] Baseline build succeeded"

# Run baseline sanity
echo ""
echo "[STEP 3] Running baseline sanity check..."
cd "$REPO_ROOT"
node eval/checks/baseline_sanity.mjs

# Install Python dependencies
echo ""
echo "[STEP 4] Installing Python dependencies..."
cd "$REPO_ROOT/solution"
if [ ! -d ".venv" ]; then
    $PYTHON_CMD -m venv .venv
fi
source .venv/bin/activate
pip install -e . --quiet

# Run evaluation
echo ""
echo "[STEP 5] Running evaluation..."
cd "$REPO_ROOT"
$PYTHON_CMD -m ggf_case.cli run-eval --output-dir eval/outputs

echo ""
echo "=========================================="
echo " Evaluation complete!"
echo " Results: eval/outputs/"
echo "=========================================="
