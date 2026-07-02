#!/usr/bin/env bash
# Setup script for a new benchmark machine.
# Run once before bench.py. Installs uv, pulls Ollama models.
set -e

echo "=== LLM MP3 perf test setup ==="

# --- uv ---
if command -v uv &>/dev/null; then
    echo "uv already installed: $(uv --version)"
else
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    echo "uv installed: $(uv --version)"
fi

# --- Python deps (for fallback without uv) ---
if command -v pip &>/dev/null; then
    pip install -q -U ollama pydantic
fi

# --- Ollama ---
if command -v ollama &>/dev/null; then
    echo "Ollama already installed: $(ollama --version)"
else
    # Ollama's installer needs zstd to extract; minimal images (e.g. Runpod) often lack it.
    if ! command -v zstd &>/dev/null; then
        echo "Installing zstd (required by Ollama installer)..."
        if command -v apt-get &>/dev/null; then
            apt-get update && apt-get install -y zstd
        elif command -v dnf &>/dev/null; then
            dnf install -y zstd
        elif command -v pacman &>/dev/null; then
            pacman -Sy --noconfirm zstd
        else
            echo "ERROR: zstd not found and no supported package manager detected. Please install zstd manually."
            exit 1
        fi
    fi

    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

# Ensure Ollama is running
if ! ollama list &>/dev/null; then
    echo "Starting Ollama..."
    ollama serve &
    sleep 3
fi

# --- Pull models ---
# Smoke-test model only (see RESULTS.md) — small and fast, good for
# confirming the pipeline runs end-to-end on new hardware.
MODELS=(
    "qwen3:8b"
)

echo ""
echo "Pulling models (this may take a while on first run)..."
for model in "${MODELS[@]}"; do
    echo "  → $model"
    ollama pull "$model"
done

# --- Check songs_export.jsonl ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JSONL="$SCRIPT_DIR/songs_export.jsonl"

echo ""
if [ -f "$JSONL" ]; then
    COUNT=$(wc -l < "$JSONL")
    echo "songs_export.jsonl found: $COUNT songs"
else
    echo "WARNING: songs_export.jsonl not found at $JSONL — it should have been included in this repo."
fi

echo ""
echo "=== Setup complete. Run the benchmark with: ==="
echo "  uv run $SCRIPT_DIR/bench.py --force-batch-size 4 --machine <your-machine-name>"
