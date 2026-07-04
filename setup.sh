#!/usr/bin/env bash
# Setup script for a new benchmark machine.
# Run once before bench.py. Installs uv and Ollama (no models — pull what you need).
set -e

echo "=== LLM MP3 perf test setup ==="

# --- uv ---
if command -v uv &>/dev/null; then
    echo "uv already installed: $(uv --version)"
else
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env"
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
    ollama serve >/tmp/ollama-serve.log 2>&1 &
    disown
    sleep 3
fi

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
echo "=== Setup complete. Run these commands: ==="
echo "  source ./setup.sh"
echo "  ollama pull qwen3.6:35b"
echo "  uv run bench.py --force-batch-size 4 --songs 4 --models qwen3.6:35b"
