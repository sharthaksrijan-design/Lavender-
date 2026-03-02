#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# LAVENDER — Milestone 1 Setup Script
# Run once on a fresh Ubuntu 22.04 install.
# Usage: chmod +x scripts/setup.sh && ./scripts/setup.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e  # Exit on any error

LAVENDER_ROOT=$(pwd)
PYTHON="python3.11"

echo ""
echo "  ██╗      █████╗ ██╗   ██╗███████╗███╗   ██╗██████╗ ███████╗██████╗ "
echo "  ██║     ██╔══██╗██║   ██║██╔════╝████╗  ██║██╔══██╗██╔════╝██╔══██╗"
echo "  ██║     ███████║██║   ██║█████╗  ██╔██╗ ██║██║  ██║█████╗  ██████╔╝"
echo "  ██║     ██╔══██║╚██╗ ██╔╝██╔══╝  ██║╚██╗██║██║  ██║██╔══╝  ██╔══██╗"
echo "  ███████╗██║  ██║ ╚████╔╝ ███████╗██║ ╚████║██████╔╝███████╗██║  ██║"
echo "  ╚══════╝╚═╝  ╚═╝  ╚═══╝  ╚══════╝╚═╝  ╚═══╝╚═════╝ ╚══════╝╚═╝  ╚═╝"
echo ""
echo "  Milestone 1 — Voice Agent Setup"
echo "  ─────────────────────────────────────────────────────────────────────"
echo ""

# ── 1. SYSTEM PACKAGES ───────────────────────────────────────────────────────
echo "[1/6] Installing system packages..."
sudo apt update -qq
sudo apt install -y \
    python3.11 python3.11-venv python3.11-dev \
    python3-pip \
    build-essential git curl wget \
    ffmpeg \
    portaudio19-dev \
    libasound2-dev \
    libsndfile1 \
    alsa-utils \
    pulseaudio \
    2>&1 | grep -E "(Installing|Unpacking|error)" || true

echo "    ✓ System packages ready"

# ── 2. NVIDIA / CUDA CHECK ───────────────────────────────────────────────────
echo ""
echo "[2/6] Checking GPU / CUDA..."
if command -v nvidia-smi &> /dev/null; then
    GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    echo "    ✓ GPU detected: $GPU"
    DEVICE="cuda"
    COMPUTE_TYPE="float16"
else
    echo "    ⚠  No NVIDIA GPU detected. Running on CPU."
    echo "    ⚠  LLM inference will be slow. Consider adding a GPU."
    DEVICE="cpu"
    COMPUTE_TYPE="int8"
fi

# ── 3. OLLAMA ────────────────────────────────────────────────────────────────
echo ""
echo "[3/6] Installing Ollama..."
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.ai/install.sh | sh
    echo "    ✓ Ollama installed"
else
    echo "    ✓ Ollama already installed"
fi

# Start Ollama in background
ollama serve &>/dev/null &
OLLAMA_PID=$!
sleep 3
echo "    ✓ Ollama running (PID $OLLAMA_PID)"

echo ""
echo "    Pulling models — this will take a while on first run."
echo "    Grab a coffee. Seriously."
echo ""

echo "    → Pulling mistral:7b (intent router, ~4GB)..."
ollama pull mistral:7b-instruct-q4_K_M

echo "    → Pulling llama3.1:70b (primary brain, ~40GB)..."
echo "      This one is large. Very large."
ollama pull llama3.1:70b-instruct-q4_K_M

echo "    → Pulling nomic-embed-text (memory embeddings)..."
ollama pull nomic-embed-text

echo "    ✓ All models ready"

# ── 4. PROJECT DIRECTORY ─────────────────────────────────────────────────────
echo ""
echo "[4/6] Setting up project directory at $LAVENDER_ROOT..."
mkdir -p $LAVENDER_ROOT/{core,perception,config,memory,tools,logs,scripts}
echo "    ✓ Directory structure created"

# ── 5. PYTHON ENVIRONMENT ────────────────────────────────────────────────────
echo ""
echo "[5/6] Creating Python virtual environment..."
$PYTHON -m venv $LAVENDER_ROOT/venv
source $LAVENDER_ROOT/venv/bin/activate

pip install --quiet --upgrade pip

echo "    Installing Python dependencies..."
pip install --quiet \
    fastapi \
    uvicorn \
    langchain \
    langchain-community \
    langchain-ollama \
    langgraph \
    chromadb \
    elevenlabs \
    faster-whisper \
    sounddevice \
    soundfile \
    numpy \
    pyaudio \
    pydantic \
    python-dotenv \
    httpx \
    aiohttp \
    pyyaml \
    rich \
    webrtcvad \
    websockets \
    psutil \
    mediapipe \
    pyrealsense2 \
    icalendar \
    dateparser \
    openwakeword

echo "    ✓ Python environment ready"
echo "    Location: $LAVENDER_ROOT/venv"

# ── 6. CONFIG ────────────────────────────────────────────────────────────────
echo ""
echo "[6/6] Writing runtime config..."

# Write device info to config so brain.py can read it
cat > $LAVENDER_ROOT/config/runtime.yaml << EOF
device: "$DEVICE"
compute_type: "$COMPUTE_TYPE"
whisper_model: "medium"
EOF

echo "    ✓ Runtime config written"

# ── DONE ─────────────────────────────────────────────────────────────────────
echo ""
echo "  ─────────────────────────────────────────────────────────────────────"
echo "  Setup complete."
echo ""
echo "  Next steps:"
echo "    1. Copy config/.env.example to config/.env"
echo "    2. Add your ElevenLabs API key to config/.env"
echo "    3. Add ElevenLabs voice IDs for each personality"
  echo "    4. Run: source $LAVENDER_ROOT/venv/bin/activate"
  echo "    5. Run: python $LAVENDER_ROOT/core/lavender.py"
echo ""
echo "  To activate the environment in future sessions:"
  echo "    source $LAVENDER_ROOT/venv/bin/activate"
echo "  ─────────────────────────────────────────────────────────────────────"
echo ""

# ── DASHBOARD DEPENDENCIES ──────────────────────────────────────────────────
pip install --quiet fastapi uvicorn[standard] mss Pillow

# ── VISION MODEL (optional — large download) ────────────────────────────────
echo ""
echo "Vision model (LLaVA) is optional but enables 'look at this' features."
read -p "Download llava:13b now? (~8GB) [y/N]: " yn
if [[ "$yn" =~ ^[Yy]$ ]]; then
    ollama pull llava:13b
fi
