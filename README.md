# LAVENDER
## Spatial AI System — Complete Build Guide

A personal AI built into a table. Voice-activated. Holographic display. Hand gesture control. Persistent memory. Local-first — no cloud required beyond optional ElevenLabs TTS.

---

## What It Is

Lavender is a complete AI workstation companion. She lives at your desk, speaks when spoken to, occasionally speaks first, controls your smart home, reads your screen, understands what you're pointing at, and remembers everything from session to session.

**Five personalities:**
- **IRIS** — Cold. Minimal. Speaks only when necessary. Sub-sentence responses.
- **NOVA** — Warm, curious, conversational. The default. Good for daily use.
- **VECTOR** — Technical. Shows reasoning. Epoch time display. Best for deep work.
- **SOLACE** — Slow, gentle, unhurried ambient presence. Late-night mode.
- **LILAC** — Unfiltered. Zero patience for lazy questions. Three-tier roast system.

---

## Hardware

### Required
| Component | Spec | Est. Cost (India) |
|-----------|------|-------------------|
| Compute   | Ubuntu 22.04, 64GB RAM, NVIDIA RTX 3090 or 4090 | ₹1.8–2.5L |
| Microphone | Array mic (ReSpeaker 4-mic HAT or USB equivalent) | ₹2–4K |
| Speakers  | Stereo bookshelf, powered | ₹4–8K |
| Table     | Custom build or modified IKEA ALEX desk | ₹8–15K |

### For Full Feature Set
| Component | Use | Est. Cost |
|-----------|-----|-----------|
| Intel RealSense D435i | Hand gesture + depth | ₹18–22K |
| Projector (ultra-short throw) | Holographic panel | ₹25–45K |
| Holographic film (Peppers Ghost) | Panel display surface | ₹3–6K |
| LED fog machine (ultrasonic) | Fog volume display | ₹2–4K |
| Capacitive strip | Surface controls (vol, personality) | ₹1–2K |
| Raspberry Pi 4 | Home Assistant controller | ₹4–6K |

**Total range: ₹3.8L–4.5L** for complete setup.
**Minimal (voice only, no display): ~₹2L**

---

## Software Stack

| Layer | Component |
|-------|-----------|
| Wake word + STT | Faster-Whisper (large-v3) |
| Intent routing | Mistral 7B (local, Ollama) |
| Reasoning | LLaMA 3.1 70B (local, Ollama) |
| Vision | LLaVA 13B (local, Ollama) |
| TTS | ElevenLabs API (or pyttsx3 fallback) |
| Memory | ChromaDB (episodic) + SQLite (semantic) |
| Tools | LangGraph ReAct agent |
| Hologram | Unity 2022.3 + NativeWebSocket |
| Gesture | MediaPipe + pyrealsense2 |
| Home control | Home Assistant REST API |
| Service | systemd |
| Dashboard | FastAPI + vanilla JS |

---

## Quick Start (Voice Only, No Hardware)

```bash
# 1. Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 2. Pull required models
ollama pull llama3.1:70b-instruct-q4_K_M
ollama pull mistral:7b-instruct-q4_K_M

# 3. Clone / extract Lavender
cd /opt
unzip lavender_complete.zip
cd lavender

# 4. Set up Python environment
python3 -m venv venv
source venv/bin/activate
./scripts/setup.sh

# 5. Configure
cp config/.env.example config/.env
# Edit config/.env — add ELEVENLABS_API_KEY if you have one
# ElevenLabs is optional — pyttsx3 is used as fallback

# 6. Run in text mode (no microphone needed)
python core/lavender.py --text --no-gesture --no-hologram

# 7. Run with voice (needs microphone)
python core/lavender.py --no-gesture --no-hologram
```

---

## Configuration

### `config/.env`
```bash
# ElevenLabs (optional — TTS quality)
ELEVENLABS_API_KEY=your_key_here
VOICE_IRIS=voice_id_here
VOICE_NOVA=voice_id_here
VOICE_VECTOR=voice_id_here
VOICE_SOLACE=voice_id_here
VOICE_LILAC=voice_id_here

# Ollama
OLLAMA_BASE_URL=http://localhost:11434

# Home Assistant (optional — device control)
HA_URL=http://homeassistant.local:8123
HA_TOKEN=your_long_lived_token_here

# Logging
LOG_LEVEL=INFO
```

### `config/lavender.yaml`
Key settings:
```yaml
system:
  default_personality: nova
  wake_words: ["hey lavender", "lavender"]
  silence_threshold_seconds: 1.2

models:
  primary: llama3.1:70b-instruct-q4_K_M
  router:  mistral:7b-instruct-q4_K_M

memory:
  max_working_memory_turns: 20
  episodic_db_path: "/opt/lavender/memory/chroma"
  semantic_db_path: "/opt/lavender/memory/semantic.db"
  decay_half_life_days: 45

audio:
  whisper_model: large-v3
  sample_rate: 16000
  output_volume: 1.0

tools:
  enable_home:        true
  enable_code_runner: true
  enable_web:         true

display:
  renderer_host: localhost
  renderer_port: 8765
```

---

## Running Commands

```bash
# Text mode (development, no hardware)
python core/lavender.py --text --no-gesture --no-hologram

# Voice, no display
python core/lavender.py --no-hologram

# Voice + gesture, no display
python core/lavender.py --no-hologram

# Full system
python core/lavender.py

# Specific personality
python core/lavender.py --personality lilac

# CPU inference (no GPU)
python core/lavender.py --cpu --text

# Dashboard (separate terminal)
python dashboard/app.py
# Open http://localhost:7860
```

---

## Production Deployment (systemd)

```bash
# Install both services
sudo ./scripts/install_service.sh

# Install dashboard service manually
sudo cp scripts/lavender-dashboard.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable lavender-dashboard
sudo systemctl start lavender

# Management
sudo systemctl start lavender
sudo systemctl stop lavender
sudo systemctl restart lavender
sudo systemctl status lavender
journalctl -u lavender -f          # live logs
journalctl -u lavender-dashboard -f
```

---

## Testing Each Subsystem

Run these independently to verify each milestone works before combining:

```bash
# M1 — Brain only (conversational, no tools)
python core/brain.py

# M2 — Memory system
python core/memory.py
python core/summarizer.py

# M3 — Hologram WebSocket server
python core/hologram.py
# Then connect Unity renderer — should see personality cycling

# M4 — Tools
python tools/code_runner.py        # sandboxed Python execution
python tools/web_tools.py          # DuckDuckGo + Open-Meteo weather
python tools/home_control.py       # Home Assistant (needs HA_URL + HA_TOKEN)
python tools/vision.py             # LLaVA (needs: ollama pull llava:13b)

# M5 — Gesture (mock mode, no hardware)
python perception/gesture.py --mock

# M6 — Proactive engine
# Start Lavender in text mode, wait a few minutes, it will speak unprompted

# Dashboard
python dashboard/app.py
```

---

## Gesture Reference

Gestures are recognized in two zones:

**Panel Zone** (in front of the holographic display):
| Gesture | Action |
|---------|--------|
| Point (index extended) | Cursor / hover |
| Press (finger crosses panel plane) | Select |
| Swipe left / right | Next / previous panel |
| Swipe up / down | Scroll |
| Open hand | Dismiss panel |
| Pinch expand | Zoom in |
| Pinch contract | Zoom out |

**Fog Zone** (above the ultrasonic well):
| Gesture | Action |
|---------|--------|
| Grab (fist in fog) | Grab object |
| Rotate CW / CCW | Rotate object |
| Two hands apart | Scale up |
| Two hands together | Scale down |
| Fling (throw) | Dismiss fog object |

---

## Surface Strip Controls (Capacitive)

Wire the strip to GPIO pins and call `fusion.process_surface()` from your driver:

| Position | Action |
|----------|--------|
| Far left | Emergency mute (interrupts immediately) |
| Left     | Volume down |
| Right    | Volume up |
| Far right | Cycle personality |
| Center tap | Pause / resume |

---

## Memory System

Lavender remembers two ways:

**Episodic** (ChromaDB) — session summaries written on shutdown:
- Semantic search: "what happened last time I worked on the API?"
- Relevance decays over 45 days by default
- Stored at: `memory/chroma/`

**Semantic** (SQLite) — structured facts extracted from conversations:
- Categories: preference, project, person, routine, decision, system, general
- Example: `preference.dark_mode = yes`, `project.current = Lavender build`
- Stored at: `memory/semantic.db`

View and edit memory at: `http://localhost:7860`

---

## Adding a New Personality

1. Open `core/personality.py`
2. Add a new `PersonalityConfig` entry to the `PERSONALITIES` dict
3. Add color + icon to `PERSONALITY_COLORS` / `PERSONALITY_ICONS` in `lavender.py`
4. Add theme entry to `UIConfig.cs` (Unity)
5. Add voice ID to `.env`
6. Add proactivity level to `PERSONALITY_PROACTIVITY` in `proactive.py`

---

## Adding a New Tool

1. Implement the function in `tools/` (or inline)
2. In `tools/tool_registry.py`, add a `@tool`-decorated wrapper with a clear docstring
3. Add it to the `tools` list in `build_toolkit()`
4. Add its intent type to `TOOL_INTENTS` in `core/brain.py` if it needs a new intent category

---

## Troubleshooting

**"Ollama not reachable"**
```bash
ollama serve
# or
sudo systemctl start ollama
```

**"Model not found"**
```bash
ollama list                              # see what's loaded
ollama pull llama3.1:70b-instruct-q4_K_M
ollama pull mistral:7b-instruct-q4_K_M
```

**"Voice not working"** — check audio devices:
```bash
python -c "import sounddevice; print(sounddevice.query_devices())"
# Set input_device and output_device in lavender.yaml
```

**"RealSense not detected"**
```bash
rs-enumerate-devices          # should list your camera
# If missing: sudo apt install librealsense2-utils
# pyrealsense2 needs the system library version to match
```

**"ElevenLabs voices flat / wrong"** — each personality needs its own voice ID in `.env`.
If IDs are blank, Lavender falls back to `pyttsx3` (robotic but functional).

**High RAM usage** — LLaMA 70B at Q4 uses ~45GB. If you have 32GB:
```yaml
# In lavender.yaml, switch to smaller model:
models:
  primary: llama3.1:8b-instruct-q4_K_M
```

**Unity renderer not connecting** — verify:
- Python WebSocket server is running (hologram.py log should say "listening on ws://localhost:8765")
- Unity `LavenderDirectorClient` Inspector shows correct URL
- No firewall blocking port 8765

---

## File Map

```
lavender/
├── config/
│   ├── .env.example          API keys, device IDs
│   └── lavender.yaml         All system config
│
├── core/
│   ├── lavender.py           Main entry point — wires everything together
│   ├── brain.py              LLM reasoning, intent routing, LangGraph agent
│   ├── personality.py        5 personality configs and switching logic
│   ├── memory.py             ChromaDB episodic + SQLite semantic memory
│   ├── summarizer.py         LLM-powered session → memory extraction
│   ├── voice_output.py       ElevenLabs TTS + pyttsx3 fallback
│   ├── hologram.py           WebSocket server → Unity renderer
│   ├── health.py             Component watchdog + auto-recovery
│   ├── intent_fusion.py      Voice + gesture + gaze → unified intent
│   └── proactive.py          Time/context triggers (Lavender speaks first)
│
├── perception/
│   ├── voice.py              Faster-Whisper STT + wake word + VAD
│   └── gesture.py            RealSense + MediaPipe hand tracking
│
├── tools/
│   ├── tool_registry.py      LangGraph tool definitions
│   ├── home_control.py       Home Assistant API wrapper
│   ├── code_runner.py        Sandboxed Python execution
│   ├── web_tools.py          DuckDuckGo search + Open-Meteo weather
│   └── vision.py             LLaVA image understanding
│
├── renderer/
│   └── Assets/Scripts/
│       ├── LavenderDirectorClient.cs   WebSocket → directive router
│       ├── ThemeManager.cs             Personality themes + transitions
│       ├── PanelManager.cs             Panel create/update/animate/destroy
│       ├── WaveformRenderer.cs         Idle/listen/think/speak waveform
│       ├── AmbientDisplay.cs           Clock, stats, state indicator
│       └── UIConfig.cs                 All colors, timing, constants
│
├── dashboard/
│   ├── app.py                FastAPI REST API
│   └── static/index.html     Dashboard UI
│
└── scripts/
    ├── setup.sh              First-time installation
    ├── install_service.sh    Register systemd services
    ├── lavender.service       Main process service unit
    └── lavender-dashboard.service   Dashboard service unit
```

---

## Milestones Completed

| # | Name | What It Added |
|---|------|---------------|
| 1 | Voice Agent | Wake word, STT, 5 personalities, TTS, intent routing |
| 2 | Persistent Memory | Episodic (ChromaDB) + semantic (SQLite), session summarization |
| 3 | Hologram Renderer | Unity WebSocket renderer, 5 personality themes, waveform |
| 4 | Device Control | Home Assistant, sandboxed code runner, web search, LangGraph agent |
| 5 | Gesture + Service | RealSense hand tracking, intent fusion, health monitor, systemd |
| 6 | Intelligence + Dashboard | Proactive triggers, LLaVA vision, web dashboard |

---

*Built for a single table. Not for production. Not for scale. For one person.*
