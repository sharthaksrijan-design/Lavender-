"""
LAVENDER — Web Dashboard
dashboard/app.py

A local web dashboard for managing Lavender without talking to her.

Runs on http://localhost:7860 (separate from the main process).
Accesses the same memory database and log files.

Routes:
  GET  /                      — dashboard HTML
  GET  /api/status            — system health + current state
  GET  /api/memory/episodes   — all episodic memories
  GET  /api/memory/facts      — all semantic facts
  POST /api/memory/facts      — add or update a fact
  DELETE /api/memory/facts/{id} — delete a fact
  DELETE /api/memory/episodes/{id} — delete an episode
  GET  /api/logs              — last N log lines
  GET  /api/personalities     — personality descriptions
  POST /api/personality       — switch personality (if Lavender is running)
  GET  /api/session           — current session info
  POST /api/note              — inject a note directly into memory
  GET  /api/health            — component health status

Start with:
  python dashboard/app.py
  # Then open http://localhost:7860
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ── PATH SETUP ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / "config" / ".env")

import yaml
with open(ROOT / "config" / "lavender.yaml") as f:
    CONFIG = yaml.safe_load(f)

logger = logging.getLogger("lavender.dashboard")

# ── APP ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Lavender Dashboard",
    description="Local management interface for the Lavender AI system",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── MEMORY ACCESS ─────────────────────────────────────────────────────────────
def get_memory():
    from core.memory import LavenderMemory
    return LavenderMemory(
        episodic_db_path=CONFIG["memory"]["episodic_db_path"],
        semantic_db_path=CONFIG["memory"]["semantic_db_path"],
    )


# ── REQUEST MODELS ────────────────────────────────────────────────────────────

class FactCreate(BaseModel):
    category: str
    key:      str
    value:    str
    confidence: float = 1.0
    source:  str = "dashboard"

class NoteCreate(BaseModel):
    text: str
    category: str = "general"

class PersonalitySwitch(BaseModel):
    personality: str


# ── ROUTES ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the dashboard HTML."""
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text())
    return HTMLResponse("<h1>Lavender Dashboard</h1><p>index.html not found.</p>")


@app.get("/api/status")
async def get_status():
    """Overall system status."""
    import httpx
    import psutil

    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_ok = False
    ollama_models = []
    try:
        r = httpx.get(f"{ollama_url}/api/tags", timeout=2.0)
        if r.status_code == 200:
            ollama_ok = True
            ollama_models = [m["name"] for m in r.json().get("models", [])]
    except Exception:
        pass

    memory = get_memory()

    try:
        cpu = psutil.cpu_percent(interval=0.1)
        ram = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
    except Exception:
        cpu, ram, disk = 0, None, None

    gpu_pct = 0.0
    gpu_vram_used = 0
    gpu_vram_total = 0
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2
        )
        parts = result.stdout.strip().split(", ")
        if len(parts) == 3:
            gpu_pct       = float(parts[0])
            gpu_vram_used = int(parts[1])
            gpu_vram_total = int(parts[2])
    except Exception:
        pass

    return {
        "ollama": {
            "reachable": ollama_ok,
            "models":    ollama_models,
            "url":       ollama_url,
        },
        "memory": {
            "episodes": memory.episodic.count,
            "facts":    memory.semantic.count,
        },
        "system": {
            "cpu_pct":       cpu,
            "ram_used_gb":   round(ram.used / 1e9, 1) if ram else 0,
            "ram_total_gb":  round(ram.total / 1e9, 1) if ram else 0,
            "ram_pct":       ram.percent if ram else 0,
            "disk_used_gb":  round(disk.used / 1e9, 1) if disk else 0,
            "disk_total_gb": round(disk.total / 1e9, 1) if disk else 0,
            "gpu_pct":       gpu_pct,
            "gpu_vram_used_mb":  gpu_vram_used,
            "gpu_vram_total_mb": gpu_vram_total,
        },
        "config": {
            "primary_model":  CONFIG["models"]["primary"],
            "router_model":   CONFIG["models"]["router"],
            "default_personality": CONFIG["system"]["default_personality"],
            "wake_words":     CONFIG["system"]["wake_words"],
        }
    }


@app.get("/api/memory/episodes")
async def get_episodes(limit: int = 50):
    """All episodic memories, most recent first."""
    memory = get_memory()
    episodes = memory.episodic.get_all()[:limit]
    return {"episodes": episodes, "total": memory.episodic.count}


@app.delete("/api/memory/episodes/{episode_id}")
async def delete_episode(episode_id: str):
    """Delete a specific episode by ID."""
    memory = get_memory()
    memory.episodic.delete(episode_id)
    return {"deleted": episode_id}


@app.delete("/api/memory/episodes")
async def clear_all_episodes():
    """Delete ALL episodic memories. Irreversible."""
    memory = get_memory()
    memory.episodic.clear_all()
    return {"cleared": True}


@app.get("/api/memory/facts")
async def get_facts(category: Optional[str] = None):
    """All semantic facts, optionally filtered by category."""
    memory = get_memory()
    if category:
        facts_dict = memory.semantic.get_category(category)
        facts = [{"category": category, "key": k, "value": v}
                 for k, v in facts_dict.items()]
    else:
        facts = memory.semantic.get_all()
    return {"facts": facts, "total": len(facts)}


@app.post("/api/memory/facts")
async def create_fact(fact: FactCreate):
    """Add or update a semantic fact."""
    memory = get_memory()
    memory.semantic.store(
        category=fact.category,
        key=fact.key,
        value=fact.value,
        confidence=fact.confidence,
        source=fact.source,
    )
    return {"stored": True, "category": fact.category, "key": fact.key}


@app.delete("/api/memory/facts/{category}/{key}")
async def delete_fact(category: str, key: str):
    """Delete a specific semantic fact."""
    memory = get_memory()
    deleted = memory.semantic.delete(category, key)
    if not deleted:
        raise HTTPException(status_code=404, detail="Fact not found")
    return {"deleted": f"{category}/{key}"}


@app.post("/api/memory/note")
async def add_note(note: NoteCreate):
    """
    Inject a note directly into episodic memory.
    Useful for bootstrapping memory before first session.
    """
    memory = get_memory()
    memory.episodic.store(
        summary=note.text,
        personality="system",
        tags=[note.category, "manual-note"],
        session_id=f"note_{int(__import__('time').time())}",
    )
    return {"stored": True}


@app.get("/api/memory/search")
async def search_memory(q: str):
    """Search across episodic and semantic memory."""
    memory = get_memory()
    episodes = memory.episodic.recall(q, n_results=10)
    facts    = memory.semantic.search(q)
    return {"query": q, "episodes": episodes, "facts": facts}


@app.get("/api/logs")
async def get_logs(lines: int = 100, level: Optional[str] = None):
    """Return the last N lines from the log file."""
    log_path = ROOT / "logs" / "lavender.log"
    if not log_path.exists():
        return {"lines": [], "path": str(log_path)}

    all_lines = log_path.read_text(errors="replace").splitlines()
    recent = all_lines[-lines:]

    if level:
        level_upper = level.upper()
        recent = [l for l in recent if level_upper in l]

    return {"lines": recent, "total_lines": len(all_lines)}


@app.get("/api/personalities")
async def get_personalities():
    """Return personality descriptions."""
    return {
        "personalities": {
            "iris": {
                "display_name": "IRIS",
                "tagline": "Minimal. Precise. Speaks only when necessary.",
                "color": "#4FC3F7",
            },
            "nova": {
                "display_name": "NOVA",
                "tagline": "Warm, conversational, curious. The default.",
                "color": "#FFB347",
            },
            "vector": {
                "display_name": "VECTOR",
                "tagline": "Technical, analytical, shows reasoning.",
                "color": "#00FF88",
            },
            "solace": {
                "display_name": "SOLACE",
                "tagline": "Gentle, unhurried, ambient presence.",
                "color": "#C9A0A0",
            },
            "lilac": {
                "display_name": "LILAC",
                "tagline": "Unfiltered intelligence. Zero patience for lazy questions.",
                "color": "#9B59B6",
            },
        }
    }


@app.get("/api/health")
async def get_health():
    """
    Component health status.
    Returns current status of all monitored components.
    """
    import httpx, socket

    checks = {}

    # Ollama
    try:
        r = httpx.get(
            f"{os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}/api/tags",
            timeout=2.0
        )
        checks["ollama"] = "healthy" if r.status_code == 200 else "degraded"
    except Exception:
        checks["ollama"] = "failed"

    # ElevenLabs
    api_key = os.getenv("ELEVENLABS_API_KEY", "")
    if api_key and api_key != "your_api_key_here":
        try:
            r = httpx.get(
                "https://api.elevenlabs.io/v1/user",
                headers={"xi-api-key": api_key},
                timeout=3.0,
            )
            checks["elevenlabs"] = "healthy" if r.status_code == 200 else "degraded"
        except Exception:
            checks["elevenlabs"] = "failed"
    else:
        checks["elevenlabs"] = "disabled"

    # Memory DB
    episodic_path = CONFIG["memory"]["episodic_db_path"]
    checks["memory_db"] = "healthy" if Path(episodic_path).exists() else "failed"

    # WebSocket renderer
    ws_port = CONFIG.get("display", {}).get("renderer_port", 8765)
    try:
        with socket.create_connection(("localhost", ws_port), timeout=1.0):
            checks["renderer"] = "healthy"
    except Exception:
        checks["renderer"] = "not_connected"

    # RealSense
    try:
        import pyrealsense2 as rs
        ctx = rs.context()
        checks["realsense"] = "healthy" if ctx.devices else "not_connected"
    except ImportError:
        checks["realsense"] = "not_installed"
    except Exception:
        checks["realsense"] = "failed"

    return {"components": checks}


@app.post("/api/personality")
async def switch_personality(req: PersonalitySwitch):
    """
    Switch personality. If Lavender is running, communicates via a shared
    state file that the main process polls. Otherwise updates config default.
    """
    valid = ["iris", "nova", "vector", "solace", "lilac"]
    if req.personality not in valid:
        raise HTTPException(400, f"Unknown personality. Valid: {valid}")

    # Write intent to a runtime state file
    state_path = ROOT / "config" / "runtime.yaml"
    runtime = {}
    if state_path.exists():
        with open(state_path) as f:
            runtime = yaml.safe_load(f) or {}

    runtime["requested_personality"] = req.personality

    with open(state_path, "w") as f:
        yaml.dump(runtime, f)

    return {"requested": req.personality,
            "note": "Lavender will switch at the next interaction if she is running."}


# ── ENTRYPOINT ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("\n  Lavender Dashboard")
    print("  Open: http://localhost:7860\n")
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="warning")
