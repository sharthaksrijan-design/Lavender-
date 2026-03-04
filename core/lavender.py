"""
LAVENDER — Main Entry Point
core/lavender.py

The complete system loop. Wires together all Milestone components:

  M1: VoicePerception → LavenderBrain → VoiceOutput
  M2: LavenderMemory + SessionSummarizer (session persistence)
  M3: HologramDirector + AmbientDataPusher (Unity renderer)
  M4: Tool registry (Home Assistant, code runner, web search)
  M5: GesturePerception + IntentFusion + HealthMonitor

Input now flows through IntentFusion which merges voice, gesture,
and gaze before reaching the brain. The main loop listens to the
fusion layer rather than directly to VoicePerception.

Usage:
  python core/lavender.py
  python core/lavender.py --personality lilac
  python core/lavender.py --text          (no microphone)
  python core/lavender.py --no-gesture    (skip RealSense)
  python core/lavender.py --no-hologram   (skip WebSocket renderer)
  python core/lavender.py --cpu           (force CPU inference)
"""

import os
import sys
import signal
import logging
import threading
import queue
import uuid
import argparse
import time
import yaml
import asyncio
from datetime import datetime
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.text import Text
from rich.panel import Panel

# ── PATH SETUP ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from perception.voice import VoicePerception
from core.brain import LavenderBrain
from core.voice_output import VoiceOutput
from core.memory import LavenderMemory
from core.hologram import HologramDirector, AmbientDataPusher, SystemState, WaveformState
from core.health import build_standard_monitors
from core.intent_fusion import IntentFusion, FusedIntent, Modality, SurfaceControl
from tools.tool_registry import build_toolkit, describe_toolkit
from core.proactive import ProactiveEngine, TriggerPriority
from core.state import instance as state_engine, UserState, SystemStatus
from core.executor import TaskExecutor, AutonomousTask, TaskStep, TaskPriority, TaskStatus

# ── CONFIG ───────────────────────────────────────────────────────────────────
CONFIG_PATH = ROOT / "config" / "lavender.yaml"
ENV_PATH    = ROOT / "config" / ".env"

load_dotenv(ENV_PATH)

with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

# ── LOGGING ───────────────────────────────────────────────────────────────────
log_level = os.getenv("LOG_LEVEL", CONFIG["system"].get("log_level", "INFO"))
(ROOT / "logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "logs" / "lavender.log"),
    ]
)
logger = logging.getLogger("lavender")
console = Console()

# ── TERMINAL COLORS ───────────────────────────────────────────────────────────
PERSONALITY_COLORS = {
    "iris": "cyan", "nova": "yellow", "vector": "green",
    "solace": "magenta", "lilac": "purple",
}
PERSONALITY_ICONS = {
    "iris": "◆", "nova": "●", "vector": "▲", "solace": "◎", "lilac": "✦",
}


def print_banner():
    console.print("")
    console.print(Panel(
        Text.from_markup(
            "[bold magenta]L A V E N D E R[/bold magenta]\n"
            "[dim]Spatial AI System — Full Stack[/dim]"
        ),
        border_style="magenta",
        padding=(1, 4),
    ))
    console.print("")


def print_you(text: str):
    console.print(f"\n[bold white]  You ›[/bold white] [white]{text}[/white]")


def print_lavender(personality: str, text: str):
    c = PERSONALITY_COLORS.get(personality, "white")
    i = PERSONALITY_ICONS.get(personality, "●")
    console.print(f"\n[bold {c}]  {i} {personality.upper()} ›[/bold {c}] [{c}]{text}[/{c}]\n")


def print_status(personality: str, state: str):
    c = PERSONALITY_COLORS.get(personality, "white")
    i = PERSONALITY_ICONS.get(personality, "●")
    console.print(f"  [{c}]{i}[/{c}] [dim]{state}[/dim]", end="\r")


# ─────────────────────────────────────────────────────────────────────────────

class Lavender:
    def __init__(
        self,
        personality:  str  = None,
        use_cpu:      bool = False,
        text_mode:    bool = False,
        no_gesture:   bool = False,
        no_hologram:  bool = False,
    ):
        self.text_mode   = text_mode
        self.no_gesture  = no_gesture
        self.no_hologram = no_hologram
        self._running    = False

        start_personality = personality or CONFIG["system"]["default_personality"]

        # ── RUNTIME CONFIG ────────────────────────────────────────────────────
        runtime_path = ROOT / "config" / "runtime.yaml"
        if runtime_path.exists():
            with open(runtime_path) as f:
                runtime = yaml.safe_load(f)
            device       = "cpu" if use_cpu else runtime.get("device",       "cpu")
            compute_type = "int8" if use_cpu else runtime.get("compute_type", "int8")
        else:
            device, compute_type = ("cpu", "int8") if use_cpu else ("cuda", "float16")

        console.print("[dim]Initializing subsystems...[/dim]")

        # M2: MEMORY
        console.print("[dim]  memory...[/dim]", end="\r")
        self.memory = LavenderMemory(
            episodic_db_path=CONFIG["memory"]["episodic_db_path"],
            semantic_db_path=CONFIG["memory"]["semantic_db_path"],
        )

        # M4: TOOLS
        console.print("[dim]  tools...[/dim]  ", end="\r")
        tools_cfg  = CONFIG.get("tools", {})
        self.toolkit = build_toolkit(
            ha_url=os.getenv("HA_URL",    tools_cfg.get("ha_url", "")),
            ha_token=os.getenv("HA_TOKEN", tools_cfg.get("ha_token", "")),
            enable_code_runner=tools_cfg.get("enable_code_runner", True),
            enable_web=tools_cfg.get("enable_web", True),
            enable_home=tools_cfg.get("enable_home", True),
            enable_vision=tools_cfg.get("enable_vision", True),
        )

        # M1: BRAIN
        console.print("[dim]  brain...[/dim]  ", end="\r")
        self.brain = LavenderBrain(
            personality=start_personality,
            primary_model=CONFIG["models"]["primary"],
            router_model=CONFIG["models"]["router"],
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            max_working_memory=CONFIG["memory"]["max_working_memory_turns"],
            memory=self.memory,
            tools=self.toolkit,
            top_k_memories=CONFIG["memory"].get("top_k_memories", 3),
            intent_threshold=CONFIG["system"].get("intent_confidence_threshold", 0.75),
            personality_overrides=CONFIG.get("personalities", {}),
        )

        # M3: HOLOGRAM
        display_cfg = CONFIG.get("display", {})
        if not no_hologram:
            console.print("[dim]  hologram...[/dim]", end="\r")
            self.hologram = HologramDirector(
                host=display_cfg.get("renderer_host", "localhost"),
                port=display_cfg.get("renderer_port", 8765),
            )
            self.hologram.start()
            self._ambient_pusher = AmbientDataPusher(self.hologram, interval_seconds=5.0)
            self._ambient_pusher.start()
            self.hologram.set_personality(start_personality)
        else:
            self.hologram = None
            self._ambient_pusher = None

        # M1: VOICE OUTPUT
        console.print("[dim]  audio out...[/dim]", end="\r")
        self.voice_out = VoiceOutput(
            api_key=os.getenv("ELEVENLABS_API_KEY", ""),
            voice_ids={p: os.getenv(f"VOICE_{p.upper()}", "")
                       for p in ["iris", "nova", "vector", "solace", "lilac"]},
            output_device=CONFIG["audio"].get("output_device"),
            volume=CONFIG["audio"].get("output_volume", 1.0),
        )
        self.voice_out.set_personality(start_personality)

        # M1: VOICE INPUT
        if not text_mode:
            console.print("[dim]  audio in...[/dim] ", end="\r")
            try:
                self.voice_in = VoicePerception(
                    model_size=CONFIG["audio"]["whisper_model"],
                    device=device,
                    compute_type=compute_type,
                    sample_rate=CONFIG["audio"]["sample_rate"],
                    wake_words=CONFIG["system"]["wake_words"],
                    silence_threshold=CONFIG["system"]["silence_threshold_seconds"],
                    input_device=CONFIG["audio"].get("input_device"),
                    language=CONFIG["audio"].get("whisper_language", "en"),
                )
            except Exception as e:
                logger.error(f"Voice perception init failed: {e}")
                console.print(f"[yellow]⚠  Voice perception failed. Falling back to text mode.[/yellow]")
                self.voice_in = None
                self.text_mode = True
        else:
            self.voice_in = None

        # M5: GESTURE
        self.gesture_perception = None
        if not no_gesture and not text_mode:
            console.print("[dim]  gesture...[/dim] ", end="\r")
            try:
                from perception.gesture import GesturePerception
                self.gesture_perception = GesturePerception(mock=False)
                self.gesture_perception.start()
            except Exception as e:
                logger.warning(f"Gesture unavailable: {e}")

        # M5: INTENT FUSION
        self.fusion = IntentFusion()

        # Update State Engine with static config
        state_engine.state.active_personality = start_personality

        # M5: HEALTH MONITOR
        console.print("[dim]  health...[/dim]  ", end="\r")
        self.health = build_standard_monitors(
            ollama_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            elevenlabs_key=os.getenv("ELEVENLABS_API_KEY", ""),
            memory_path=CONFIG["memory"]["episodic_db_path"],
            ws_port=display_cfg.get("renderer_port", 8765),
            on_critical=self._on_health_critical,
            on_recovery=self._on_health_recovery,
        )
        self.health.start()

        # TASK EXECUTOR (Jarvis v2.0)
        self.executor = TaskExecutor(self.brain, self.memory)
        self.executor.start()

        # PROACTIVE ENGINE
        console.print("[dim]  proactive...[/dim]", end="\r")
        self.proactive = ProactiveEngine(
            on_trigger=self._on_proactive_trigger,
            get_personality=lambda: self.brain.personality_name,
            memory=self.memory,
            session_start=time.time(),
        )
        self.proactive.start()

        # Register existing calendar events as proactive triggers
        try:
            from tools.calendar import _load_events
            from datetime import datetime
            events = _load_events()
            logger.info(f"Loading {len(events)} calendar events for proactive triggers.")
            for event in events:
                try:
                    # Parse ISO string back to datetime
                    t = datetime.fromisoformat(event["start"])
                    # Only register for today
                    if t.date() == datetime.now().date():
                        self.proactive.add_calendar_trigger(event["title"], t)
                        logger.info(f"Registered proactive warning for: {event['title']} at {t.strftime('%H:%M')}")
                except Exception as e:
                    logger.debug(f"Skipping event {event.get('title')}: {e}")
        except Exception as e:
            logger.warning(f"Could not load calendar triggers: {e}")

        console.print("[dim]  all systems ready.[/dim]")
        console.print("")

    # ── INTENT HANDLER ────────────────────────────────────────────────────────

    def handle(self, intent: FusedIntent):
        if intent.is_system_control:
            self._handle_surface(intent.surface_control)
            return

        text = intent.text.strip()
        if not text:
            return

        personality_before = self.brain.personality_name

        print_you(text)
        state_engine.update_user_activity()

        # ── REFLEX CHECK ──
        reflex_response = self.brain._reflex_match(text)
        if reflex_response:
            logger.info("Reflex match found. Skipping streaming loop.")
            print_lavender(self.brain.personality_name, reflex_response)
            if self.hologram:
                self.hologram.on_wake()
                self.hologram.on_speaking(reflex_response)
            self.voice_out.speak(reflex_response)
            if self.hologram:
                self.hologram.on_done_speaking()
            self.brain._store_turn(text, reflex_response)
            return

        print_status(self.brain.personality_name, "thinking...")

        if self.hologram:
            self.hologram.on_wake()
            self.hologram.on_thinking()

        self.proactive.note_interaction()

        # ── JARVIS v2.0 AUTONOMOUS TASK ROUTING ──
        intent_result = self.brain.route(text)
        if intent_result.get("intent") == "operational_async":
            logger.info(f"Routing request to Autonomous Task Executor: {text}")

            # Use Brain to create a Task Object
            task = self.brain.create_autonomous_task(text)
            if task:
                self.executor.submit(task)

                resp = f"Autonomous execution started. Task ID: {task.task_id}. I'll notify you upon completion."
                print_lavender(self.brain.personality_name, resp)
                self.voice_out.speak(resp)
                return

        full_response = ""
        # Use sentence streaming to reduce perceived latency
        for chunk in self.brain.think_streaming(text):
            if not chunk: continue
            full_response += " " + chunk

            # Sync personality if it switched (only happens on first chunk usually)
            if self.brain.personality_name != personality_before:
                self.voice_out.set_personality(self.brain.personality_name)
                if self.hologram:
                    self.hologram.set_personality(self.brain.personality_name)
                personality_before = self.brain.personality_name

            print_lavender(self.brain.personality_name, chunk)

            if self.hologram:
                panel = self._response_to_panel(chunk)
                if panel:
                    self.hologram.show_panel(**panel)
                else:
                    self.hologram.on_speaking(chunk)

            self.voice_out.speak(chunk)

        if self.hologram:
            self.hologram.on_done_speaking()

    def _response_to_panel(self, response: str) -> Optional[dict]:
        """
        Convert a tool result response into a hologram panel directive.
        Returns show_panel kwargs or None (→ use default show_response).
        """
        text_lower = response.lower()

        # Weather response
        if any(w in text_lower for w in ("temperature", "°c", "humidity", "forecast", "rain")):
            return {
                "panel_id":   "weather",
                "content":    response,
                "content_type": "info",
                "title":      "Weather",
                "persist":    False,
            }

        # Search results
        if any(w in text_lower for w in ("according to", "search results", "found:")):
            return {
                "panel_id":   "search",
                "content":    response,
                "content_type": "list",
                "title":      "Search",
                "persist":    False,
            }

        # Code output
        if "```" in response or any(w in text_lower for w in ("output:", "result:", "executed")):
            return {
                "panel_id":   "code",
                "content":    response,
                "content_type": "code",
                "title":      "Code",
                "persist":    False,
            }

        # Calendar listing
        if any(w in text_lower for w in ("events today", "events tomorrow", "no events", " — ")):
            return {
                "panel_id":   "calendar",
                "content":    response,
                "content_type": "list",
                "title":      "Calendar",
                "persist":    False,
            }

        return None

    def _handle_surface(self, control: SurfaceControl):
        if control == SurfaceControl.EMERGENCY_MUTE:
            from core.safety import instance as safety
            safety.activate_hard_stop() # Escalate mute to hard stop
            self.voice_out.interrupt()
            if self.hologram:
                self.hologram.set_waveform(WaveformState.IDLE)
                self.hologram.show_alert("EMERGENCY STOP", "All actions blocked.", severity="critical")

        elif control == SurfaceControl.PAUSE_RESUME:
            pass  # Fusion layer handles state

        elif control in (SurfaceControl.VOLUME_UP, SurfaceControl.VOLUME_DOWN):
            self.voice_out.set_volume(self.fusion.current_volume)

        elif control == SurfaceControl.PERSONALITY_NEXT:
            name = self.fusion.next_personality
            response = self.brain.switch_personality(name)
            self.voice_out.set_personality(name)
            if self.hologram:
                self.hologram.set_personality(name)
            print_lavender(name, response)
            self.voice_out.speak(response)

    # ── FEEDER THREADS ────────────────────────────────────────────────────────

    def _voice_feeder(self):
        for utterance in self.voice_in.listen():
            if not self._running:
                break
            if self.voice_out.is_speaking():
                self.voice_out.interrupt()
                time.sleep(0.1)
            self.fusion.process_voice(utterance)

    def _gesture_feeder(self):
        for event in self.gesture_perception.gesture():
            if not self._running:
                break
            self.fusion.process_gesture(event)

    # ── MAIN LOOPS ────────────────────────────────────────────────────────────

    def _run_voice_mode(self):
        threading.Thread(
            target=self._voice_feeder, name="voice-feeder", daemon=True
        ).start()

        if self.gesture_perception:
            threading.Thread(
                target=self._gesture_feeder, name="gesture-feeder", daemon=True
            ).start()
            console.print("[dim]Gesture perception: active.[/dim]")
        else:
            console.print("[dim]Gesture perception: not available (no RealSense or use --no-gesture).[/dim]")

        console.print(
            f"[dim]Wake word: [bold]'{CONFIG['system']['wake_words'][0]}'[/bold][/dim]\n"
        )

        for intent in self.fusion.fused_intents():
            if not self._running:
                break
            self.handle(intent)

    def _run_text_mode(self):
        console.print("[dim]Text mode. Type input. 'quit' to exit. 'STOP' for emergency shutdown.[/dim]\n")
        while self._running:
            try:
                text = input().strip()
                if not text:
                    continue
                if text == "STOP":
                    from core.safety import instance as safety
                    safety.activate_hard_stop()
                    console.print("[bold red]EMERGENCY STOP ACTIVATED.[/bold red]")
                    continue
                if text.lower() in ("quit", "exit", "q"):
                    break
                self.fusion.process_voice(text)
                intent = self.fusion.get(timeout=0.1)
                if intent:
                    self.handle(intent)
            except (EOFError, KeyboardInterrupt):
                break

    # ── STARTUP ───────────────────────────────────────────────────────────────

    def greet(self):
        phrase = self.brain.current_personality.activation_phrase
        print_lavender(self.brain.personality_name, phrase)
        if self.hologram:
            self.hologram.on_speaking(phrase)
        self.voice_out.speak(phrase)
        if self.hologram:
            self.hologram.on_done_speaking()

    def run(self):
        self._running = True
        print_banner()
        self._check_ollama()
        if self.toolkit:
            console.print(f"[dim]{describe_toolkit(self.toolkit)}[/dim]\n")
        self.greet()
        try:
            if self.text_mode:
                self._run_text_mode()
            else:
                self._run_voice_mode()
        finally:
            self._shutdown()

    def _on_proactive_trigger(self, message: str, priority: TriggerPriority):
        """
        Called by ProactiveEngine when a trigger fires.
        Injects the message into the fusion layer so it flows
        through the same handle() pipeline as any other input.
        HIGH priority triggers interrupt ongoing speech.
        """
        if not self._running:
            return

        if priority == TriggerPriority.HIGH and self.voice_out.is_speaking():
            self.voice_out.interrupt()
            time.sleep(0.2)

        logger.info(f"Proactive trigger: '{message[:60]}' [{priority.value}]")

        # Print so it's visible in terminal
        p = self.brain.personality_name
        c = PERSONALITY_COLORS.get(p, "white")
        i = PERSONALITY_ICONS.get(p, "●")
        console.print(f"\n  [{c}]{i}[/{c}] [dim italic]proactive →[/dim italic] [{c}]{message}[/{c}]\n")

        if self.hologram:
            self.hologram.on_speaking(message)

        self.voice_out.speak(message)

        if self.hologram:
            self.hologram.on_done_speaking()

    def _check_ollama(self):
        import httpx
        url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        try:
            r = httpx.get(f"{url}/api/tags", timeout=3.0)
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                logger.info(f"Ollama: {models}")
        except Exception:
            console.print(
                f"[yellow]⚠  Ollama not reachable. Run: ollama serve[/yellow]\n"
            )

    # ── HEALTH CALLBACKS ──────────────────────────────────────────────────────

    def _on_health_critical(self, component: str, message: str):
        logger.error(f"CRITICAL: {component} — {message}")
        if self.hologram:
            self.hologram.show_alert(
                title=f"System: {component}", message=message, severity="critical"
            )
        if component != "elevenlabs":
            self.voice_out.speak(
                f"Warning. {component} has stopped responding. "
                "Some features may be unavailable."
            )

    def _on_health_recovery(self, component: str):
        logger.info(f"Recovered: {component}")
        if self.hologram:
            self.hologram.show_alert(
                title=f"{component} recovered", message="Back online.", severity="info"
            )


    # ── SHUTDOWN ──────────────────────────────────────────────────────────────

    def _shutdown(self):
        self._running = False
        console.print("\n[dim]Shutting down...[/dim]")

        if self.voice_out.is_speaking():
            self.voice_out.interrupt()
        if self.voice_in:
            self.voice_in.stop()
        if self.gesture_perception:
            self.gesture_perception.stop()
        if self._ambient_pusher:
            self._ambient_pusher.stop()
        if self.hologram:
            self.hologram.stop()
        self.health.stop()
        self.proactive.stop()

        console.print("[dim]Writing session to memory...[/dim]")
        self.brain.close_session()

        # Apply episodic decay
        decay_days = CONFIG["memory"].get("episodic_decay_days", 90)
        self.memory.episodic.decay(days_half_life=decay_days)

        console.print(f"[dim]{self.brain.get_session_summary()}[/dim]")
        console.print(f"[dim]{self.health.format_status()}[/dim]")
        console.print("[dim]Goodbye.[/dim]\n")

    def stop(self):
        self._running = False
        if self.voice_in:
            self.voice_in.stop()
        if self.gesture_perception:
            self.gesture_perception.stop()


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Lavender AI System")
    parser.add_argument("--personality", "-p", type=str, default=None,
                        choices=["iris", "nova", "vector", "solace", "lilac"])
    parser.add_argument("--cpu",          action="store_true")
    parser.add_argument("--text",         action="store_true")
    parser.add_argument("--no-gesture",   action="store_true")
    parser.add_argument("--no-hologram",  action="store_true")
    args = parser.parse_args()

    lavender = Lavender(
        personality=args.personality,
        use_cpu=args.cpu,
        text_mode=args.text,
        no_gesture=args.no_gesture,
        no_hologram=args.no_hologram,
    )

    def handle_signal(sig, frame):
        lavender.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT,  handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    lavender.run()


if __name__ == "__main__":
    main()
