"""
LAVENDER — Hologram Director
core/hologram.py

A WebSocket server that the Unity renderer connects to.
The brain calls methods on this class; they become JSON directives
sent to Unity, which renders them on the holographic display.

Every system state change, every response, every personality switch
goes through here. The renderer is a thin client — all logic lives here.

WebSocket runs in a background asyncio thread so it never blocks the
main voice loop.
"""

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Any

import websockets
from websockets.server import WebSocketServerProtocol

logger = logging.getLogger("lavender.hologram")


# ─────────────────────────────────────────────────────────────────────────────
# DIRECTIVE TYPES
# Everything Lavender can tell the renderer to do.
# ─────────────────────────────────────────────────────────────────────────────

class DirectiveType:
    # System state
    SET_STATE        = "set_state"        # ambient / active / focus / spatial
    SET_THEME        = "set_theme"        # personality theme switch
    SET_BRIGHTNESS   = "set_brightness"   # 0.0 – 1.0

    # Panels
    SHOW_PANEL       = "show_panel"       # create or update a named panel
    UPDATE_PANEL     = "update_panel"     # update content without recreating
    CLOSE_PANEL      = "close_panel"      # dismiss a panel
    CLEAR_ALL        = "clear_all"        # remove everything

    # Waveform
    SET_WAVEFORM     = "set_waveform"     # idle / listening / thinking / speaking
    PUSH_AUDIO       = "push_audio"       # raw amplitude for live waveform

    # Ambient
    UPDATE_CLOCK     = "update_clock"     # time + date string
    UPDATE_AMBIENT   = "update_ambient"   # dashboard data (cpu, ram, etc.)
    SHOW_ALERT       = "show_alert"       # urgent override notification

    # Transitions
    PERSONALITY_TRANSITION = "personality_transition"  # full themed transition


class SystemState(str, Enum):
    AMBIENT    = "ambient"    # present but idle
    ACTIVE     = "active"     # in conversation
    FOCUS      = "focus"      # deep work, minimal display
    SPATIAL    = "spatial"    # fog mode active
    THINKING   = "thinking"   # processing, waveform active


class WaveformState(str, Enum):
    IDLE       = "idle"       # slow sine, barely visible
    LISTENING  = "listening"  # reacting to mic input
    THINKING   = "thinking"   # slow irregular pulse
    SPEAKING   = "speaking"   # live speech waveform


# Panel layer constants — match UIConfig.cs
class Layer:
    BACKGROUND  = 0   # clock, slow data
    PERSISTENT  = 1   # calendar, system status
    ACTIVE      = 2   # current task, response text
    FOREGROUND  = 3   # alerts, confirmations, direct speech


@dataclass
class Directive:
    type: str
    payload: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({"type": self.type, "payload": self.payload})


# ─────────────────────────────────────────────────────────────────────────────
# HOLOGRAM DIRECTOR
# ─────────────────────────────────────────────────────────────────────────────

class HologramDirector:
    """
    Manages the WebSocket server and all directive emission.

    The brain and main loop call methods like:
        director.show_response("Hello, how can I help?")
        director.set_state(SystemState.THINKING)
        director.set_personality("lilac")

    These get serialized and sent to all connected Unity clients.
    """

    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port

        self._clients: set[WebSocketServerProtocol] = set()
        self._clients_lock = asyncio.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._server = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # Current display state — sent to new clients on connect
        self._current_theme    = "nova"
        self._current_state    = SystemState.AMBIENT
        self._current_brightness = 1.0

        # Panel registry — track what's showing
        self._active_panels: dict[str, dict] = {}

        logger.info(f"HologramDirector created. Will bind to ws://{host}:{port}")

    # ── SERVER LIFECYCLE ──────────────────────────────────────────────────────

    def start(self):
        """
        Start the WebSocket server in a background daemon thread.
        Returns immediately. Non-blocking.
        """
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._run_server_thread,
            name="hologram-ws",
            daemon=True,
        )
        self._thread.start()
        logger.info("Hologram WebSocket server starting...")

        # Give the loop a moment to start
        time.sleep(0.3)

    def stop(self):
        """Cleanly stop the server."""
        self._running = False
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        logger.info("Hologram director stopped.")

    def _run_server_thread(self):
        """Entry point for the background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except Exception as e:
            logger.error(f"WebSocket server error: {e}")
        finally:
            self._loop.close()

    async def _serve(self):
        """Async server coroutine."""
        async with websockets.serve(
            self._handle_client,
            self.host,
            self.port,
            ping_interval=20,
            ping_timeout=10,
        ) as server:
            self._server = server
            logger.info(f"Hologram WebSocket listening on ws://{self.host}:{self.port}")
            await asyncio.Future()  # Run forever

    async def _handle_client(self, websocket: WebSocketServerProtocol):
        """Handle a new Unity client connection."""
        async with self._clients_lock:
            self._clients.add(websocket)

        client_id = f"{websocket.remote_address}"
        logger.info(f"Renderer connected: {client_id}")

        # Send current state to new client immediately
        await self._send_to(websocket, Directive(
            type=DirectiveType.SET_THEME,
            payload={"theme": self._current_theme}
        ))
        await self._send_to(websocket, Directive(
            type=DirectiveType.SET_STATE,
            payload={"state": self._current_state.value}
        ))
        await self._send_to(websocket, Directive(
            type=DirectiveType.SET_BRIGHTNESS,
            payload={"value": self._current_brightness}
        ))
        # Restore active panels
        for panel_id, panel_data in self._active_panels.items():
            await self._send_to(websocket, Directive(
                type=DirectiveType.SHOW_PANEL,
                payload=panel_data
            ))

        try:
            # Keep connection alive — Unity may send back telemetry in future
            async for message in websocket:
                try:
                    data = json.loads(message)
                    logger.debug(f"Renderer message: {data}")
                except json.JSONDecodeError:
                    pass
        except websockets.ConnectionClosed:
            pass
        finally:
            async with self._clients_lock:
                self._clients.discard(websocket)
            logger.info(f"Renderer disconnected: {client_id}")

    # ── DIRECTIVE EMISSION ────────────────────────────────────────────────────

    def _emit(self, directive: Directive):
        """
        Thread-safe directive emission.
        Called from main thread; dispatches to the async loop.
        """
        if not self._loop or not self._running:
            return

        asyncio.run_coroutine_threadsafe(
            self._broadcast(directive),
            self._loop
        )

    async def _broadcast(self, directive: Directive):
        """Send a directive to all connected clients."""
        if not self._clients:
            return

        message = directive.to_json()
        async with self._clients_lock:
            clients = set(self._clients)

        dead = set()
        for client in clients:
            try:
                await client.send(message)
            except websockets.ConnectionClosed:
                dead.add(client)
            except Exception as e:
                logger.warning(f"Failed to send to client: {e}")
                dead.add(client)

        if dead:
            async with self._clients_lock:
                self._clients -= dead

    async def _send_to(self, websocket: WebSocketServerProtocol, directive: Directive):
        """Send a directive to a specific client."""
        try:
            await websocket.send(directive.to_json())
        except Exception as e:
            logger.warning(f"Failed to send directive: {e}")

    @property
    def has_clients(self) -> bool:
        return len(self._clients) > 0

    # ── PUBLIC API — called by brain and main loop ────────────────────────────

    def set_state(self, state: SystemState):
        """Update Lavender's interaction state — affects waveform and brightness."""
        self._current_state = state
        self._emit(Directive(
            type=DirectiveType.SET_STATE,
            payload={"state": state.value}
        ))
        logger.debug(f"State → {state.value}")

    def set_personality(self, personality: str):
        """
        Trigger a full personality theme transition.
        Unity will animate the collapse and expansion of the new theme.
        """
        self._current_theme = personality
        self._emit(Directive(
            type=DirectiveType.PERSONALITY_TRANSITION,
            payload={
                "theme": personality,
                "duration_ms": 1800,  # transition animation duration
            }
        ))
        logger.info(f"Personality theme → {personality}")

    def set_brightness(self, value: float):
        """Adjust hologram brightness. 0.0 = off, 1.0 = full."""
        self._current_brightness = max(0.0, min(1.0, value))
        self._emit(Directive(
            type=DirectiveType.SET_BRIGHTNESS,
            payload={"value": self._current_brightness}
        ))

    def set_waveform(self, state: WaveformState, amplitude: float = 0.0):
        """Update waveform visual state."""
        self._emit(Directive(
            type=DirectiveType.SET_WAVEFORM,
            payload={"state": state.value, "amplitude": amplitude}
        ))

    def show_response(self, text: str, auto_dismiss_seconds: float = 12.0):
        """
        Display Lavender's spoken response text on the hologram.
        Text appears synchronized with speech — phrase by phrase.
        Auto-dismisses after auto_dismiss_seconds.
        """
        panel_id = "response"
        payload = {
            "panel_id":     panel_id,
            "layer":        Layer.FOREGROUND,
            "content_type": "response_text",
            "text":         text,
            "animate_in":   "fade",
            "auto_dismiss": auto_dismiss_seconds,
            "personality":  self._current_theme,
        }
        self._active_panels[panel_id] = payload
        self._emit(Directive(type=DirectiveType.SHOW_PANEL, payload=payload))

    def show_panel(
        self,
        panel_id: str,
        content: Any,
        layer: int = Layer.ACTIVE,
        content_type: str = "text",
        title: str = "",
        animate_in: str = "slide_right",
        persist: bool = True,
        auto_dismiss: float = 0.0,
    ):
        """
        Show a named panel on the hologram.

        panel_id:     Unique ID — showing same ID again updates it in place.
        content:      String or dict depending on content_type.
        layer:        Z-depth layer (0=background, 3=foreground).
        content_type: "text" | "data_table" | "key_value" | "calendar" | "system_stats"
        title:        Optional panel title.
        animate_in:   "slide_left" | "slide_right" | "fade" | "materialize"
        persist:      If True, panel survives clear_transient(). False = session panel.
        auto_dismiss: Seconds until auto-close. 0 = never.
        """
        payload = {
            "panel_id":     panel_id,
            "layer":        layer,
            "content_type": content_type,
            "content":      content,
            "title":        title,
            "animate_in":   animate_in,
            "persist":      persist,
            "auto_dismiss": auto_dismiss,
            "personality":  self._current_theme,
        }

        if persist:
            self._active_panels[panel_id] = payload
        self._emit(Directive(type=DirectiveType.SHOW_PANEL, payload=payload))

    def update_panel(self, panel_id: str, content: Any, title: str = ""):
        """Update an existing panel's content without recreating it."""
        payload = {"panel_id": panel_id, "content": content}
        if title:
            payload["title"] = title

        if panel_id in self._active_panels:
            self._active_panels[panel_id].update(payload)

        self._emit(Directive(type=DirectiveType.UPDATE_PANEL, payload=payload))

    def close_panel(self, panel_id: str):
        """Dismiss a panel with its exit animation."""
        self._active_panels.pop(panel_id, None)
        self._emit(Directive(
            type=DirectiveType.CLOSE_PANEL,
            payload={"panel_id": panel_id}
        ))

    def clear_transient(self):
        """
        Clear all non-persistent panels (response text, temporary info).
        Persistent panels (calendar, system status) remain.
        """
        transient_ids = [
            pid for pid, pdata in self._active_panels.items()
            if not pdata.get("persist", True)
        ]
        for pid in transient_ids:
            self.close_panel(pid)

    def clear_all(self):
        """Remove everything from the display."""
        self._active_panels.clear()
        self._emit(Directive(type=DirectiveType.CLEAR_ALL, payload={}))

    def show_alert(self, title: str, message: str, severity: str = "info"):
        """
        Show an urgent alert that overrides personality theme.
        severity: "info" | "warning" | "critical"
        """
        self._emit(Directive(
            type=DirectiveType.SHOW_ALERT,
            payload={
                "title":    title,
                "message":  message,
                "severity": severity,
            }
        ))

    def update_ambient(self, data: dict):
        """
        Push updated ambient dashboard data.
        data keys: cpu_pct, ram_pct, gpu_pct, temperature, network_mbps, etc.
        """
        self._emit(Directive(
            type=DirectiveType.UPDATE_AMBIENT,
            payload=data
        ))

    def update_clock(self, time_str: str, date_str: str):
        """Push updated clock strings."""
        self._emit(Directive(
            type=DirectiveType.UPDATE_CLOCK,
            payload={"time": time_str, "date": date_str}
        ))

    # ── COMPOUND HELPERS ──────────────────────────────────────────────────────

    def on_wake(self):
        """Called when wake word detected — shift from ambient to attention state."""
        self.set_state(SystemState.ACTIVE)
        self.set_waveform(WaveformState.LISTENING)

    def on_thinking(self):
        """Called when brain starts processing."""
        self.set_waveform(WaveformState.THINKING)

    def on_speaking(self, response_text: str):
        """Called when Lavender starts speaking — show text and waveform."""
        self.set_state(SystemState.ACTIVE)
        self.set_waveform(WaveformState.SPEAKING)
        self.show_response(response_text)

    def on_done_speaking(self):
        """Called when TTS playback finishes."""
        self.set_waveform(WaveformState.IDLE)

    def on_focus_mode(self):
        """Shift to focus state — minimal display."""
        self.set_state(SystemState.FOCUS)
        self.clear_transient()
        self.set_brightness(0.4)

    def on_ambient(self):
        """Return to ambient idle state."""
        self.set_state(SystemState.AMBIENT)
        self.set_waveform(WaveformState.IDLE)
        self.set_brightness(1.0)

    def on_fog_mode_start(self):
        """Fog/spatial mode activating."""
        self.set_state(SystemState.SPATIAL)
        self._emit(Directive(
            type=DirectiveType.SET_STATE,
            payload={"state": "spatial", "minimize_panels": True}
        ))

    def on_fog_mode_end(self):
        """Fog mode deactivating — restore panels."""
        self.set_state(SystemState.ACTIVE)


# ─────────────────────────────────────────────────────────────────────────────
# AMBIENT DATA PUSHER
# Runs in background, pushes clock and system stats every few seconds
# ─────────────────────────────────────────────────────────────────────────────

class AmbientDataPusher:
    """
    Background thread that keeps the ambient display updated.
    Clock, system stats, etc.
    """

    def __init__(self, director: HologramDirector, interval_seconds: float = 5.0):
        self.director = director
        self.interval = interval_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="ambient-pusher",
            daemon=True,
        )

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        import datetime
        while not self._stop.wait(self.interval):
            try:
                now = datetime.datetime.now()
                self.director.update_clock(
                    time_str=now.strftime("%H:%M"),
                    date_str=now.strftime("%A, %d %B"),
                )

                # System stats
                try:
                    import psutil
                    self.director.update_ambient({
                        "cpu_pct": psutil.cpu_percent(interval=None),
                        "ram_pct": psutil.virtual_memory().percent,
                        "gpu_pct": self._get_gpu_usage(),
                    })
                except ImportError:
                    pass  # psutil not installed — skip stats

            except Exception as e:
                logger.warning(f"Ambient pusher error: {e}")

    def _get_gpu_usage(self) -> float:
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=1
            )
            return float(result.stdout.strip())
        except Exception:
            return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test — run without Unity to verify the server starts
# python core/hologram.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    director = HologramDirector(port=8765)
    director.start()

    pusher = AmbientDataPusher(director, interval_seconds=2.0)
    pusher.start()

    print("\nHologram Director running.")
    print("Open Unity and connect to ws://localhost:8765")
    print("Ctrl+C to stop.\n")

    personalities = ["iris", "nova", "vector", "solace", "lilac"]
    idx = 0

    try:
        while True:
            time.sleep(5)

            # Cycle through personalities for testing
            p = personalities[idx % len(personalities)]
            print(f"→ Setting personality: {p}")
            director.set_personality(p)
            director.show_response(
                f"This is a test response from {p.upper()}. "
                "The display should show the correct theme."
            )
            idx += 1

    except KeyboardInterrupt:
        pusher.stop()
        director.stop()
        print("Stopped.")
