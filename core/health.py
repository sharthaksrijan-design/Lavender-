"""
LAVENDER — Health Monitor
core/health.py

Watches every subsystem and recovers from failures automatically.
Lavender should never need a manual restart for transient errors.

Monitored components:
  - Ollama (LLM inference server)
  - ElevenLabs API reachability
  - RealSense pipeline
  - WebSocket renderer connection
  - Voice input stream
  - Memory database

Each component has:
  - A health check function (called every N seconds)
  - A recovery function (called on failure)
  - A failure count before alerting the user
  - A maximum failure count before marking component as dead

The health loop runs in a daemon thread.
Critical failures are announced via voice (if voice is up).
Non-critical failures are logged and retried silently.
"""

import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional
from enum import Enum

logger = logging.getLogger("lavender.health")


class ComponentStatus(str, Enum):
    HEALTHY    = "healthy"
    DEGRADED   = "degraded"    # intermittent failures, recovering
    FAILED     = "failed"      # persistent failure, action required
    RECOVERING = "recovering"  # recovery attempt in progress
    DISABLED   = "disabled"    # not configured, not checked


@dataclass
class ComponentHealth:
    name: str
    check_fn: Callable[[], bool]
    recover_fn: Optional[Callable[[], bool]] = None
    check_interval: float = 30.0          # seconds between checks
    failure_threshold: int = 3            # failures before DEGRADED
    dead_threshold: int = 10              # failures before FAILED
    critical: bool = False                # if True, alert user on failure
    last_check: float = 0.0
    last_success: float = 0.0
    failure_count: int = 0
    status: ComponentStatus = ComponentStatus.HEALTHY
    last_error: str = ""


class HealthMonitor:
    """
    Runs component health checks in a background thread.
    Exposes status for the hologram ambient display and voice alerts.
    """

    def __init__(
        self,
        check_interval: float = 15.0,
        on_critical_failure: Optional[Callable[[str, str], None]] = None,
        on_recovery: Optional[Callable[[str], None]] = None,
    ):
        self._components: dict[str, ComponentHealth] = {}
        self._default_interval = check_interval
        self._on_critical_failure = on_critical_failure  # (name, message) → None
        self._on_recovery = on_recovery                   # (name) → None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    # ── REGISTRATION ──────────────────────────────────────────────────────────

    def register(
        self,
        name: str,
        check_fn: Callable[[], bool],
        recover_fn: Optional[Callable[[], bool]] = None,
        check_interval: float = None,
        failure_threshold: int = 3,
        dead_threshold: int = 10,
        critical: bool = False,
    ):
        """
        Register a component for health monitoring.

        check_fn:   Returns True if component is healthy, False otherwise.
        recover_fn: Called on failure. Returns True if recovery succeeded.
        critical:   If True, user is alerted on persistent failure.
        """
        component = ComponentHealth(
            name=name,
            check_fn=check_fn,
            recover_fn=recover_fn,
            check_interval=check_interval or self._default_interval,
            failure_threshold=failure_threshold,
            dead_threshold=dead_threshold,
            critical=critical,
        )
        with self._lock:
            self._components[name] = component
        logger.info(f"Health monitor: registered '{name}' (interval: {component.check_interval}s)")

    def disable(self, name: str):
        with self._lock:
            if name in self._components:
                self._components[name].status = ComponentStatus.DISABLED

    # ── LIFECYCLE ─────────────────────────────────────────────────────────────

    def start(self):
        self._thread = threading.Thread(
            target=self._run,
            name="health-monitor",
            daemon=True,
        )
        self._thread.start()
        logger.info("Health monitor started.")

    def stop(self):
        self._stop.set()

    # ── CHECK LOOP ─────────────────────────────────────────────────────────────

    def _run(self):
        while not self._stop.wait(5.0):   # Check schedule every 5s
            now = time.time()
            with self._lock:
                components = list(self._components.values())

            for component in components:
                if component.status == ComponentStatus.DISABLED:
                    continue

                if now - component.last_check < component.check_interval:
                    continue

                self._check_component(component, now)

    def _check_component(self, c: ComponentHealth, now: float):
        c.last_check = now

        try:
            healthy = c.check_fn()
        except Exception as e:
            healthy = False
            c.last_error = str(e)

        if healthy:
            was_degraded = c.status in (ComponentStatus.DEGRADED, ComponentStatus.RECOVERING)
            c.failure_count = 0
            c.last_success  = now
            c.status        = ComponentStatus.HEALTHY
            c.last_error    = ""

            if was_degraded:
                logger.info(f"[health] '{c.name}' recovered.")
                if self._on_recovery:
                    try:
                        self._on_recovery(c.name)
                    except Exception:
                        pass
        else:
            c.failure_count += 1
            logger.warning(
                f"[health] '{c.name}' check failed "
                f"(count: {c.failure_count}/{c.dead_threshold}) "
                f"{c.last_error}"
            )

            if c.failure_count >= c.dead_threshold:
                prev = c.status
                c.status = ComponentStatus.FAILED
                if prev != ComponentStatus.FAILED and c.critical:
                    msg = f"{c.name} has stopped responding and could not recover."
                    logger.error(f"[health] CRITICAL FAILURE: {c.name}")
                    if self._on_critical_failure:
                        try:
                            self._on_critical_failure(c.name, msg)
                        except Exception:
                            pass

            elif c.failure_count >= c.failure_threshold:
                c.status = ComponentStatus.DEGRADED

                # Attempt recovery
                if c.recover_fn:
                    logger.info(f"[health] Attempting recovery: '{c.name}'")
                    c.status = ComponentStatus.RECOVERING
                    try:
                        recovered = c.recover_fn()
                        if recovered:
                            c.failure_count = 0
                            c.status = ComponentStatus.HEALTHY
                            logger.info(f"[health] Recovery succeeded: '{c.name}'")
                        else:
                            c.status = ComponentStatus.DEGRADED
                            logger.warning(f"[health] Recovery failed: '{c.name}'")
                    except Exception as e:
                        c.status = ComponentStatus.DEGRADED
                        logger.error(f"[health] Recovery exception for '{c.name}': {e}")

    # ── STATUS QUERIES ────────────────────────────────────────────────────────

    def get_status(self, name: str) -> ComponentStatus:
        with self._lock:
            comp = self._components.get(name)
            return comp.status if comp else ComponentStatus.DISABLED

    def all_healthy(self) -> bool:
        with self._lock:
            return all(
                c.status in (ComponentStatus.HEALTHY, ComponentStatus.DISABLED)
                for c in self._components.values()
            )

    def get_summary(self) -> dict[str, str]:
        with self._lock:
            return {name: c.status.value for name, c in self._components.items()}

    def format_status(self) -> str:
        """Human-readable health status for display."""
        summary = self.get_summary()
        lines = []
        for name, status in summary.items():
            icon = {
                "healthy":    "✓",
                "degraded":   "⚠",
                "failed":     "✗",
                "recovering": "↻",
                "disabled":   "—",
            }.get(status, "?")
            lines.append(f"  {icon} {name}: {status}")
        return "\n".join(lines)


# ── PRE-BUILT CHECK FUNCTIONS ─────────────────────────────────────────────────

def make_ollama_check(base_url: str = "http://localhost:11434") -> Callable:
    def check() -> bool:
        import httpx
        try:
            r = httpx.get(f"{base_url}/api/tags", timeout=3.0)
            return r.status_code == 200
        except Exception:
            return False
    return check


def make_ollama_recover(base_url: str = "http://localhost:11434") -> Callable:
    def recover() -> bool:
        """Try to restart the Ollama service via systemd, then re-check."""
        import subprocess, time
        try:
            subprocess.run(["systemctl", "restart", "ollama"],
                           capture_output=True, timeout=15)
            time.sleep(5)
            import httpx
            r = httpx.get(f"{base_url}/api/tags", timeout=5.0)
            return r.status_code == 200
        except Exception:
            return False
    return recover


def make_elevenlabs_check(api_key: str) -> Callable:
    def check() -> bool:
        if not api_key or api_key == "your_api_key_here":
            return True  # Not configured → not checked
        import httpx
        try:
            r = httpx.get(
                "https://api.elevenlabs.io/v1/user",
                headers={"xi-api-key": api_key},
                timeout=5.0,
            )
            return r.status_code == 200
        except Exception:
            return False
    return check


def make_memory_check(episodic_db_path: str) -> Callable:
    def check() -> bool:
        import os
        return os.path.isdir(episodic_db_path)
    return check


def make_realsense_check() -> Callable:
    def check() -> bool:
        try:
            import pyrealsense2 as rs
            ctx = rs.context()
            return len(ctx.devices) > 0
        except Exception:
            return False
    return check


def make_websocket_check(port: int = 8765) -> Callable:
    def check() -> bool:
        import socket
        try:
            with socket.create_connection(("localhost", port), timeout=1.0):
                return True
        except Exception:
            return False
    return check


def build_standard_monitors(
    ollama_url: str,
    elevenlabs_key: str,
    memory_path: str,
    ws_port: int,
    on_critical: Optional[Callable] = None,
    on_recovery: Optional[Callable] = None,
) -> HealthMonitor:
    """
    Build and return a pre-configured HealthMonitor for the standard Lavender stack.
    Call monitor.start() after building.
    """
    monitor = HealthMonitor(
        on_critical_failure=on_critical,
        on_recovery=on_recovery,
    )

    monitor.register(
        name="ollama",
        check_fn=make_ollama_check(ollama_url),
        recover_fn=make_ollama_recover(ollama_url),
        check_interval=20.0,
        failure_threshold=2,
        dead_threshold=6,
        critical=True,
    )

    monitor.register(
        name="elevenlabs",
        check_fn=make_elevenlabs_check(elevenlabs_key),
        check_interval=60.0,
        failure_threshold=3,
        dead_threshold=9,
        critical=False,   # Falls back to pyttsx3, so not critical
    )

    monitor.register(
        name="memory_db",
        check_fn=make_memory_check(memory_path),
        check_interval=30.0,
        failure_threshold=1,
        dead_threshold=3,
        critical=True,
    )

    monitor.register(
        name="renderer",
        check_fn=make_websocket_check(ws_port),
        check_interval=15.0,
        failure_threshold=3,
        dead_threshold=12,
        critical=False,   # Lavender works fine without display
    )

    return monitor
