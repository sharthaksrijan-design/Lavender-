"""
LAVENDER — Proactive Intelligence
core/proactive.py

Makes Lavender speak first. Without being asked.

Four trigger categories:

  TIME — based on clock and schedule
    • Morning briefing at configured time
    • Meeting warnings (15min, 5min before)
    • End-of-day summary offer
    • Idle check-in after long silence (personality-dependent)

  CONTEXT — based on what the system knows is happening
    • System resource warnings (CPU/RAM/GPU > threshold)
    • Long session without a break (>90min)
    • Memory: "last time you worked on X was 3 days ago"

  INFORMATION — based on subscribed data feeds
    • Weather change (rain incoming during outdoor plans)
    • Calendar event starting
    • Background task completion (model download, etc.)

  INFERENCE — Lavender notices patterns and offers
    • Same question asked multiple times → offer to make a note
    • Consistent late-night sessions → "you should sleep"
    • Repeated errors in code → suggest a different approach

Each trigger has:
  - A check function (called on interval)
  - A cooldown (minimum time between same trigger firing)
  - A personality gate (e.g., Solace makes rest suggestions, Lilac doesn't)
  - A priority level (low/medium/high)

The proactive engine runs in a background thread.
When a trigger fires, it calls a callback with the message text.
That callback is the main loop's handle() function.
"""

import time
import logging
import threading
import random
from dataclasses import dataclass, field
from typing import Callable, Optional
from datetime import datetime, timedelta
from enum import Enum
from core.state import instance as state_engine, UserState

logger = logging.getLogger("lavender.proactive")


class TriggerPriority(str, Enum):
    LOW    = "low"     # Offer, easily ignored
    MEDIUM = "medium"  # Should be heard
    HIGH   = "high"    # Time-sensitive, interrupts


# Which personalities are allowed to use proactive triggers
# Higher proactivity personalities use more trigger types
PERSONALITY_PROACTIVITY = {
    "iris":   0.1,   # Only high-priority system alerts
    "nova":   0.6,   # Most triggers
    "vector": 0.4,   # Technical + time-sensitive only
    "solace": 0.3,   # Gentle ambient suggestions
    "lilac":  0.2,   # Rarely — and only if it's worth her time
}


@dataclass
class ProactiveTrigger:
    name: str
    check_fn: Callable[[], Optional[str]]  # Returns message text or None
    interval_seconds: float                # How often to check
    cooldown_seconds: float                # Min time between firings
    priority: TriggerPriority
    min_proactivity: float = 0.0           # Minimum personality proactivity level
    last_check: float = 0.0
    last_fired: float = 0.0
    enabled: bool = True

    def is_ready_to_check(self, now: float) -> bool:
        return (now - self.last_check) >= self.interval_seconds

    def is_off_cooldown(self, now: float) -> bool:
        return (now - self.last_fired) >= self.cooldown_seconds


class ProactiveEngine:
    """
    Runs trigger checks in a background thread.
    When a trigger fires, calls the provided callback with the message.
    """

    def __init__(
        self,
        on_trigger: Callable[[str, TriggerPriority], None],
        get_personality: Callable[[], str],
        memory=None,
        session_start: float = None,
    ):
        """
        on_trigger:      Called with (message_text, priority) when trigger fires
        get_personality: Returns current personality name (checked at fire time)
        memory:          LavenderMemory instance for context-aware triggers
        """
        self._on_trigger    = on_trigger
        self._get_personality = get_personality
        self._memory        = memory
        self._session_start = session_start or time.time()

        self._triggers: list[ProactiveTrigger] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Session tracking
        self._last_interaction = time.time()
        self._interaction_count = 0
        self._cpu_warned = False
        self._break_suggested = False

        # Build standard trigger set
        self._register_standard_triggers()

    # ── TRIGGER REGISTRATION ──────────────────────────────────────────────────

    def register(self, trigger: ProactiveTrigger):
        self._triggers.append(trigger)
        logger.info(f"Proactive trigger registered: '{trigger.name}'")

    def _register_standard_triggers(self):
        """Register the built-in trigger set."""

        # ── MORNING BRIEFING ──────────────────────────────────────────────────
        self.register(ProactiveTrigger(
            name="morning_briefing",
            check_fn=self._check_morning,
            interval_seconds=60.0,
            cooldown_seconds=86400.0,   # Once per day
            priority=TriggerPriority.MEDIUM,
            min_proactivity=0.3,
        ))

        # ── IDLE CHECK-IN ─────────────────────────────────────────────────────
        self.register(ProactiveTrigger(
            name="idle_checkin",
            check_fn=self._check_idle,
            interval_seconds=120.0,
            cooldown_seconds=1800.0,    # 30-minute cooldown
            priority=TriggerPriority.LOW,
            min_proactivity=0.5,        # Only for nova and above
        ))

        # ── LONG SESSION BREAK ────────────────────────────────────────────────
        self.register(ProactiveTrigger(
            name="break_reminder",
            check_fn=self._check_break_needed,
            interval_seconds=300.0,
            cooldown_seconds=3600.0,    # Once per hour
            priority=TriggerPriority.LOW,
            min_proactivity=0.3,
        ))

        # ── LATE NIGHT ───────────────────────────────────────────────────────
        self.register(ProactiveTrigger(
            name="late_night",
            check_fn=self._check_late_night,
            interval_seconds=1800.0,
            cooldown_seconds=10800.0,   # 3 hours
            priority=TriggerPriority.LOW,
            min_proactivity=0.2,
        ))

        # ── CPU/RAM ALERT ─────────────────────────────────────────────────────
        self.register(ProactiveTrigger(
            name="resource_alert",
            check_fn=self._check_resources,
            interval_seconds=30.0,
            cooldown_seconds=300.0,
            priority=TriggerPriority.HIGH,
            min_proactivity=0.0,        # All personalities
        ))

    # ── LIFECYCLE ─────────────────────────────────────────────────────────────

    def start(self):
        self._thread = threading.Thread(
            target=self._run,
            name="proactive-engine",
            daemon=True,
        )
        self._thread.start()
        logger.info("Proactive engine started.")

    def stop(self):
        self._stop.set()

    def note_interaction(self):
        """Call this every time the user interacts. Resets idle timer."""
        self._last_interaction = time.time()
        self._interaction_count += 1

    # ── MAIN LOOP ────────────────────────────────────────────────────────────

    def _run(self):
        while not self._stop.wait(10.0):
            # Context Awareness: Don't speak if user is in FOCUS mode
            if state_engine.state.user == UserState.FOCUS:
                logger.debug("Proactive trigger skipped (User in Focus mode)")
                continue

            now = time.time()
            personality = self._get_personality()
            proactivity = PERSONALITY_PROACTIVITY.get(personality, 0.3)

            for trigger in self._triggers:
                if not trigger.enabled:
                    continue

                if proactivity < trigger.min_proactivity:
                    continue

                if not trigger.is_ready_to_check(now):
                    continue

                trigger.last_check = now

                if not trigger.is_off_cooldown(now):
                    continue

                try:
                    message = trigger.check_fn()
                except Exception as e:
                    logger.warning(f"Trigger '{trigger.name}' check failed: {e}")
                    continue

                if message:
                    logger.info(f"Proactive trigger fired: '{trigger.name}'")
                    trigger.last_fired = now

                    try:
                        self._on_trigger(message, trigger.priority)
                    except Exception as e:
                        logger.error(f"on_trigger callback error: {e}")

    # ── CHECK FUNCTIONS ───────────────────────────────────────────────────────

    def _check_morning(self) -> Optional[str]:
        """Fire at 8:00–8:30am if this is the first session of the day."""
        now = datetime.now()
        if now.hour == 8 and now.minute < 30:
            return self._build_morning_message()
        return None

    def _build_morning_message(self) -> str:
        """Build a morning briefing from memory context."""
        personality = self._get_personality()

        # Try to pull relevant context from memory
        context_hint = ""
        if self._memory:
            facts = self._memory.semantic.get_category("project")
            if facts:
                first_project = next(iter(facts.values()), "")
                context_hint = f" You were working on {first_project}."

        messages = {
            "iris":   f"Morning.{context_hint}",
            "nova":   f"Good morning.{context_hint} Ready when you are.",
            "vector": f"Morning. Systems nominal.{context_hint}",
            "solace": f"Good morning. Take your time settling in.{context_hint}",
            "lilac":  f"You're up.{context_hint} Let's hope today is more productive.",
        }
        return messages.get(personality, f"Good morning.{context_hint}")

    def _check_idle(self) -> Optional[str]:
        """Check in after extended silence."""
        idle_seconds = time.time() - self._last_interaction
        personality = self._get_personality()

        # Iris and Lilac don't check in — not their style
        if personality in ("iris", "lilac"):
            return None

        thresholds = {
            "nova":   25 * 60,   # 25 minutes
            "vector": 40 * 60,   # 40 minutes
            "solace": 30 * 60,   # 30 minutes
        }
        threshold = thresholds.get(personality, 30 * 60)

        if idle_seconds < threshold:
            return None

        messages = {
            "nova":   random.choice([
                "Still here if you need anything.",
                "Quiet in here. Working through something?",
            ]),
            "vector": "Still present. Flag me if you need input.",
            "solace": "Just checking in. Everything okay?",
        }
        return messages.get(personality)

    def _check_break_needed(self) -> Optional[str]:
        """Suggest a break after 90+ minutes of continuous work."""
        if self._break_suggested:
            return None

        session_minutes = (time.time() - self._session_start) / 60
        idle_minutes = (time.time() - self._last_interaction) / 60

        if session_minutes < 90 or idle_minutes > 10:
            return None

        personality = self._get_personality()
        self._break_suggested = True

        messages = {
            "iris":   "90 minutes. Consider a break.",
            "nova":   "You've been at this for over an hour and a half. "
                      "Might be worth stepping away for a few minutes.",
            "vector": "Session duration: 90+ minutes. "
                      "Cognitive performance degrades without breaks.",
            "solace": "You've been working for a while. "
                      "There's no urgency. A short break might help.",
            "lilac":  "You've been here for 90 minutes. "
                      "Whether that was productive is a separate question.",
        }
        return messages.get(personality)

    def _check_late_night(self) -> Optional[str]:
        """Gentle late-night nudge."""
        now = datetime.now()
        if now.hour < 23 and not (now.hour == 0):
            return None

        personality = self._get_personality()
        if personality in ("iris",):
            return None

        messages = {
            "nova":   "It's past midnight. Not judging, just noting.",
            "vector": "Past midnight. Sleep deprivation measurably impairs "
                      "the kind of work you're probably doing.",
            "solace": "It's late. The work will still be here tomorrow.",
            "lilac":  "It's past midnight. I don't know what you're expecting "
                      "to accomplish at this point, but here we are.",
        }
        return messages.get(personality)

    def _check_resources(self) -> Optional[str]:
        """Alert on high CPU or RAM usage."""
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=1)
            ram = psutil.virtual_memory().percent

            # GPU check
            gpu = 0.0
            try:
                import subprocess
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=1
                )
                gpu = float(result.stdout.strip())
            except Exception:
                pass

            if ram > 92:
                return f"RAM is at {ram:.0f}%. Things may start slowing down."
            if cpu > 95 and not self._cpu_warned:
                self._cpu_warned = True
                return f"CPU is fully saturated at {cpu:.0f}%. Something is running hard."
            if cpu < 80:
                self._cpu_warned = False  # Reset warning when it calms down

        except ImportError:
            pass
        return None

    # ── CUSTOM TRIGGER BUILDER ────────────────────────────────────────────────

    def add_calendar_trigger(
        self,
        event_name: str,
        event_time: datetime,
        warning_minutes: list[int] = None,
    ):
        """
        Add a one-time trigger for an upcoming calendar event.
        Fires at warning_minutes before the event (default: [15, 5]).
        """
        warning_minutes = warning_minutes or [15, 5]

        for mins in warning_minutes:
            fire_time = event_time - timedelta(minutes=mins)

            def make_check(ft=fire_time, en=event_name, m=mins):
                fired = [False]
                def check() -> Optional[str]:
                    if fired[0]:
                        return None
                    now = datetime.now()
                    if now >= ft and now < ft + timedelta(minutes=2):
                        fired[0] = True
                        return f"{en} starts in {m} minutes."
                    return None
                return check

            self.register(ProactiveTrigger(
                name=f"calendar_{event_name}_{mins}min",
                check_fn=make_check(),
                interval_seconds=30.0,
                cooldown_seconds=86400.0,
                priority=TriggerPriority.HIGH if mins <= 5 else TriggerPriority.MEDIUM,
                min_proactivity=0.0,
            ))
