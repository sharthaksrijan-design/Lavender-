"""
LAVENDER — Intent Fusion
core/intent_fusion.py

Merges signals from all input modalities into a single, resolved intent
before passing to the brain. This is the arbiter.

Modalities:
  VOICE   — primary semantic carrier (always wins on meaning)
  GESTURE — wins on spatial/panel operations
  GAZE    — wins on reference resolution ("that one", "this document")
  SURFACE — wins on system controls (volume, emergency mute)

A FusedIntent contains:
  - The resolved text to send to the brain
  - The originating modality (for logging and hologram feedback)
  - Spatial context (what zone, what object was looked at)
  - Whether any modifier gestures are active

Conflict resolution:
  1. Surface wins on system controls always
  2. Voice wins on semantic intent when unambiguous
  3. Gesture wins on spatial operations (in panel or fog zone)
  4. Gaze enriches voice when voice uses deictic references
     ("that", "this", "those") → substituted with gazed-at entity

The fusion layer also handles temporal binding:
  A gesture that occurs within 2 seconds of a voice command is
  considered part of the same intent and can modify it.
"""

import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import threading
import queue

logger = logging.getLogger("lavender.fusion")


class Modality(str, Enum):
    VOICE   = "voice"
    GESTURE = "gesture"
    GAZE    = "gaze"
    SURFACE = "surface"
    FUSED   = "fused"


class SurfaceControl(str, Enum):
    VOLUME_UP      = "volume_up"
    VOLUME_DOWN    = "volume_down"
    PERSONALITY_NEXT = "personality_next"
    BRIGHTNESS_UP  = "brightness_up"
    BRIGHTNESS_DOWN = "brightness_down"
    PAUSE_RESUME   = "pause_resume"
    EMERGENCY_MUTE = "emergency_mute"


@dataclass
class GazeState:
    """Current gaze fixation point and what it's looking at."""
    target_type: str = ""      # "panel", "fog_object", "surface", "away"
    target_id:   str = ""      # ID of the specific panel or object
    fixation_duration_ms: float = 0.0
    position_3d: Optional[object] = None  # numpy array


@dataclass
class GestureState:
    """Most recently confirmed gesture and its context."""
    gesture_type: str = ""
    zone: str = ""
    hand: str = ""
    timestamp: float = 0.0
    is_active: bool = False    # Still being held


@dataclass
class FusedIntent:
    """
    The resolved, enriched intent ready for the brain.
    """
    # What to send to the brain
    text: str

    # Where it came from
    primary_modality: Modality

    # Spatial context (may be None for pure voice)
    zone: Optional[str] = None
    gaze_target: Optional[str] = None
    gesture_type: Optional[str] = None

    # Is this a system control? (handled before brain)
    surface_control: Optional[SurfaceControl] = None

    # Original text before gaze substitution
    raw_text: str = ""

    timestamp: float = field(default_factory=time.time)

    @property
    def is_system_control(self) -> bool:
        return self.surface_control is not None

    @property
    def has_spatial_context(self) -> bool:
        return bool(self.zone or self.gaze_target or self.gesture_type)


# Deictic words that can be resolved via gaze
DEICTIC_WORDS = {
    "that", "this", "those", "these",
    "it", "there", "here",
    "the one", "that one", "this one",
}

# How long (seconds) a gesture window stays open after recognition
GESTURE_TEMPORAL_WINDOW = 2.0


class IntentFusion:
    """
    Collects signals from all input modalities.
    Voice events arrive via process_voice().
    Gesture events arrive via process_gesture().
    Gaze events arrive via process_gaze().
    Surface control events arrive via process_surface().

    Fused intents are retrieved via get() or the fused_intents generator.
    """

    def __init__(self):
        self._output_queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()

        # Current state of each modality
        self._gaze   = GazeState()
        self._gesture = GestureState()

        # Pending voice utterance waiting for gesture window to close
        self._pending_voice: Optional[str] = None
        self._pending_voice_ts: float = 0.0

        # Whether the system is currently paused (surface control)
        self._paused = False

        # Volume level 0.0–2.0
        self._volume: float = 1.0
        self._volume_step: float = 0.1

        # Personality list for cycling
        self._personalities = ["iris", "nova", "vector", "solace", "lilac"]
        self._personality_idx: int = 1  # default: nova

    # ── INPUT METHODS ─────────────────────────────────────────────────────────

    def process_voice(self, text: str):
        """
        Called by VoicePerception when a new utterance is transcribed.
        Enriches with gaze context, then emits a FusedIntent.
        """
        if not text.strip():
            return

        if self._paused:
            logger.debug("Paused — ignoring voice input.")
            return

        raw_text = text
        enriched_text = text

        # Gaze substitution — replace deictic references with gazed target
        if self._gaze.target_id and self._gaze.fixation_duration_ms > 300:
            enriched_text = self._substitute_deictics(text, self._gaze.target_id)

        # Check if there's an active gesture to merge in
        gesture_context = None
        with self._lock:
            if (self._gesture.is_active and
                    time.time() - self._gesture.timestamp < GESTURE_TEMPORAL_WINDOW):
                gesture_context = self._gesture.gesture_type

        intent = FusedIntent(
            text=enriched_text,
            primary_modality=Modality.VOICE,
            zone=self._gesture.zone if gesture_context else None,
            gaze_target=self._gaze.target_id or None,
            gesture_type=gesture_context,
            raw_text=raw_text,
        )

        logger.debug(
            f"Voice intent: '{text[:50]}'"
            + (f" [gaze: {self._gaze.target_id}]" if self._gaze.target_id else "")
            + (f" [gesture: {gesture_context}]" if gesture_context else "")
        )

        self._output_queue.put(intent)

    def process_gesture(self, event):
        """
        Called by GesturePerception with a GestureEvent.
        Panel and fog gestures that have no pending voice command
        emit their own FusedIntent.
        """
        from perception.gesture import GestureType, Zone, gesture_to_hologram_directive

        with self._lock:
            self._gesture = GestureState(
                gesture_type=event.gesture.value,
                zone=event.zone.value,
                hand=event.hand,
                timestamp=event.timestamp,
                is_active=True,
            )

        # Pure gesture intent — no voice needed for spatial operations
        if event.zone in (Zone.PANEL, Zone.FOG):
            directive = gesture_to_hologram_directive(event)
            if directive:
                gesture_text = self._gesture_to_text(event)
                if gesture_text:
                    intent = FusedIntent(
                        text=gesture_text,
                        primary_modality=Modality.GESTURE,
                        zone=event.zone.value,
                        gesture_type=event.gesture.value,
                    )
                    self._output_queue.put(intent)

    def process_gaze(self, target_type: str, target_id: str,
                     fixation_ms: float, position_3d=None):
        """
        Called by gaze tracker when fixation is stable (>300ms).
        Updates internal gaze state — does not emit intents directly.
        """
        with self._lock:
            self._gaze = GazeState(
                target_type=target_type,
                target_id=target_id,
                fixation_duration_ms=fixation_ms,
                position_3d=position_3d,
            )

    def process_surface(self, control: SurfaceControl, value: float = None):
        """
        Called by capacitive surface strip when a control is activated.
        Emits a FusedIntent with surface_control set.
        """
        # Handle locally if possible
        response_text = None

        if control == SurfaceControl.EMERGENCY_MUTE:
            self._paused = True
            response_text = None  # Silent

        elif control == SurfaceControl.PAUSE_RESUME:
            self._paused = not self._paused
            response_text = None  # Handled in main loop

        elif control == SurfaceControl.VOLUME_UP:
            self._volume = min(2.0, self._volume + self._volume_step)
            logger.info(f"Volume → {self._volume:.1f}")

        elif control == SurfaceControl.VOLUME_DOWN:
            self._volume = max(0.0, self._volume - self._volume_step)
            logger.info(f"Volume → {self._volume:.1f}")

        elif control == SurfaceControl.PERSONALITY_NEXT:
            self._personality_idx = (self._personality_idx + 1) % len(self._personalities)

        intent = FusedIntent(
            text="",
            primary_modality=Modality.SURFACE,
            surface_control=control,
        )
        self._output_queue.put(intent)

    # ── OUTPUT ────────────────────────────────────────────────────────────────

    def get(self, timeout: float = 0.5) -> Optional[FusedIntent]:
        """
        Get the next FusedIntent (blocking with timeout).
        Returns None on timeout.
        """
        try:
            return self._output_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def fused_intents(self):
        """
        Generator that yields FusedIntents indefinitely.
        Use in the main loop.
        """
        while True:
            intent = self.get(timeout=0.5)
            if intent is not None:
                yield intent

    # ── HELPERS ───────────────────────────────────────────────────────────────

    def _substitute_deictics(self, text: str, target_id: str) -> str:
        """
        Replace deictic references in text with the gazed-at target.
        "summarize that" + gaze=panel:project_notes → "summarize the project notes panel"
        """
        text_lower = text.lower()
        for word in DEICTIC_WORDS:
            if word in text_lower:
                # Format the target ID nicely
                target_label = target_id.replace("_", " ").replace("-", " ")
                enriched = text_lower.replace(word, f"the {target_label}", 1)
                logger.debug(f"Deictic substitution: '{text}' → '{enriched}'")
                return enriched
        return text

    def _gesture_to_text(self, event) -> Optional[str]:
        """
        Convert a gesture event to a natural language command for the brain.
        Returns None for gestures handled entirely by the hologram
        (pure UI interactions that don't need language reasoning).
        """
        from perception.gesture import GestureType, Zone

        # Panel zone — UI only, brain not involved
        if event.zone == Zone.PANEL:
            return None

        # Fog zone — spatial manipulation, may need brain context
        mapping = {
            GestureType.GRAB:       "grab the object in the fog display",
            GestureType.THROW:      "dismiss the current fog display object",
            GestureType.FIST:       "close the fog volume",
        }
        return mapping.get(event.gesture)

    @property
    def current_volume(self) -> float:
        return self._volume

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def next_personality(self) -> str:
        return self._personalities[self._personality_idx]
