"""
LAVENDER — Gesture Perception
perception/gesture.py

Reads the Intel RealSense D435i depth + color stream.
Uses MediaPipe Hands for 21-landmark hand detection.
Maps fingertip depth to 3D coordinates relative to the table zones.

Three interaction zones with distinct gesture vocabularies:
  PANEL_ZONE  — above the holographic panel (2D control)
  FOG_ZONE    — above/inside the fog well (3D manipulation)
  SURFACE_ZONE — direct table surface touch

Yields GestureEvent objects consumed by the Intent Fusion Layer.

Hardware required for full function:
  - Intel RealSense D435i (primary — panel zone)
  - Intel RealSense D415 (secondary — fog zone, optional)

Runs without hardware using mock mode for development:
  python perception/gesture.py --mock
"""

import time
import logging
import threading
import queue
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Generator
import numpy as np

logger = logging.getLogger("lavender.gesture")


# ── ZONES ────────────────────────────────────────────────────────────────────

class Zone(str, Enum):
    PANEL   = "panel"    # 2D holographic panel interaction
    FOG     = "fog"      # 3D fog volume manipulation
    SURFACE = "surface"  # Table surface touch
    NONE    = "none"     # Outside all zones / not tracked


# ── GESTURE TYPES ────────────────────────────────────────────────────────────

class GestureType(str, Enum):
    # Universal
    POINT      = "point"       # Single index finger extended
    OPEN_HAND  = "open_hand"   # All fingers extended
    FIST       = "fist"        # All fingers curled
    PINCH      = "pinch"       # Thumb + index close together
    PEACE      = "peace"       # Index + middle extended (V)
    THUMBS_UP  = "thumbs_up"
    SWIPE_LEFT  = "swipe_left"
    SWIPE_RIGHT = "swipe_right"
    SWIPE_UP    = "swipe_up"
    SWIPE_DOWN  = "swipe_down"

    # Panel zone specific
    PRESS        = "press"        # Fingertip crosses panel plane
    PINCH_OPEN   = "pinch_open"   # Pinch expanding (zoom in)
    PINCH_CLOSE  = "pinch_close"  # Pinch contracting (zoom out)
    TWO_HAND_SPREAD = "two_hand_spread"
    DISMISS      = "dismiss"      # Flick away from panel

    # Fog zone specific
    GRAB          = "grab"         # Close fist in fog volume
    ROTATE_CW     = "rotate_cw"
    ROTATE_CCW    = "rotate_ccw"
    SCALE_UP      = "scale_up"     # Two hands moving apart
    SCALE_DOWN    = "scale_down"   # Two hands moving together
    THROW         = "throw"        # Fast fling out of fog zone

    # None / unknown
    UNKNOWN = "unknown"


# ── DATA STRUCTURES ───────────────────────────────────────────────────────────

@dataclass
class HandPose:
    """
    Processed pose for one hand at one moment.
    Coordinates are in meters, relative to camera.
    """
    landmarks_2d: np.ndarray     # Shape (21, 2) — pixel coordinates
    landmarks_3d: np.ndarray     # Shape (21, 3) — x, y, depth in meters
    handedness: str              # "Left" or "Right"
    confidence: float

    # Derived
    index_tip: np.ndarray        # 3D position of index fingertip
    thumb_tip: np.ndarray        # 3D position of thumb tip
    wrist: np.ndarray            # 3D position of wrist

    @property
    def pinch_distance(self) -> float:
        """Distance between thumb tip and index fingertip in meters."""
        return float(np.linalg.norm(self.thumb_tip - self.index_tip))

    @property
    def is_pinching(self) -> bool:
        return self.pinch_distance < 0.03  # 3cm threshold


@dataclass
class GestureEvent:
    """
    A recognized gesture with full context.
    Consumed by the Intent Fusion Layer.
    """
    gesture:    GestureType
    zone:       Zone
    hand:       str              # "Left", "Right", "Both"
    position_3d: np.ndarray      # 3D position in table-space (meters)
    velocity:   np.ndarray       # Movement velocity vector
    confidence: float
    timestamp:  float = field(default_factory=time.time)
    duration_ms: float = 0.0     # How long the gesture has been held
    metadata:   dict = field(default_factory=dict)  # gesture-specific data

    @property
    def is_panel_control(self) -> bool:
        return self.zone == Zone.PANEL

    @property
    def is_fog_control(self) -> bool:
        return self.zone == Zone.FOG


# ── ZONE BOUNDS (calibrated to table geometry) ───────────────────────────────
# These are in camera coordinates (meters).
# Calibrate to your actual table by running:
#   python perception/gesture.py --calibrate

class ZoneBounds:
    """
    Defines the 3D bounding boxes for each interaction zone.
    Loaded from calibration file if it exists, else falls back to defaults.
    Z = depth from camera (meters). Larger = further from camera.
    Run: python scripts/calibrate_zones.py to generate calibration.
    """

    _calibration_loaded = False

    @classmethod
    def load_calibration(cls):
        """Load calibrated zone bounds from config/zone_calibration.yaml if present."""
        if cls._calibration_loaded:
            return
        cal_path = Path(__file__).parent.parent / "config" / "zone_calibration.yaml"
        if cal_path.exists():
            try:
                import yaml
                with open(cal_path) as f:
                    data = yaml.safe_load(f)
                zones = data.get("zones", {})
                panel = zones.get("panel", {})
                fog   = zones.get("fog", {})
                if panel:
                    cls.PANEL_Z_MIN   = panel.get("z_min",   cls.PANEL_Z_MIN)
                    cls.PANEL_Z_MAX   = panel.get("z_max",   cls.PANEL_Z_MAX)
                    cls.PANEL_X_MIN   = panel.get("x_min",   cls.PANEL_X_MIN)
                    cls.PANEL_X_MAX   = panel.get("x_max",   cls.PANEL_X_MAX)
                    cls.PANEL_Y_MIN   = panel.get("y_min",   cls.PANEL_Y_MIN)
                    cls.PANEL_Y_MAX   = panel.get("y_max",   cls.PANEL_Y_MAX)
                    cls.PANEL_PLANE_Z = panel.get("z_plane", cls.PANEL_PLANE_Z)
                if fog:
                    cls.FOG_Z_MIN = fog.get("z_min", cls.FOG_Z_MIN)
                    cls.FOG_Z_MAX = fog.get("z_max", cls.FOG_Z_MAX)
                    cls.FOG_X_MIN = fog.get("x_min", cls.FOG_X_MIN)
                    cls.FOG_X_MAX = fog.get("x_max", cls.FOG_X_MAX)
                logger.info("Zone calibration loaded from config/zone_calibration.yaml")
            except Exception as e:
                logger.warning(f"Could not load zone calibration: {e}")
        cls._calibration_loaded = True

    # Panel zone: in front of the holographic display
    PANEL_Z_MIN = 0.30   # 30cm from camera
    PANEL_Z_MAX = 0.70   # 70cm from camera
    PANEL_X_MIN = -0.35  # 35cm left of center
    PANEL_X_MAX =  0.35  # 35cm right of center
    PANEL_Y_MIN = -0.20  # 20cm below center
    PANEL_Y_MAX =  0.30  # 30cm above center

    # Fog zone: above the recessed well (left side of table)
    FOG_Z_MIN = 0.45
    FOG_Z_MAX = 0.90
    FOG_X_MIN = -0.55
    FOG_X_MAX = -0.15
    FOG_Y_MIN = -0.10
    FOG_Y_MAX =  0.40

    # Panel plane (the holographic surface itself)
    # When fingertip crosses this plane, it registers as PRESS
    PANEL_PLANE_Z = 0.42  # meters from camera

    @classmethod
    def classify_point(cls, x: float, y: float, z: float) -> Zone:
        cls.load_calibration()
        if (cls.FOG_Z_MIN < z < cls.FOG_Z_MAX and
                cls.FOG_X_MIN < x < cls.FOG_X_MAX and
                cls.FOG_Y_MIN < y < cls.FOG_Y_MAX):
            return Zone.FOG

        if (cls.PANEL_Z_MIN < z < cls.PANEL_Z_MAX and
                cls.PANEL_X_MIN < x < cls.PANEL_X_MAX and
                cls.PANEL_Y_MIN < y < cls.PANEL_Y_MAX):
            return Zone.PANEL

        return Zone.NONE


# ── GESTURE CLASSIFIER ────────────────────────────────────────────────────────

class GestureClassifier:
    """
    Takes MediaPipe landmark arrays and classifies gesture type.
    Uses simple geometric rules — no ML needed for this vocabulary.
    """

    # MediaPipe landmark indices
    WRIST       = 0
    THUMB_TIP   = 4
    INDEX_MCP   = 5;  INDEX_TIP   = 8
    MIDDLE_MCP  = 9;  MIDDLE_TIP  = 12
    RING_MCP    = 13; RING_TIP    = 16
    PINKY_MCP   = 17; PINKY_TIP   = 20

    def classify(self, landmarks: np.ndarray, handedness: str) -> GestureType:
        """
        landmarks: shape (21, 3) — normalized 0-1 coordinates from MediaPipe
        Returns the most likely gesture type.
        """
        if landmarks is None or len(landmarks) < 21:
            return GestureType.UNKNOWN

        fingers_extended = self._fingers_extended(landmarks)
        pinch_dist = np.linalg.norm(
            landmarks[self.THUMB_TIP] - landmarks[self.INDEX_TIP]
        )

        n_extended = sum(fingers_extended)

        # FIST — no fingers extended
        if n_extended == 0:
            return GestureType.FIST

        # POINT — only index extended
        if fingers_extended == [False, True, False, False, False]:
            return GestureType.POINT

        # PEACE — index + middle extended
        if fingers_extended == [False, True, True, False, False]:
            return GestureType.PEACE

        # OPEN HAND — all extended
        if n_extended >= 4:
            return GestureType.OPEN_HAND

        # PINCH — thumb + index close
        if pinch_dist < 0.08:  # normalized threshold
            return GestureType.PINCH

        # THUMBS UP — thumb extended, others curled
        if fingers_extended == [True, False, False, False, False]:
            return GestureType.THUMBS_UP

        return GestureType.UNKNOWN

    def _fingers_extended(self, lm: np.ndarray) -> list[bool]:
        """
        Returns [thumb, index, middle, ring, pinky] extension status.
        A finger is extended if its tip is further from the wrist than its MCP.
        """
        wrist = lm[self.WRIST]

        # Thumb: compare tip to IP joint (different axis)
        thumb_ext = np.linalg.norm(lm[self.THUMB_TIP] - wrist) > \
                    np.linalg.norm(lm[3] - wrist)

        # Other fingers: tip further from wrist than MCP
        index_ext  = np.linalg.norm(lm[self.INDEX_TIP]  - wrist) > \
                     np.linalg.norm(lm[self.INDEX_MCP]   - wrist) * 1.1
        middle_ext = np.linalg.norm(lm[self.MIDDLE_TIP] - wrist) > \
                     np.linalg.norm(lm[self.MIDDLE_MCP]  - wrist) * 1.1
        ring_ext   = np.linalg.norm(lm[self.RING_TIP]   - wrist) > \
                     np.linalg.norm(lm[self.RING_MCP]    - wrist) * 1.1
        pinky_ext  = np.linalg.norm(lm[self.PINKY_TIP]  - wrist) > \
                     np.linalg.norm(lm[self.PINKY_MCP]   - wrist) * 1.1

        return [thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext]


# ── SWIPE DETECTOR ─────────────────────────────────────────────────────────────

class SwipeDetector:
    """
    Detects directional swipes by tracking fingertip position over time.
    A swipe is a fast, directional movement starting with POINT and ending open.
    """

    def __init__(self, history_size: int = 10, min_velocity: float = 0.4):
        self.history_size = history_size
        self.min_velocity = min_velocity   # m/s minimum to register as swipe
        self._positions: list[tuple[float, np.ndarray]] = []  # (timestamp, pos)

    def update(self, position: np.ndarray, timestamp: float = None) -> Optional[GestureType]:
        """
        Add a position sample. Returns a swipe direction if one was just completed,
        or None if no swipe detected.
        """
        ts = timestamp or time.time()
        self._positions.append((ts, position.copy()))

        if len(self._positions) > self.history_size:
            self._positions.pop(0)

        if len(self._positions) < 4:
            return None

        return self._detect_swipe()

    def _detect_swipe(self) -> Optional[GestureType]:
        oldest_ts, oldest_pos = self._positions[0]
        newest_ts, newest_pos = self._positions[-1]

        dt = newest_ts - oldest_ts
        if dt < 0.01:
            return None

        displacement = newest_pos - oldest_pos
        velocity = np.linalg.norm(displacement) / dt

        if velocity < self.min_velocity:
            return None

        # Dominant direction
        dx, dy = displacement[0], displacement[1]
        if abs(dx) > abs(dy):
            return GestureType.SWIPE_RIGHT if dx > 0 else GestureType.SWIPE_LEFT
        else:
            return GestureType.SWIPE_UP if dy > 0 else GestureType.SWIPE_DOWN

    def reset(self):
        self._positions.clear()


# ── MAIN GESTURE PERCEPTION CLASS ────────────────────────────────────────────

class GesturePerception:
    """
    Main entry point for gesture perception.

    Runs a background thread reading from RealSense.
    Yields GestureEvent objects from the gesture() generator.
    """

    # How long a gesture must be held before registering (ms)
    HOLD_THRESHOLD_PANEL = 120
    HOLD_THRESHOLD_FOG   = 80   # Fog zone — entering is already intentional

    def __init__(
        self,
        device_serial: str = None,   # None = first available device
        mock: bool = False,
        event_queue_size: int = 64,
    ):
        self.mock = mock
        self._device_serial = device_serial
        self._event_queue: queue.Queue = queue.Queue(maxsize=event_queue_size)
        self._stop_event = threading.Event()

        self._classifier = GestureClassifier()
        self._swipe_detector = SwipeDetector()

        # Gesture hold state
        self._current_gesture: Optional[GestureType] = None
        self._gesture_start: float = 0.0
        self._gesture_registered = False

        # Previous hand positions for velocity calculation
        self._prev_positions: dict[str, np.ndarray] = {}
        self._prev_time: float = 0.0

        if not mock:
            self._init_hardware()
        else:
            logger.info("Gesture perception running in MOCK mode.")

    def _init_hardware(self):
        """Initialize RealSense pipeline."""
        try:
            import pyrealsense2 as rs
            self._rs = rs

            self._pipeline = rs.pipeline()
            config = rs.config()

            if self._device_serial:
                config.enable_device(self._device_serial)

            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

            self._profile = self._pipeline.start(config)
            self._align   = rs.align(rs.stream.color)

            # Get depth scale
            depth_sensor = self._profile.get_device().first_depth_sensor()
            self._depth_scale = depth_sensor.get_depth_scale()

            logger.info("RealSense pipeline started.")

        except ImportError:
            logger.warning("pyrealsense2 not installed. Falling back to mock mode.")
            self.mock = True
        except Exception as e:
            logger.warning(f"RealSense init failed: {e}. Falling back to mock mode.")
            self.mock = True

    # ── BACKGROUND THREAD ─────────────────────────────────────────────────────

    def start(self):
        """Start the gesture perception thread."""
        self._thread = threading.Thread(
            target=self._run,
            name="gesture-perception",
            daemon=True,
        )
        self._thread.start()
        logger.info("Gesture perception thread started.")

    def stop(self):
        self._stop_event.set()

    def _run(self):
        if self.mock:
            self._run_mock()
        else:
            self._run_real()

    def _run_real(self):
        """Real RealSense processing loop."""
        try:
            import mediapipe as mp
            mp_hands = mp.solutions.hands
            hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.6,
            )
        except ImportError:
            logger.error("mediapipe not installed. Run: pip install mediapipe")
            return

        while not self._stop_event.is_set():
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=100)
                aligned = self._align.process(frames)

                depth_frame = aligned.get_depth_frame()
                color_frame = aligned.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                import cv2
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())

                # RGB for MediaPipe
                rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                if not results.multi_hand_landmarks:
                    self._gesture_start = 0.0
                    self._current_gesture = None
                    self._gesture_registered = False
                    continue

                h, w = color_image.shape[:2]
                now = time.time()

                for hand_idx, (hand_lm, hand_class) in enumerate(zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness,
                )):
                    handedness = hand_class.classification[0].label

                    # Extract landmark arrays
                    lm_2d = np.array([[lm.x * w, lm.y * h]
                                      for lm in hand_lm.landmark])
                    lm_norm = np.array([[lm.x, lm.y, lm.z]
                                        for lm in hand_lm.landmark])

                    # Get depth at index fingertip
                    ix, iy = int(lm_2d[8][0]), int(lm_2d[8][1])
                    ix = max(0, min(w-1, ix))
                    iy = max(0, min(h-1, iy))

                    tip_depth_m = (
                        depth_frame.get_distance(ix, iy)
                        or depth_image[iy, ix] * self._depth_scale
                    )

                    # Build 3D tip position (camera space)
                    cx = lm_norm[8][0] - 0.5  # center at 0
                    cy = -(lm_norm[8][1] - 0.5)  # flip Y
                    tip_3d = np.array([cx * tip_depth_m * 1.3,
                                       cy * tip_depth_m * 1.3,
                                       tip_depth_m])

                    # Classify zone
                    zone = ZoneBounds.classify_point(*tip_3d)

                    # Classify gesture
                    gesture_type = self._classifier.classify(lm_norm, handedness)

                    # Check for swipe
                    swipe = self._swipe_detector.update(tip_3d, now)
                    if swipe:
                        gesture_type = swipe
                        self._swipe_detector.reset()

                    # Velocity
                    velocity = np.zeros(3)
                    if handedness in self._prev_positions and self._prev_time > 0:
                        dt = now - self._prev_time
                        if dt > 0:
                            velocity = (tip_3d - self._prev_positions[handedness]) / dt

                    self._prev_positions[handedness] = tip_3d
                    self._prev_time = now

                    # Hold threshold check
                    threshold_ms = (self.HOLD_THRESHOLD_FOG
                                    if zone == Zone.FOG
                                    else self.HOLD_THRESHOLD_PANEL)

                    if gesture_type != self._current_gesture:
                        self._current_gesture = gesture_type
                        self._gesture_start = now
                        self._gesture_registered = False
                    else:
                        held_ms = (now - self._gesture_start) * 1000
                        if (not self._gesture_registered and
                                held_ms >= threshold_ms and
                                zone != Zone.NONE and
                                gesture_type != GestureType.UNKNOWN):

                            event = GestureEvent(
                                gesture=gesture_type,
                                zone=zone,
                                hand=handedness,
                                position_3d=tip_3d,
                                velocity=velocity,
                                confidence=hand_class.classification[0].score,
                                duration_ms=held_ms,
                            )

                            try:
                                self._event_queue.put_nowait(event)
                                self._gesture_registered = True
                                logger.debug(
                                    f"Gesture: {gesture_type.value} "
                                    f"in {zone.value} "
                                    f"({handedness})"
                                )
                            except queue.Full:
                                pass  # Drop if queue full

            except Exception as e:
                if not self._stop_event.is_set():
                    logger.error(f"Gesture loop error: {e}")
                    time.sleep(0.1)

        try:
            self._pipeline.stop()
        except Exception:
            pass

    def _run_mock(self):
        """
        Mock mode — generates synthetic gesture events for development.
        Cycles through a realistic sequence every few seconds.
        """
        import random
        mock_sequence = [
            (GestureType.POINT,    Zone.PANEL, "Right"),
            (GestureType.SWIPE_RIGHT, Zone.PANEL, "Right"),
            (GestureType.PINCH,    Zone.PANEL, "Right"),
            (GestureType.OPEN_HAND, Zone.PANEL, "Right"),
            (GestureType.GRAB,     Zone.FOG,   "Right"),
            (GestureType.ROTATE_CW, Zone.FOG,  "Right"),
            (GestureType.FIST,     Zone.FOG,   "Right"),
        ]

        idx = 0
        while not self._stop_event.is_set():
            time.sleep(3.0 + random.uniform(0, 2))
            if self._stop_event.is_set():
                break

            gesture_type, zone, hand = mock_sequence[idx % len(mock_sequence)]
            idx += 1

            event = GestureEvent(
                gesture=gesture_type,
                zone=zone,
                hand=hand,
                position_3d=np.array([0.0, 0.0, 0.5]),
                velocity=np.zeros(3),
                confidence=0.9,
            )
            try:
                self._event_queue.put_nowait(event)
            except queue.Full:
                pass

    # ── GENERATOR INTERFACE ───────────────────────────────────────────────────

    def gesture(self) -> Generator[GestureEvent, None, None]:
        """
        Generator that yields GestureEvents.
        Blocks until the next event, or raises StopIteration on stop().

        Usage:
            for event in gesture_perception.gesture():
                handle(event)
        """
        while not self._stop_event.is_set():
            try:
                event = self._event_queue.get(timeout=0.5)
                yield event
            except queue.Empty:
                continue


# ── MAP GESTURES TO HOLOGRAM ACTIONS ─────────────────────────────────────────

def gesture_to_hologram_directive(event: GestureEvent) -> Optional[dict]:
    """
    Maps a GestureEvent to a hologram directive dict.
    Returned dict is passed to HologramDirector.
    Returns None if gesture needs no display response.
    """
    g = event.gesture
    z = event.zone

    if z == Zone.PANEL:
        mapping = {
            GestureType.SWIPE_LEFT:  {"action": "next_panel"},
            GestureType.SWIPE_RIGHT: {"action": "prev_panel"},
            GestureType.SWIPE_UP:    {"action": "scroll_up"},
            GestureType.SWIPE_DOWN:  {"action": "scroll_down"},
            GestureType.PINCH_OPEN:  {"action": "zoom_in"},
            GestureType.PINCH_CLOSE: {"action": "zoom_out"},
            GestureType.OPEN_HAND:   {"action": "dismiss_panel"},
            GestureType.PRESS:       {"action": "select"},
        }
        return mapping.get(g)

    if z == Zone.FOG:
        mapping = {
            GestureType.GRAB:       {"action": "grab_object"},
            GestureType.ROTATE_CW:  {"action": "rotate_cw"},
            GestureType.ROTATE_CCW: {"action": "rotate_ccw"},
            GestureType.SCALE_UP:   {"action": "scale_up"},
            GestureType.SCALE_DOWN: {"action": "scale_down"},
            GestureType.THROW:      {"action": "dismiss_fog_object"},
            GestureType.OPEN_HAND:  {"action": "release"},
        }
        return mapping.get(g)

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# python perception/gesture.py          — requires RealSense
# python perception/gesture.py --mock   — no hardware needed
# python perception/gesture.py --calibrate — interactive zone calibration
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    mock = "--mock" in sys.argv

    print(f"\nGesture Perception Test {'(MOCK MODE)' if mock else '(HARDWARE)'}")
    print("Move your hand in front of the RealSense camera.")
    print("Ctrl+C to stop.\n")

    gp = GesturePerception(mock=mock)
    gp.start()

    try:
        for event in gp.gesture():
            print(
                f"  [{event.zone.value:7s}] "
                f"{event.gesture.value:20s} "
                f"({event.hand}) "
                f"pos=({event.position_3d[0]:.2f}, "
                f"{event.position_3d[1]:.2f}, "
                f"{event.position_3d[2]:.2f}m) "
                f"conf={event.confidence:.2f}"
            )
    except KeyboardInterrupt:
        gp.stop()
        print("\nStopped.")
