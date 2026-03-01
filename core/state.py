"""
LAVENDER — State Engine
core/state.py

Manages the global world model:
  - User state (focus, idle, talking, away)
  - Environment context (time, ambient noise, presence)
  - System state (active tasks, memory status, health)

The intelligence layer (brain.py) and proactive layer (proactive.py)
read from here to decide HOW and WHEN to interact.
"""

import time
import logging
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

logger = logging.getLogger("lavender.state")

class UserState(str, Enum):
    IDLE      = "idle"       # Present but not interacting
    FOCUS     = "focus"      # Deep work, should not be interrupted
    TALKING   = "talking"    # Actively speaking to Lavender
    STRESSED  = "stressed"   # High interaction frequency / fast pace
    AWAY      = "away"       # Not detected by sensors
    UNKNOWN   = "unknown"

class EnvState(str, Enum):
    QUIET     = "quiet"
    NORMAL    = "normal"
    NOISY     = "noisy"

class SystemStatus(str, Enum):
    NOMINAL   = "nominal"
    BUSY      = "busy"       # Processing complex task / tool
    DEGRADED  = "degraded"   # Component failure (monitored by health.py)
    CRITICAL  = "critical"   # Major system issue

@dataclass
class WorldState:
    # User
    user: UserState = UserState.UNKNOWN
    last_user_interaction: float = field(default_factory=time.time)
    interaction_count_1h: int = 0

    # Environment
    env: EnvState = EnvState.NORMAL
    is_dark: bool = False
    ambient_noise_db: float = 0.0

    # System
    status: SystemStatus = SystemStatus.NOMINAL
    active_personality: str = "nova"
    active_tasks: list = field(default_factory=list)

    # Telemetry
    cpu_usage: float = 0.0
    mem_usage: float = 0.0
    gpu_usage: float = 0.0

class StateEngine:
    """
    Central manager for Lavender's world model.
    Updated by sensors (voice, gesture) and system monitors (health).
    """

    def __init__(self):
        self.state = WorldState()
        self._start_time = time.time()
        logger.info("State engine initialized.")

    def update_user_activity(self):
        """Called whenever user interacts with the system."""
        now = time.time()
        self.state.last_user_interaction = now
        self.state.interaction_count_1h += 1

        # If very frequent interactions, flag as STRESSED or high-focus
        # (Simple heuristic for now)
        if self.state.user != UserState.TALKING:
            self.set_user_state(UserState.TALKING)

    def set_user_state(self, new_state: UserState):
        if self.state.user != new_state:
            logger.info(f"User state transition: {self.state.user} -> {new_state}")
            self.state.user = new_state

    def set_system_status(self, status: SystemStatus):
        self.state.status = status

    def get_context_summary(self) -> Dict[str, Any]:
        """Returns a dict suitable for LLM context injection."""
        return {
            "user_state": self.state.user.value,
            "system_status": self.state.status.value,
            "active_personality": self.state.active_personality,
            "time_of_day": time.strftime("%H:%M"),
            "is_idle": (time.time() - self.state.last_user_interaction) > 300, # 5 mins
        }

    def __repr__(self):
        return f"<StateEngine user={self.state.user} status={self.state.status}>"

# Global instance for easy access across modules
# In a real system, this would be passed via dependency injection or a registry.
instance = StateEngine()
