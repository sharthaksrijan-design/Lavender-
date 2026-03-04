"""
LAVENDER — Action Safety Layer
core/safety.py

Validates tool calls before execution:
  - Permission checks
  - Intent validation
  - Simulation / Dry-run (where applicable)
  - Sensitive action confirmation
"""

import logging
import time
from typing import Dict, Any, Tuple

logger = logging.getLogger("lavender.safety")

# Tools that require explicit user confirmation or stricter validation
SENSITIVE_TOOLS = {
    "control_device",   # Home automation can be physical
    "run_python",       # Code execution
    "send_notification", # Disturbance
    "social_post",      # Public reputation
    "system_command",   # OS level changes
    "make_call",        # External contact
    "deploy_new_tool",  # Self-modification
}

# Tools that are restricted if the user is not focused/active
SENSITIVE_CONTEXT_TOOLS = {
    "control_device",
    "social_post",
    "make_call",
}

class SafetyLayer:
    def __init__(self):
        self._hard_stop_active = False
        self._user_confirmed = False # Current turn confirmation
        self._interaction_history = []
        logger.info("Safety layer initialized.")

    def set_user_confirmed(self, value: bool):
        self._user_confirmed = value

    def activate_hard_stop(self):
        logger.warning("HARD STOP ACTIVATED. All tool execution blocked.")
        self._hard_stop_active = True

    def reset_hard_stop(self):
        self._hard_stop_active = False

    def validate_tool_call(self, tool_name: str, arguments: Dict[str, Any], context: Dict[str, Any] = None) -> Tuple[bool, str]:
        """
        Validates a tool call with multi-factor check.
        """
        if self._hard_stop_active:
            return False, "Hard stop is active. System is in lockdown."

        context = context or {}
        from core.state import instance as state
        world_state = state.get_context_summary()

        # ── FACTOR 1: EXPLICIT CONFIRMATION ──
        confirmed = context.get("user_confirmed", False) or self._user_confirmed

        # ── FACTOR 2: PROXIMITY & ATTENTION ──
        # If user is away or idle, block sensitive context tools even if LLM thinks it should proceed
        if tool_name in SENSITIVE_CONTEXT_TOOLS:
            if world_state.get('user_state') in ('away', 'idle') and not confirmed:
                return False, f"User is not active. {tool_name} blocked for safety."

        # ── FACTOR 3: FREQUENCY & ANOMALY ──
        # Track interactions to prevent 'loop attacks' or rapid-fire actions
        self._interaction_history.append((tool_name, time.time()))
        recent = [t for n, t in self._interaction_history if t > time.time() - 60]
        if len(recent) > 10:
            self.activate_hard_stop()
            return False, "Rate limit exceeded. Emergency lockdown activated."

        # 1. Human-in-the-loop check for sensitive tools
        if tool_name in SENSITIVE_TOOLS:
            if not confirmed:
                return False, f"Confirmation required for sensitive action: {tool_name}"

        if tool_name == "run_python":
            code = arguments.get("code", "")
            # Basic sanity check (CodeRunner already has a sandbox, but we can add meta-rules here)
            if "os.system" in code or "subprocess" in code:
                return False, "Direct system calls are prohibited even in sandbox."

        if tool_name == "control_device":
            # Prevent accidental bulk actions if not clearly intended
            entity = arguments.get("entity", "")
            if entity == "all" or "*" in entity:
                return False, "Bulk device control requires explicit confirmation."

        return True, ""

    def simulate_outcome(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Optional: returns a string describing what WILL happen."""
        return f"Executing {tool_name} with args {arguments}"

instance = SafetyLayer()
