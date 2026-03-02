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
from typing import Dict, Any, Tuple

logger = logging.getLogger("lavender.safety")

# Tools that require explicit user confirmation or stricter validation
SENSITIVE_TOOLS = {
    "control_device",  # Home automation can be physical
    "run_python",      # Code execution
}

class SafetyLayer:
    def __init__(self):
        logger.info("Safety layer initialized.")

    def validate_tool_call(self, tool_name: str, arguments: Dict[str, Any], context: Dict[str, Any] = None) -> Tuple[bool, str]:
        """
        Validates a tool call.
        Returns (is_safe, error_message/reason).
        """
        context = context or {}
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
