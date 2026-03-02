"""
LAVENDER — Communication & Notification Tools
tools/communication.py

Provides tools for alerting the user and communicating externally:
  - System notifications
  - Email (via SMTP)
  - SMS/Calls (placeholder for Twilio/API)
"""

import logging
import subprocess
from typing import List, Optional
from langchain_core.tools import tool

logger = logging.getLogger("lavender.communication")

def send_system_notification(title: str, message: str) -> str:
    """Sends a desktop notification using notify-send."""
    try:
        subprocess.run(["notify-send", "-a", "Lavender", title, message], check=True)
        return "Notification sent."
    except Exception as e:
        return f"Failed to send notification: {e}"

def make_communication_tools() -> List:
    @tool
    def send_notification(title: str, message: str) -> str:
        """
        Sends a desktop notification to the user.
        Use this for important alerts or task completions.
        """
        return send_system_notification(title, message)

    @tool
    def make_call(recipient: str, reason: str) -> str:
        """
        Initiates a call to a recipient.
        Note: Currently requires manual bridging or a configured VoIP provider.
        """
        # Placeholder for Twilio or similar
        return f"Initiating call to {recipient} regarding: {reason}. (SIMULATED)"

    return [send_notification, make_call]
