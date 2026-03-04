"""
LAVENDER — Audit Logging
core/audit.py
"""

import logging
import json
import time
from pathlib import Path
from typing import Any, Dict

class AuditLogger:
    def __init__(self, log_path: str = "logs/audit.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(exist_ok=True)
        self.logger = logging.getLogger("lavender.audit")

    def log_action(self, action_type: str, actor: str, details: Dict[str, Any], outcome: str = "success"):
        entry = {
            "timestamp": time.time(),
            "action": action_type,
            "actor": actor,
            "details": details,
            "outcome": outcome
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        self.logger.info(f"AUDIT: {action_type} by {actor} - {outcome}")

instance = AuditLogger()
