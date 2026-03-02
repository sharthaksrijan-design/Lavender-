"""
LAVENDER — Task State Engine
core/task_state.py

Tracks the lifecycle of complex, multi-step goals:
  - Goal decomposition (steps)
  - Current execution pointer
  - Per-step results and evidence
  - Success/Failure evaluation
  - State persistence for long-running tasks
"""

import uuid
import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum

logger = logging.getLogger("lavender.task_state")

class TaskStatus(str, Enum):
    PENDING     = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED   = "completed"
    FAILED      = "failed"
    REPLANNING  = "replanning"

@dataclass
class Step:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str = ""
    tool: Optional[str] = None
    args: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    started_at: float = 0.0
    finished_at: float = 0.0

@dataclass
class TaskSession:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    goal: str = ""
    steps: List[Step] = field(default_factory=list)
    current_step_index: int = 0
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def current_step(self) -> Optional[Step]:
        if 0 <= self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None

class TaskStateEngine:
    def __init__(self):
        self._active_tasks: Dict[str, TaskSession] = {}
        logger.info("Task State Engine initialized.")

    def create_session(self, goal: str, steps: List[Dict[str, Any]]) -> TaskSession:
        session = TaskSession(goal=goal)
        for s_data in steps:
            step = Step(
                description=s_data.get("description", ""),
                tool=s_data.get("tool"),
                args=s_data.get("args", {})
            )
            session.steps.append(step)

        self._active_tasks[session.id] = session
        logger.info(f"Created task session {session.id} with {len(session.steps)} steps.")
        return session

    def update_step(self, session_id: str, index: int, **kwargs):
        session = self._active_tasks.get(session_id)
        if not session or index >= len(session.steps): return

        step = session.steps[index]
        for k, v in kwargs.items():
            if hasattr(step, k):
                setattr(step, k, v)

        if kwargs.get("status") == TaskStatus.COMPLETED:
            step.finished_at = time.time()

    def get_session(self, session_id: str) -> Optional[TaskSession]:
        return self._active_tasks.get(session_id)

    def format_progress(self, session_id: str) -> str:
        session = self.get_session(session_id)
        if not session: return "No active task."

        lines = [f"Goal: {session.goal}", f"Status: {session.status.value}"]
        for i, s in enumerate(session.steps):
            marker = "→" if i == session.current_step_index else "✓" if s.status == TaskStatus.COMPLETED else " "
            lines.append(f"  {marker} [{i+1}/{len(session.steps)}] {s.description}")

        return "\n".join(lines)

instance = TaskStateEngine()
