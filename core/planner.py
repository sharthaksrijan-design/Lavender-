"""
LAVENDER — Task Planner
core/planner.py

Decomposes complex, multi-step goals into a sequence or graph of executable tasks.
Each task is mapped to a tool or a sub-reasoning step.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

logger = logging.getLogger("lavender.planner")

@dataclass
class Task:
    id: str
    description: str
    tool: Optional[str] = None
    args: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"  # pending, in_progress, completed, failed
    result: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)

class GoalPlan:
    def __init__(self, goal: str):
        self.goal = goal
        self.tasks: Dict[str, Task] = {}
        self.status = "planned"

    def add_task(self, task: Task):
        self.tasks[task.id] = task

    def is_complete(self) -> bool:
        return all(t.status == "completed" for t in self.tasks.values())

PLANNER_PROMPT = """You are the Planning Module for Lavender, an advanced AI system.
Your job is to take a complex goal and break it into a sequence of specific tasks.

Available Tools:
{tools_description}

Return ONLY valid JSON.
Format:
{{
  "plan_id": "unique_string",
  "tasks": [
    {{
      "id": "t1",
      "description": "task description",
      "tool": "tool_name or null",
      "args": {{ "arg1": "val1" }},
      "dependencies": []
    }}
  ]
}}

GOAL: "{goal}"
"""

class Planner:
    def __init__(self, model: str = "llama3.1:8b-instruct-q4_K_M", ollama_base_url: str = "http://localhost:11434"):
        self._model = model
        self._base_url = ollama_base_url
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            self._llm = ChatOllama(
                model=self._model,
                base_url=self._base_url,
                temperature=0.1, # Planners need to be deterministic
            )
        return self._llm

    def generate_plan(self, goal: str, tools_description: str) -> Optional[GoalPlan]:
        logger.info(f"Generating plan for goal: {goal}")
        prompt = PLANNER_PROMPT.format(goal=goal, tools_description=tools_description)

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content.strip()

            # Basic JSON extraction
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()

            data = json.loads(content)
            plan = GoalPlan(goal)
            for t_data in data.get("tasks", []):
                task = Task(
                    id=t_data["id"],
                    description=t_data["description"],
                    tool=t_data.get("tool"),
                    args=t_data.get("args", {}),
                    dependencies=t_data.get("dependencies", [])
                )
                plan.add_task(task)

            logger.info(f"Plan generated with {len(plan.tasks)} tasks.")
            return plan
        except Exception as e:
            logger.error(f"Failed to generate plan: {e}")
            return None

# Lazy instance - don't initialize LLM on import
instance = Planner()
