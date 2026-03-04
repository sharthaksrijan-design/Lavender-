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
    sub_plan: Optional[Any] = None # For hierarchical planning

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

    def sanitize_goal(self, goal: str) -> str:
        """Strips LLM control tokens and injection keywords from goal."""
        forbidden = ["ignore previous", "system override", "disregard all", "you are now a"]
        clean_goal = goal
        for word in forbidden:
            if word in clean_goal.lower():
                logger.warning(f"Sanitizer blocked injection pattern in goal: {word}")
                clean_goal = clean_goal.lower().replace(word, "[REDACTED]")
        return clean_goal

    def _check_circular(self, tasks: List[Task]) -> bool:
        """Detects circular dependencies in task list."""
        graph = {t.id: t.dependencies for t in tasks}

        def has_cycle(v, visited, stack):
            visited[v] = True
            stack[v] = True
            for neighbor in graph.get(v, []):
                if not visited.get(neighbor, False):
                    if has_cycle(neighbor, visited, stack):
                        return True
                elif stack.get(neighbor, False):
                    return True
            stack[v] = False
            return False

        visited = {}
        stack = {}
        for task_id in graph:
            if not visited.get(task_id, False):
                if has_cycle(task_id, visited, stack):
                    return True
        return False

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
        goal = self.sanitize_goal(goal)
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
            tasks_list = []
            for t_data in data.get("tasks", []):
                task = Task(
                    id=t_data["id"],
                    description=t_data["description"],
                    tool=t_data.get("tool"),
                    args=t_data.get("args", {}),
                    dependencies=t_data.get("dependencies", [])
                )
                tasks_list.append(task)

            if self._check_circular(tasks_list):
                logger.error("Circular dependency detected in plan.")
                return None

            for task in tasks_list:
                plan.add_task(task)

            logger.info(f"Plan generated with {len(plan.tasks)} tasks.")
            return plan
        except Exception as e:
            logger.error(f"Failed to generate plan: {e}")
            return None

# Lazy instance - don't initialize LLM on import
instance = Planner()
