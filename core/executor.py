"""
LAVENDER — Autonomous Task Executor
Handles long-running, multi-step operations in the background.
"""

import asyncio
import logging
import uuid
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Callable, Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger("lavender.executor")


class TaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass
class TaskStep:
    """Single step in a task plan"""
    action: str
    tool: str
    params: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    timeout: int = 300
    retry_count: int = 3
    status: TaskStatus = TaskStatus.QUEUED
    result: Optional[Any] = None
    error: Optional[str] = None


@dataclass
class AutonomousTask:
    """Autonomous task with execution plan"""
    task_id: str
    description: str
    steps: List[TaskStep]
    status: TaskStatus
    priority: TaskPriority
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[dict] = None
    error: Optional[str] = None
    progress: float = 0.0
    on_complete: Optional[Callable] = None
    on_error: Optional[Callable] = None
    checkpoint_frequency: int = 5  # Save state every N steps


class TaskExecutor:
    """
    Autonomous task execution engine.
    Handles background tasks, checkpointing, recovery.
    """

    def __init__(self, brain, memory, max_concurrent=3):
        self.brain = brain
        self.memory = memory
        self.max_concurrent = max_concurrent
        self.tasks: Dict[str, AutonomousTask] = {}
        self.queue = asyncio.PriorityQueue()
        self.running = False
        self._loop_task: Optional[asyncio.Task] = None

    def start(self):
        """Start the execution loop in a background thread with its own loop"""
        if not self.running:
            self.running = True
            import threading
            def run_loop():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self._loop_task = loop.create_task(self.execute_loop())
                loop.run_forever()

            self._worker_thread = threading.Thread(target=run_loop, daemon=True)
            self._worker_thread.start()
            logger.info("Task Executor started in background thread.")

    async def stop(self):
        """Stop the execution loop"""
        self.running = False
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        logger.info("Task Executor stopped.")

    async def submit(self, task: AutonomousTask) -> str:
        """Submit task for execution"""
        self.tasks[task.task_id] = task
        # PriorityQueue stores (priority_value, item)
        await self.queue.put((task.priority.value, task))

        # Store in memory for recovery (assuming LavenderMemory has this method)
        if hasattr(self.memory, 'store_autonomous_task'):
            self.memory.store_autonomous_task(task)

        logger.info(f"Task {task.task_id} submitted: {task.description}")
        return task.task_id

    async def execute_loop(self):
        """Main execution loop"""
        workers = []
        for i in range(self.max_concurrent):
            worker = asyncio.create_task(self._worker(i))
            workers.append(worker)

        await asyncio.gather(*workers)

    async def _worker(self, worker_id: int):
        """Worker coroutine"""
        while self.running:
            try:
                # Get next task from priority queue
                _, task = await asyncio.wait_for(
                    self.queue.get(),
                    timeout=1.0
                )

                logger.info(f"Worker {worker_id} starting task {task.task_id}")
                await self._execute_task(task)
                self.queue.task_done()

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")

    async def _execute_task(self, task: AutonomousTask):
        """Execute a single task with multiple steps"""
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()

        try:
            for i, step in enumerate(task.steps):
                if task.status == TaskStatus.CANCELLED:
                    break

                # Check for PAUSED state
                while task.status == TaskStatus.PAUSED:
                    await asyncio.sleep(1)

                logger.info(f"Task {task.task_id} - Step {i+1}/{len(task.steps)}: {step.action}")
                step.status = TaskStatus.RUNNING

                # Execute step with retry logic
                result = await self._execute_step(step)
                step.result = result
                step.status = TaskStatus.COMPLETED

                # Update task progress
                task.progress = (i + 1) / len(task.steps)

                # Checkpoint
                if (i + 1) % task.checkpoint_frequency == 0:
                    if hasattr(self.memory, 'checkpoint_task'):
                        self.memory.checkpoint_task(task)

            if task.status != TaskStatus.CANCELLED:
                # Task complete
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now()

                if task.on_complete:
                    if asyncio.iscoroutinefunction(task.on_complete):
                        await task.on_complete(task)
                    else:
                        task.on_complete(task)

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            logger.error(f"Task {task.task_id} failed: {e}")

            if task.on_error:
                if asyncio.iscoroutinefunction(task.on_error):
                    await task.on_error(task, e)
                else:
                    task.on_error(task, e)

        finally:
            if hasattr(self.memory, 'store_autonomous_task'):
                self.memory.store_autonomous_task(task)

    async def _execute_step(self, step: TaskStep):
        """Execute a single step with retry and timeout"""
        last_error = None
        for attempt in range(step.retry_count):
            try:
                # In our architecture, the brain handles tool execution via LangGraph or direct calls
                # Here we assume a direct call for simplicity or a specialized brain method
                result = await self.brain.execute_tool(step.tool, step.params, timeout=step.timeout)
                return result

            except asyncio.TimeoutError:
                logger.warning(f"Step timeout (attempt {attempt+1}/{step.retry_count})")
                last_error = "TimeoutExpired"

            except Exception as e:
                logger.error(f"Step error (attempt {attempt+1}/{step.retry_count}): {e}")
                last_error = str(e)
                if attempt < step.retry_count - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff

        raise Exception(f"Step failed after {step.retry_count} attempts. Last error: {last_error}")

    def pause_task(self, task_id: str):
        """Pause a running task"""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            if task.status == TaskStatus.RUNNING:
                task.status = TaskStatus.PAUSED
                logger.info(f"Task {task_id} paused.")

    def resume_task(self, task_id: str):
        """Resume a paused task"""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            if task.status == TaskStatus.PAUSED:
                task.status = TaskStatus.RUNNING
                logger.info(f"Task {task_id} resumed.")

    def cancel_task(self, task_id: str):
        """Cancel a task"""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task.status = TaskStatus.CANCELLED
            logger.info(f"Task {task_id} cancelled.")

    def get_task_status(self, task_id: str) -> Optional[AutonomousTask]:
        return self.tasks.get(task_id)
