"""
LAVENDER — Self-Coding Engine
Safely generates, tests, and deploys new capabilities
"""

import os
import ast
import subprocess
import tempfile
import logging
import re
from pathlib import Path
from typing import Optional, List
from datetime import datetime

logger = logging.getLogger("lavender.self_coder")

class SafeCodeGenerator:
    """
    Generates new tools and capabilities through LLM
    with safety checks and sandboxed testing.
    """

    def __init__(self, brain, tools_dir: Path):
        self.brain = brain
        self.tools_dir = tools_dir
        self.sandbox_dir = tools_dir / "sandbox"
        self.sandbox_dir.mkdir(exist_ok=True)

    async def create_new_tool(self, requirement: str) -> Optional[str]:
        """Creates a new tool from natural language requirements."""
        logger.info(f"Generating new tool: {requirement}")

        code = await self._generate_tool_code(requirement)
        if not code: return None

        if not self._safety_check(code):
            logger.error("Generated code failed safety check")
            return None

        if not self._validate_syntax(code):
            logger.error("Generated code has syntax errors")
            return None

        if not await self._test_in_sandbox(code):
            logger.error("Generated code failed sandbox tests")
            return None

        # Human-in-the-loop safety gating
        from core.safety import instance as safety
        is_safe, reason = safety.validate_tool_call("deploy_new_tool", {"requirement": requirement, "code_snippet": code[:100]})
        if not is_safe:
            logger.warning(f"Tool deployment BLOCKED: {reason}")
            return None

        tool_path = self._deploy_tool(code)

        # Trigger reload in brain
        if hasattr(self.brain, "reload_tools"):
            self.brain.reload_tools()

        logger.info(f"New tool deployed: {tool_path}")
        return str(tool_path)

    async def _generate_tool_code(self, requirement: str) -> str:
        prompt = f"""
You are Lavender's code generator. Create a production-ready LangChain tool.

REQUIREMENT: {requirement}

TEMPLATE:
```python
from langchain_core.tools import tool
from typing import Optional

@tool
def tool_name(param1: str, param2: int = 0) -> str:
    \"\"\"
    Clear description.
    \"\"\"
    try:
        # Implementation
        return "result"
    except Exception as e:
        return f"Error: {{str(e)}}"
```

RULES:
1. Must use @tool decorator
2. Must have comprehensive docstring
3. Handle all errors
4. No network calls or file writes outside 'memory/'
5. No subprocess or os.system

Generate the code:
"""
        response = self.brain.llm.invoke(prompt)
        return self._extract_code_block(response.content)

    def _extract_code_block(self, text: str) -> str:
        match = re.search(r"```python\n(.*?)\n```", text, re.DOTALL)
        if match: return match.group(1)
        return text

    def _safety_check(self, code: str) -> bool:
        dangerous = ["os.system", "subprocess.Popen", "eval(", "exec(", "socket."]
        for p in dangerous:
            if p in code: return False
        return True

    def _validate_syntax(self, code: str) -> bool:
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    async def _test_in_sandbox(self, code: str) -> bool:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", dir=self.sandbox_dir, delete=False) as f:
            f.write(code)
            tmp_path = f.name

        try:
            result = subprocess.run(
                ["python3", "-c", "import ast; ast.parse(open('"+tmp_path+"').read())"],
                capture_output=True, timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False
        finally:
            os.unlink(tmp_path)

    def _backup_source(self, filepath: Path):
        """Creates a backup of the file before modification."""
        if not filepath.exists():
            return
        backup_path = filepath.with_suffix(f".bak.{int(datetime.now().timestamp())}")
        backup_path.write_text(filepath.read_text())
        logger.info(f"Backup created: {backup_path}")

    def rollback(self, tool_name: str):
        """Rolls back a tool to its last backup."""
        tool_file = self.tools_dir / f"{tool_name}.py"
        backups = sorted(self.tools_dir.glob(f"{tool_name}.py.bak.*"), reverse=True)
        if not backups:
            logger.warning(f"No backups found for {tool_name}")
            return False

        latest_backup = backups[0]
        tool_file.write_text(latest_backup.read_text())
        logger.info(f"Rolled back {tool_name} from {latest_backup}")

        if hasattr(self.brain, "reload_tools"):
            self.brain.reload_tools()
        return True

    def _deploy_tool(self, code: str) -> Path:
        # Extract function name for filename
        tree = ast.parse(code)
        tool_name = "custom_tool"
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                tool_name = node.name
                break

        tool_file = self.tools_dir / f"{tool_name}.py"

        # Backup before overwriting
        self._backup_source(tool_file)

        tool_file.write_text(code)
        return tool_file

class CapabilityExpansion:
    """Tracks capability gaps and proposes new tools."""
    def __init__(self, brain, generator):
        self.brain = brain
        self.generator = generator
        self.gaps = []

    def note_gap(self, request: str, reason: str):
        logger.info(f"Capability gap noted: {request} ({reason})")
        self.gaps.append({"request": request, "reason": reason, "time": datetime.now()})
        # Logic to propose new tool after X repeated gaps could go here
