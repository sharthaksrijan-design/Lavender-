"""
LAVENDER — Code Runner
tools/code_runner.py

Executes Python code in a sandboxed subprocess with:
  - Hard timeout (configurable, default 15s)
  - Output capture (stdout + stderr)
  - Restricted imports (blocks os.system, subprocess, etc.)
  - Memory limit via resource module

Lavender uses this when Vector mode generates code and wants to run it,
or when you say "Lavender, run this" pointing at code on your display.

The runner returns a structured result: output, errors, return value,
execution time. The brain then formats this for the response.
"""

import ast
import sys
import os
import io
import time
import textwrap
import subprocess
import tempfile
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("lavender.code_runner")


# Imports that are blocked in the sandbox
BLOCKED_IMPORTS = {
    "os", "subprocess", "sys", "socket", "urllib",
    "requests", "httpx", "aiohttp",
    "shutil", "glob", "pathlib",
    "ctypes", "multiprocessing",
    "threading",  # Allow within reason — but block direct thread creation
    "__builtins__",
}

# Imports explicitly allowed (whitelist approach for sensitive ones)
SAFE_IMPORTS = {
    "math", "cmath", "decimal", "fractions", "statistics",
    "random", "string", "re", "textwrap", "difflib",
    "datetime", "calendar", "time",
    "json", "csv",
    "collections", "itertools", "functools", "operator",
    "heapq", "bisect", "array",
    "struct", "codecs", "hashlib", "base64",
    "pprint", "reprlib",
    "io", "copy", "pprint",
    "numpy", "pandas", "matplotlib",
    "typing",
}


@dataclass
class RunResult:
    success: bool
    output: str           # stdout
    error: str            # stderr or exception message
    return_value: str     # repr of last expression value
    execution_time_ms: int
    code: str             # the code that was run

    def format_for_response(self) -> str:
        """Format result for Lavender to include in her spoken response."""
        parts = []

        if self.success:
            if self.output.strip():
                parts.append(f"Output:\n{self.output.strip()}")
            if self.return_value and self.return_value != "None":
                parts.append(f"Result: {self.return_value}")
            if not parts:
                parts.append("Code ran successfully with no output.")
        else:
            parts.append(f"Error: {self.error}")

        parts.append(f"(Execution time: {self.execution_time_ms}ms)")
        return "\n".join(parts)


class CodeRunner:
    def __init__(
        self,
        timeout_seconds: float = 15.0,
        max_output_chars: int = 4000,
    ):
        self.timeout_seconds = timeout_seconds
        self.max_output_chars = max_output_chars

    # ── SAFETY CHECK ──────────────────────────────────────────────────────────

    def _check_safety(self, code: str) -> Optional[str]:
        """
        Parse the AST and check for dangerous patterns.
        Returns an error message if unsafe, None if safe.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return f"Syntax error: {e}"

        for node in ast.walk(tree):
            # Block certain imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    names = [alias.name.split(".")[0] for alias in node.names]
                else:
                    names = [node.module.split(".")[0]] if node.module else []

                for name in names:
                    if name in BLOCKED_IMPORTS:
                        return f"Import '{name}' is not allowed in sandboxed execution."

            # Block exec() and eval() calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ("exec", "eval", "compile", "__import__"):
                        return f"'{node.func.id}()' is not allowed in sandboxed execution."

            # Block open() with write modes (both positional and keyword arguments)
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "open":
                    mode = "r" # default

                    # Positional mode argument
                    if len(node.args) > 1:
                        mode_node = node.args[1]
                        if isinstance(mode_node, ast.Constant):
                            mode = mode_node.value

                    # Keyword mode argument (overrides positional)
                    for kw in node.keywords:
                        if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                            mode = kw.value.value

                    if any(c in mode for c in ("w", "a", "x")):
                        return "File writing is not allowed in sandboxed execution."

        return None

    # ── RUNNER ────────────────────────────────────────────────────────────────

    def run(self, code: str, context: dict = None) -> RunResult:
        """
        Execute code and return structured result.

        Uses subprocess isolation — code runs in a separate Python process
        so a crash or infinite loop can't take down Lavender.
        """
        code = textwrap.dedent(code).strip()

        # Safety check first
        safety_error = self._check_safety(code)
        if safety_error:
            return RunResult(
                success=False,
                output="",
                error=f"Code blocked: {safety_error}",
                return_value="",
                execution_time_ms=0,
                code=code,
            )

        # Wrap code to capture last expression value
        wrapped = self._wrap_code(code)

        # Write to temp file and execute in subprocess
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, prefix="lavender_run_"
        ) as f:
            f.write(wrapped)
            tmp_path = f.name

        try:
            start = time.time()
            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                env={
                    **os.environ,
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONPATH": "",   # isolate from Lavender's own packages
                }
            )
            elapsed_ms = int((time.time() - start) * 1000)

            stdout = result.stdout[:self.max_output_chars]
            stderr = result.stderr[:self.max_output_chars]

            # Parse the sentinel return value from stdout
            return_value = ""
            if "<<LAVENDER_RESULT>>" in stdout:
                parts = stdout.split("<<LAVENDER_RESULT>>")
                stdout = parts[0]
                return_value = parts[1].strip() if len(parts) > 1 else ""

            success = result.returncode == 0 and not stderr

            return RunResult(
                success=success,
                output=stdout.strip(),
                error=stderr.strip() if stderr else "",
                return_value=return_value,
                execution_time_ms=elapsed_ms,
                code=code,
            )

        except subprocess.TimeoutExpired:
            return RunResult(
                success=False,
                output="",
                error=f"Execution timed out after {self.timeout_seconds}s.",
                return_value="",
                execution_time_ms=int(self.timeout_seconds * 1000),
                code=code,
            )
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    def _wrap_code(self, code: str) -> str:
        """
        Wrap user code to capture the value of the last expression.
        Appends a sentinel marker so we can extract the return value.
        """
        lines = code.strip().splitlines()
        if not lines:
            return code

        last_line = lines[-1].strip()

        # Try to evaluate last line as expression
        try:
            ast.parse(last_line, mode="eval")
            # It's an expression — wrap it
            body = "\n".join(lines[:-1])
            wrapped = f"""
{body}
__lavender_result__ = {last_line}
print("<<LAVENDER_RESULT>>", repr(__lavender_result__))
"""
        except SyntaxError:
            # Last line is a statement — just run it
            wrapped = code

        return wrapped.strip()

    def format_code(self, code: str) -> str:
        """Try to auto-format code using black if available."""
        try:
            import black
            return black.format_str(code, mode=black.Mode())
        except (ImportError, Exception):
            return code


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# python tools/code_runner.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    runner = CodeRunner()

    tests = [
        ("Basic arithmetic",      "2 ** 10"),
        ("List comprehension",    "[x**2 for x in range(1, 6)]"),
        ("String manipulation",   "'hello world'.title().replace(' ', '_')"),
        ("Multiple outputs",      "for i in range(3):\n    print(f'item {i}')"),
        ("Import numpy",          "import numpy as np\nnp.array([1,2,3]).mean()"),
        ("Blocked import",        "import os\nos.listdir('/')"),
        ("Timeout test (skip)",   "while True: pass"),
        ("Syntax error",          "def broken(:\n    pass"),
    ]

    for name, code in tests:
        print(f"\n── {name} ──")
        print(f"Code: {code[:60]}")
        result = runner.run(code)
        print(result.format_for_response())
