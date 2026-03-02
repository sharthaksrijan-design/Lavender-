"""
LAVENDER — System Tools
tools/system_tools.py

Provides tools for direct interaction with the host system:
  - File operations (list, read, write)
  - Application management (open, close)
  - System status (uptime, resources)
"""

import os
import subprocess
import logging
from typing import List, Optional
from pathlib import Path

logger = logging.getLogger("lavender.system_tools")

def list_files(path: str = ".") -> str:
    """Lists files in a directory."""
    try:
        files = os.listdir(path)
        return "\n".join(files)
    except Exception as e:
        return f"Error: {e}"

def read_file(filepath: str) -> str:
    """Reads the content of a file."""
    try:
        return Path(filepath).read_text()
    except Exception as e:
        return f"Error: {e}"

def write_file(filepath: str, content: str) -> str:
    """Writes content to a file."""
    try:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return f"Successfully wrote to {filepath}"
    except Exception as e:
        return f"Error: {e}"

def open_app(app_name: str) -> str:
    """Opens a system application."""
    try:
        # Platform specific - assuming Linux/Ubuntu as per README
        subprocess.Popen([app_name], start_new_session=True)
        return f"Opening {app_name}..."
    except Exception as e:
        return f"Error: {e}"

def get_system_summary() -> str:
    """Returns a brief system status summary."""
    try:
        uptime = subprocess.check_output(["uptime", "-p"]).decode().strip()
        return f"System is up. {uptime}"
    except Exception as e:
        return f"Error: {e}"

def make_system_tools() -> List:
    from langchain_core.tools import tool

    @tool
    def file_ops(action: str, path: str, content: Optional[str] = None) -> str:
        """
        Perform file operations.
        action: 'list', 'read', 'write'
        path: directory or file path
        content: required for 'write'
        """
        if action == "list":
            return list_files(path)
        elif action == "read":
            return read_file(path)
        elif action == "write":
            if content is None: return "Error: content required for write."
            return write_file(path, content)
        return "Error: invalid action."

    @tool
    def launch_app(name: str) -> str:
        """Launch a system application by name."""
        return open_app(name)

    @tool
    def system_status() -> str:
        """Get host system status."""
        return get_system_summary()

    return [file_ops, launch_app, system_status]
