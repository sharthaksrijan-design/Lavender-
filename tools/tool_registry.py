"""
LAVENDER — Tool Registry
tools/tool_registry.py

Defines all tools as LangGraph-compatible callables.
The brain imports build_toolkit() and passes the result to the agent.

Each tool is:
  - A Python function with a clear docstring (the LLM reads this)
  - Decorated or wrapped as a LangGraph Tool
  - Connected to the underlying implementation (HomeControl, CodeRunner, etc.)

Adding a new tool:
  1. Implement the underlying function in the appropriate tools/ file
  2. Write a wrapper function here with a clear docstring
  3. Add it to build_toolkit()
  4. Add its intent category to brain.py's router prompt if needed
"""

import json
import logging
import os
from typing import Optional
from langchain_core.tools import tool

logger = logging.getLogger("lavender.tools")


def build_toolkit(
    ha_url: str = None,
    ha_token: str = None,
    enable_code_runner: bool = True,
    enable_web: bool = True,
    enable_home: bool = True,
):
    """
    Build and return the full list of LangGraph tools.
    Only includes tools whose dependencies are configured.

    Returns list of tool functions ready to pass to LangGraph agent.
    """
    tools = []

    # ── HOME CONTROL ──────────────────────────────────────────────────────────
    if enable_home and ha_url and ha_token:
        try:
            from tools.home_control import HomeControl
            _hc = HomeControl(ha_url=ha_url, token=ha_token)

            # Pre-load entity cache
            if _hc.is_available():
                _hc.get_all_states()
                logger.info("Home Assistant connected. Entity cache loaded.")
            else:
                logger.warning("Home Assistant not reachable — home tools disabled.")
                _hc = None

            if _hc:
                @tool
                def control_device(command_json: str) -> str:
                    """
                    Control a smart home device via Home Assistant.

                    Use this tool for commands like:
                    - turning lights on or off
                    - setting light brightness or color
                    - adjusting AC temperature
                    - activating scenes
                    - controlling media players
                    - toggling switches

                    Pass a JSON object as a string with these fields:
                    {
                      "action": "turn on" | "turn off" | "toggle" |
                                "set brightness" | "set temperature" |
                                "set volume" | "activate scene" | "status",
                      "entity": "<friendly name or entity_id>",
                      "value": <number if needed, else omit>,
                      "domain": "<optional: light, switch, climate, media_player>"
                    }

                    Examples:
                    {"action": "turn on", "entity": "desk lamp"}
                    {"action": "set brightness", "entity": "light.office", "value": 40}
                    {"action": "set temperature", "entity": "office AC", "value": 22}
                    {"action": "activate scene", "entity": "evening"}
                    {"action": "status", "entity": "office AC"}
                    """
                    return _hc.execute_natural_command(command_json)

                @tool
                def list_devices(domain: str = "") -> str:
                    """
                    List available smart home devices, optionally filtered by type.
                    domain can be: light, switch, climate, media_player, scene, sensor
                    Leave domain empty to list all devices.

                    Use this when the user asks "what devices do I have"
                    or before controlling a device if you're unsure of its name.
                    """
                    if domain:
                        entities = _hc.get_entities_by_domain(domain)
                    else:
                        entities = _hc.get_all_states()

                    if not entities:
                        return f"No {'devices of type ' + domain if domain else 'devices'} found."

                    lines = []
                    for e in entities[:30]:  # Cap at 30 for readability
                        eid  = e["entity_id"]
                        name = e.get("attributes", {}).get("friendly_name", eid)
                        state = e.get("state", "unknown")
                        lines.append(f"  {name} ({eid}): {state}")

                    return "\n".join(lines)

                tools.extend([control_device, list_devices])
                logger.info("Home control tools registered.")

        except ImportError as e:
            logger.warning(f"Home control tools unavailable: {e}")

    # ── CODE RUNNER ───────────────────────────────────────────────────────────
    if enable_code_runner:
        try:
            from tools.code_runner import CodeRunner
            _runner = CodeRunner(timeout_seconds=15.0)

            @tool
            def run_python(code: str) -> str:
                """
                Execute Python code and return the output.

                Use this tool when:
                - The user asks you to run or test code
                - A calculation is complex enough to need actual execution
                - Data analysis or manipulation is needed
                - You want to verify a computation result

                The code runs in a sandbox: file writing and network access
                are blocked. Standard library and numpy/pandas are available.

                Pass complete, runnable Python code as a string.
                The last expression's value is automatically captured.

                Example:
                  run_python("import math\nmath.factorial(20)")
                """
                result = _runner.run(code)
                return result.format_for_response()

            tools.append(run_python)
            logger.info("Code runner tool registered.")

        except ImportError as e:
            logger.warning(f"Code runner unavailable: {e}")

    # ── WEB TOOLS ─────────────────────────────────────────────────────────────
    if enable_web:
        try:
            from tools.web_tools import WebTools
            _web = WebTools()

            @tool
            def search_web(query: str) -> str:
                """
                Search the web for current information.

                Use this tool for:
                - Current news or events
                - Real-time facts (stock prices, sports scores, etc.)
                - Information that may have changed recently
                - Anything where your training data might be outdated

                Do NOT use for: things you already know confidently,
                or questions about the user's personal data (use memory instead).

                Returns a summary of top search results.
                """
                return _web.search(query)

            @tool
            def fetch_page(url: str) -> str:
                """
                Fetch the content of a specific web page.

                Use this tool when:
                - The user shares a URL and asks about its content
                - A search result gives a URL worth reading in full
                - Documentation or reference material needs to be retrieved

                Returns the page's text content, stripped of HTML.
                """
                return _web.fetch(url)

            @tool
            def get_weather(location: str) -> str:
                """
                Get current weather and a 3-day forecast for a location.

                Use this for any weather-related question.
                Location can be a city name, city + country, or region.

                Examples: "Mumbai", "Deoghar Jharkhand", "London UK"
                """
                return _web.get_weather(location)

            tools.extend([search_web, fetch_page, get_weather])
            logger.info("Web tools registered.")

        except ImportError as e:
            logger.warning(f"Web tools unavailable: {e}")

    # ── MEMORY TOOLS ─────────────────────────────────────────────────────────
    # These are registered here but connected to LavenderMemory in brain.py
    # via a closure. Placeholder definitions shown for documentation.

    # ── VISION ────────────────────────────────────────────────────────────────
    if enable_web:  # reuse the enable_web flag — vision also needs Ollama
        try:
            from tools.vision import Vision, make_vision_tools
            _vision = Vision(
                model="llava:13b",
                ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            )
            if _vision.is_available():
                vision_tools = make_vision_tools(_vision)
                tools.extend(vision_tools)
                logger.info("Vision tools registered (LLaVA available).")
            else:
                logger.info("Vision tools skipped (llava:13b not loaded — run: ollama pull llava:13b).")
        except ImportError as e:
            logger.warning(f"Vision tools unavailable: {e}")

    logger.info(f"Toolkit built: {len(tools)} tools registered.")
    return tools


def describe_toolkit(tools: list) -> str:
    """
    Return a compact description of available tools for injection into
    the system prompt when the agent needs to know what it can do.
    """
    if not tools:
        return "No tools available in this session."

    lines = ["Available tools:"]
    for t in tools:
        first_line = (t.__doc__ or "").strip().split("\n")[0]
        lines.append(f"  - {t.name}: {first_line}")

    return "\n".join(lines)
