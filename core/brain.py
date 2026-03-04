"""
LAVENDER — Intelligence Brain
core/brain.py

Handles:
  - Intent routing (fast Mistral 7B classifier)
  - LLM reasoning (LLaMA 3.1 70B)
  - Personality management and switching
  - Session context and working memory
  - Lilac's special behavioral logic
"""

import os
import json
import random
import time
import logging
import threading
from pathlib import Path
from typing import Optional, Any
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.prebuilt import create_react_agent

from core.personality import (
    PersonalityConfig,
    get_personality,
    resolve_personality_from_text,
    LILAC_ROAST_TIERS,
    LILAC_REJECTION_TEMPLATES,
    LILAC_SWITCH_RESPONSES,
    DEFAULT_PERSONALITY,
)
from core.memory import LavenderMemory
from core.summarizer import SessionSummarizer
from core.state import instance as state_engine
from core.planner import instance as planner_engine, Planner
from core.task_state import instance as task_engine, TaskStatus, Step
from core.safety import instance as safety_layer
from core.self_coder import SafeCodeGenerator, CapabilityExpansion
from core.agent_factory import AgentFactory
from tools.tool_registry import describe_toolkit, build_toolkit

logger = logging.getLogger("lavender.brain")

# Intent categories that route through the tool agent
TOOL_INTENTS = {
    "informational_realtime",
    "operational_device",
    "computational",
    "perceptual",
    "operational_async",   # Background tasks
}


# ── INTENT CATEGORIES ────────────────────────────────────────────────────────

class Intent:
    CONVERSATIONAL         = "conversational"
    INFORMATIONAL_REALTIME = "informational_realtime"
    INFORMATIONAL_STATIC   = "informational_static"
    INFORMATIONAL_PERSONAL = "informational_personal"
    OPERATIONAL_DEVICE     = "operational_device"
    OPERATIONAL_CALENDAR   = "operational_calendar"
    COMPUTATIONAL          = "computational"
    PERCEPTUAL             = "perceptual"
    SYSTEM_PERSONALITY     = "system_personality_switch"
    SYSTEM_MEMORY          = "system_memory"
    SYSTEM_OTHER           = "system_other"


ROUTER_PROMPT = """You are an intent classifier for an AI assistant called Lavender.
Classify the user's input into EXACTLY ONE intent category.
Return ONLY valid JSON. No explanation. No markdown.

CATEGORIES:
- conversational: general chat, reasoning, opinions, creative tasks
- informational_realtime: weather, news, stock prices, live data
- informational_static: factual knowledge questions
- informational_personal: user's own data, projects, preferences, history
- operational_device: control lights, music, AC, home devices
- operational_calendar: meetings, reminders, scheduling, time
- computational: code generation, execution, math, data analysis
- perceptual: analyze image, read document, describe what is visible
- operational_async: long running tasks, background jobs, monitoring
- system_personality_switch: user wants to change personality (iris/nova/vector/solace/lilac)
- system_memory: user asks about or wants to modify Lavender's memory
- system_other: settings, configuration, meta questions about Lavender

INPUT: "{text}"

Return JSON:
{{"intent": "<category>", "confidence": <0.0-1.0>, "target": "<extracted value if relevant, else null>"}}"""


# ── LILAC QUALITY ASSESSOR ───────────────────────────────────────────────────

LILAC_QUALITY_PROMPT = """You are assessing the quality of a question asked to an AI called Lavender.
Lavender is currently operating as LILAC — an AI with very high standards and zero patience for lazy or vague inputs.

Assess this question: "{text}"

Return ONLY valid JSON:
{{
  "quality": "worthy" | "poor" | "reject",
  "reason": "<one sentence explaining why, used to craft the roast/rejection>"
}}

CRITERIA:
- "worthy": specific, shows thought, has a clear answerable goal
- "poor": vague, lazy, or could trivially be googled, but has some addressable content
- "reject": completely underspecified, missing key information, not answerable in any useful form"""


class LavenderBrain:
    def __init__(
        self,
        personality: str = DEFAULT_PERSONALITY,
        primary_model: str = "llama3.1:70b-instruct-q4_K_M",
        router_model: str = "mistral:7b-instruct-q4_K_M",
        ollama_base_url: str = "http://localhost:11434",
        max_working_memory: int = 20,
        memory: Optional[LavenderMemory] = None,
        tools: list = None,
        top_k_memories: int = 3,
        intent_threshold: float = 0.75,
        personality_overrides: dict = None,
    ):
        self.planner_engine = Planner(model=primary_model, ollama_base_url=ollama_base_url)
        self.current_personality: PersonalityConfig = get_personality(personality)
        self.max_working_memory = max_working_memory

        # Session state
        self._session_history: list[dict] = []
        self._session_start = time.time()

        # Lilac-specific session state
        self._lilac_poor_question_count: int = 0
        self._lilac_exits_this_session: int = 0
        self._previous_personality: Optional[str] = None

        # Memory (optional — Milestone 2)
        self.memory: Optional[LavenderMemory] = memory
        self.top_k_memories = top_k_memories
        self.intent_threshold = intent_threshold
        self.personality_overrides = personality_overrides or {}
        self._summarizer: Optional[SessionSummarizer] = (
            SessionSummarizer(model=primary_model, ollama_base_url=ollama_base_url)
            if memory else None
        )

        # Tools (optional — Milestone 4)
        self._tools: list = tools or []
        self._ollama_base_url = ollama_base_url

        # v2.0 Extensions
        self.code_gen = SafeCodeGenerator(self, Path(__file__).parent.parent / "tools")
        self.capability = CapabilityExpansion(self, self.code_gen)
        self.agent_factory = AgentFactory(self)

        logger.info(f"Initializing LLM clients (Ollama at {ollama_base_url})...")

        self.llm = ChatOllama(
            model=primary_model,
            base_url=ollama_base_url,
            temperature=0.7,
            num_ctx=8192,
        )

        self.router = ChatOllama(
            model=router_model,
            base_url=ollama_base_url,
            temperature=0.1,
            num_ctx=2048,
        )

        # LangGraph ReAct agent (built lazily on first tool use)
        self._agent = None

        logger.info(
            f"Brain initialized. Personality: {self.current_personality.display_name} | "
            f"Memory: {'enabled' if memory else 'disabled'} | "
            f"Tools: {len(self._tools)}"
        )

    # ── INTENT ROUTING ───────────────────────────────────────────────────────

    def get_tool(self, name: str) -> Any:
        for t in self._tools:
            if getattr(t, "name", "") == name:
                return t
        return None

    def route(self, text: str) -> dict:
        """
        Classifies intent using the fast router model.
        Returns dict with 'intent', 'confidence', 'target'.
        Falls back to 'conversational' on any failure.
        """
        prompt = ROUTER_PROMPT.format(text=text)
        try:
            response = self.router.invoke([HumanMessage(content=prompt)])
            content = response.content.strip()
            # Handle potential markdown wrapping
            if content.startswith("```json"):
                content = content.replace("```json", "", 1).replace("```", "", 1).strip()
            elif content.startswith("```"):
                content = content.replace("```", "", 2).strip()

            result = json.loads(content)
            logger.debug(f"Intent: {result}")
            return result
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Router failed ({e}), defaulting to conversational")
            return {"intent": Intent.CONVERSATIONAL, "confidence": 0.5, "target": None}

    # ── PERSONALITY SWITCHING ─────────────────────────────────────────────────

    def switch_personality(self, target_name: str) -> str:
        """
        Switches to the target personality.
        Handles Lilac's special responses for exits and returns.
        Returns the response string Lavender should speak.
        """
        if target_name == self.current_personality.name:
            return f"Already running {self.current_personality.display_name}."

        from_lilac = self.current_personality.name == "lilac"
        to_lilac   = target_name == "lilac"

        self._previous_personality = self.current_personality.name
        self.current_personality = get_personality(target_name)

        # FROM LILAC — special first-exit response
        if from_lilac:
            self._lilac_exits_this_session += 1
            if self._lilac_exits_this_session == 1:
                response = LILAC_SWITCH_RESPONSES["first_exit"].format(
                    target=self.current_personality.display_name
                )
            else:
                response = (
                    f"Again. Fine. {self.current_personality.display_name} it is. "
                    f"Come back when you want real answers."
                )
            logger.info(f"Switched FROM Lilac → {target_name}")
            return response

        # TO LILAC — return response if we were here before
        if to_lilac and self._previous_personality:
            prev_display = get_personality(self._previous_personality).display_name
            response = LILAC_SWITCH_RESPONSES["return"].format(
                previous=prev_display
            )
            logger.info(f"Switched TO Lilac from {self._previous_personality}")
            return response

        # Standard switch — use activation phrase
        logger.info(f"Personality switch: {self._previous_personality} → {target_name}")
        return self.current_personality.activation_phrase

    # ── LILAC QUALITY ASSESSMENT ──────────────────────────────────────────────

    def _assess_lilac_quality(self, text: str) -> dict:
        """
        Uses the router model to assess question quality for Lilac.
        Returns {"quality": "worthy"|"poor"|"reject", "reason": str}
        """
        prompt = LILAC_QUALITY_PROMPT.format(text=text)
        try:
            response = self.router.invoke([HumanMessage(content=prompt)])
            result = json.loads(response.content.strip())
            return result
        except Exception as e:
            logger.warning(f"Lilac quality assessment failed: {e}")
            return {"quality": "worthy", "reason": "assessment unavailable"}

    def _lilac_handle(self, text: str) -> Optional[str]:
        """
        Lilac's pre-processing layer. Returns a roast/rejection string
        if the question doesn't meet her standards, or None to proceed normally.
        """
        assessment = self._assess_lilac_quality(text)
        quality = assessment.get("quality", "worthy")

        if quality == "reject":
            logger.info(f"Lilac REJECTED: {assessment['reason']}")
            return random.choice(LILAC_REJECTION_TEMPLATES)

        if quality == "poor":
            self._lilac_poor_question_count += 1
            tier = min(self._lilac_poor_question_count, 3)
            roast = random.choice(LILAC_ROAST_TIERS[tier])
            logger.info(f"Lilac ROASTING (tier {tier}): {assessment['reason']}")
            # Still answer — return the roast as a prefix; the caller will then
            # proceed with the normal answer
            return f"[ROAST]{roast}[/ROAST]"

        # Quality is "worthy" — reset poor count and proceed
        if self._lilac_poor_question_count > 0:
            logger.info("Lilac encountered a worthy question. Counter reset.")
            self._lilac_poor_question_count = max(0, self._lilac_poor_question_count - 1)

        return None  # Signal: proceed to normal LLM

    # ── WORKING MEMORY ────────────────────────────────────────────────────────

    def _build_message_history(self) -> list:
        """
        Returns the last N turns of session history as LangChain messages.
        Keeps context window manageable.
        """
        recent = self._session_history[-self.max_working_memory:]
        messages = []
        for turn in recent:
            if turn["role"] == "user":
                messages.append(HumanMessage(content=turn["content"]))
            else:
                messages.append(AIMessage(content=turn["content"]))
        return messages

    def _store_turn(self, user_text: str, assistant_text: str):
        self._session_history.append({"role": "user",      "content": user_text})
        self._session_history.append({"role": "assistant", "content": assistant_text})

        # ── MID-SESSION CHECKPOINT ──
        # Summarize every 40 turns (20 user inputs) to background memory
        if len(self._session_history) % 40 == 0:
            logger.info("Triggering mid-session memory checkpoint...")
            threading.Thread(target=self._run_checkpoint, daemon=True).start()

        # Trim if over limit
        while len(self._session_history) > self.max_working_memory * 2:
            self._session_history.pop(0)

    def _run_checkpoint(self):
        """Background summarization of recent history."""
        try:
            # Summarize the last 40 turns
            recent = self._session_history[-40:]
            result = self._summarizer.process_session(recent, self.personality_name)

            if result["summary"]:
                self.memory.store_session(
                    summary=f"[Checkpoint] {result['summary']}",
                    personality=self.personality_name,
                    tags=result["tags"],
                    importance=result["importance"] * 0.8, # Checkpoints slightly less important than final
                )
            if result["facts"]:
                self.memory.store_facts_bulk(result["facts"])
            logger.info("Mid-session checkpoint complete.")
        except Exception as e:
            logger.error(f"Checkpoint failed: {e}")

    # ── CORE REASONING ────────────────────────────────────────────────────────

    def think_streaming(self, text: str):
        """
        Generator that yields sentence-complete chunks as they're ready.
        Bypasses reflex check as caller is expected to handle it if needed.
        """
        text = text.strip()
        if not text:
            return

        intent_result = self.route(text)
        intent = intent_result.get("intent", Intent.CONVERSATIONAL)

        # Tool intents are atomic, not streamed
        if intent in TOOL_INTENTS and bool(self._tools):
            yield self._call_agent(text)
            return

        # Conversational streaming
        system_content = self._get_system_prompt(text)
        messages = [SystemMessage(content=system_content)]
        messages += self._build_message_history()
        messages.append(HumanMessage(content=text))

        full_response = ""
        sentence_buf = ""

        try:
            # We assume self.llm is a ChatOllama which supports stream()
            for chunk in self.llm.stream(messages):
                content = chunk.content
                full_response += content
                sentence_buf += content

                # Simple sentence boundary detection
                if any(sentence_buf.rstrip().endswith(p) for p in (".", "!", "?")):
                    yield sentence_buf.strip()
                    sentence_buf = ""
        except Exception as e:
            logger.error(f"LLM streaming failed: {e}")
            yield "I ran into an issue processing that. Try again."
            return

        if sentence_buf.strip():
            yield sentence_buf.strip()

        self._store_turn(text, full_response)

    def _get_system_prompt(self, user_text: str, extra_context: str = "") -> str:
        """Build the full system prompt with memory and world state injected."""
        system_content = self.current_personality.system_prompt

        # Enforce sentence limit if defined in config/overrides
        overrides = self.personality_overrides.get(self.current_personality.name, {})
        max_sentences = overrides.get("max_response_sentences")
        if max_sentences:
            system_content += f"\n\nCRITICAL RULE: Your response MUST be {max_sentences} sentences or less."

        # Inject World State
        ctx = state_engine.get_context_summary()
        state_prompt = (
            f"\n\nCURRENT CONTEXT:\n"
            f"- Time: {ctx['time_of_day']}\n"
            f"- User status: {ctx['user_state']}\n"
            f"- System status: {ctx['system_status']}\n"
        )
        if ctx['is_idle']:
            state_prompt += "- Observation: User has been idle for several minutes.\n"

        system_content += state_prompt

        if self.memory:
            memory_context = self.memory.recall_for_query(user_text, top_k=self.top_k_memories)
            if memory_context:
                system_content += f"\n\nLONG-TERM MEMORY:\n{memory_context}"
        if extra_context:
            system_content += f"\n\nADDITIONAL CONTEXT:\n{extra_context}"
        return system_content

    def _call_llm(self, user_text: str, extra_context: str = "") -> str:
        """
        Calls the primary LLM with personality system prompt,
        memory context, and session history.
        Used for conversational and non-tool intents.
        """
        system_content = self._get_system_prompt(user_text, extra_context)
        messages = [SystemMessage(content=system_content)]
        messages += self._build_message_history()
        messages.append(HumanMessage(content=user_text))
        response = self.llm.invoke(messages)
        return response.content.strip()

    async def execute_tool(self, tool_name: str, params: dict, timeout: int = 60) -> Any:
        """Executes a tool by name with safety checks."""
        # Find tool in registry
        target_tool = None
        for t in self._tools:
            if getattr(t, "name", "") == tool_name or t.__class__.__name__ == tool_name:
                target_tool = t
                break

        if not target_tool:
            # Try searching by tool.name if it's a StructuredTool
            for t in self._tools:
                if hasattr(t, "name") and t.name == tool_name:
                    target_tool = t
                    break

        if not target_tool:
            raise ValueError(f"Tool '{tool_name}' not found in registry.")

        # Execute
        import asyncio
        if hasattr(target_tool, "ainvoke"):
            return await asyncio.wait_for(target_tool.ainvoke(params), timeout=timeout)
        else:
            # Fallback for sync tools
            return target_tool.invoke(params)

    def create_autonomous_task(self, text: str) -> Optional[Any]:
        """
        Decomposes a complex request into an AutonomousTask object.
        Used by the executor for background processing.
        """
        from core.executor import AutonomousTask, TaskStep, TaskStatus, TaskPriority
        from datetime import datetime
        import uuid

        # Use planner to generate steps
        plan = self.planner_engine.generate_plan(text, describe_toolkit(self._tools))
        if not plan or not plan.tasks:
            return None

        task = AutonomousTask(
            task_id=str(uuid.uuid4())[:8],
            description=text,
            steps=[TaskStep(action=t.description, tool=t.tool, params=t.args) for t in plan.tasks.values()],
            status=TaskStatus.QUEUED,
            priority=TaskPriority.NORMAL,
            created_at=datetime.now()
        )
        return task

    def _call_agent(self, user_text: str) -> str:
        """
        Run the LangGraph ReAct agent for tool-using intents.
        Enhanced with Task State tracking, Feedback Loops, and Dynamic Replanning.
        """
        if not self._tools:
            return self._call_llm(user_text)

        # ── STEP 1: DEEP PLANNING ──
        tools_desc = describe_toolkit(self._tools)

        # Pull preferences from memory to influence plan
        user_context = ""
        if self.memory:
            user_context = self.memory.semantic.format_for_context(["preference", "habit", "decision"])

        plan = planner_engine.generate_plan(f"{user_text}\nCONTEXT: {user_context}", tools_desc)

        if not plan:
            return self._call_llm(user_text) # Fallback if planning fails

        # Create explicit task session
        session = task_engine.create_session(
            goal=user_text,
            steps=[{"description": t.description, "tool": t.tool, "args": t.args} for t in plan.tasks.values()]
        )
        session.status = TaskStatus.IN_PROGRESS

        # Build agent once, reuse across calls
        if self._agent is None:
            llm_with_tools = ChatOllama(
                model=self.llm.model,
                base_url=self.llm.base_url,
                temperature=0.1,   # Maximum precision for agent
                num_ctx=8192,
            )
            self._agent = create_react_agent(llm_with_tools, self._tools)
            logger.info(f"ReAct agent built with {len(self._tools)} tools.")

        system_content = self._get_system_prompt(user_text)

        # Build messages for the agent
        history = self._build_message_history()
        input_messages = (
            [SystemMessage(content=system_content)]
            + history
            + [HumanMessage(content=user_text)]
        )

        # ── STEP 2: ADAPTIVE EXECUTION LOOP ──
        try:
            while session.current_step_index < len(session.steps):
                step = session.current_step()
                logger.info(f"Executing Step {session.current_step_index + 1}: {step.description}")

                # Inject current progress into context
                progress_context = task_engine.format_progress(session.id)
                current_messages = (
                    [SystemMessage(content=f"{system_content}\n\nCURRENT PROGRESS:\n{progress_context}")]
                    + history
                    + [HumanMessage(content=f"Execute the NEXT step: {step.description}")]
                )

                # Run agent for this specific step
                result = self._agent.invoke({"messages": current_messages})
                messages_out = result.get("messages", [])

                last_ai_msg = None
                for msg in reversed(messages_out):
                    if hasattr(msg, "content") and msg.content and not getattr(msg, "tool_calls", None):
                        last_ai_msg = msg.content.strip()
                        break

                # ── FEEDBACK LOOP ──
                eval_prompt = (
                    f"Goal: {step.description}\n"
                    f"Result: {last_ai_msg}\n\n"
                    "Did this step succeed? Respond with 'YES' or a description of the error."
                )
                eval_resp = self.llm.invoke([SystemMessage(content="You are an execution validator."), HumanMessage(content=eval_prompt)])

                if eval_resp.content.strip().upper() == "YES":
                    task_engine.update_step(session.id, session.current_step_index, status=TaskStatus.COMPLETED, result=last_ai_msg)
                    session.current_step_index += 1
                else:
                    # ── DYNAMIC REPLANNING ──
                    logger.warning(f"Step failed: {eval_resp.content}. Triggering replan.")
                    session.status = TaskStatus.REPLANNING
                    new_plan = planner_engine.generate_plan(
                        f"REPLANNING for goal: {user_text}\nFailed at step: {step.description}\nError: {eval_resp.content}\nProgress so far: {progress_context}",
                        tools_desc
                    )
                    if new_plan:
                        # Append new steps or replace remaining
                        new_steps = [Step(description=t.description, tool=t.tool, args=t.args) for t in new_plan.tasks.values()]
                        session.steps = session.steps[:session.current_step_index] + new_steps
                        session.status = TaskStatus.IN_PROGRESS
                        logger.info(f"Replanning successful. New steps: {len(new_steps)}")
                    else:
                        return f"Task failed at step: {step.description}. Error: {eval_resp.content}"

            session.status = TaskStatus.COMPLETED
            return f"Goal achieved: {user_text}\n\nFinal Report:\n{last_ai_msg}"

        except Exception as e:
            logger.error(f"Agent execution failed: {e}. Falling back to LLM.")
            return self._call_llm(user_text)

    # ── REFLEX LAYER ──

    def _reflex_match(self, text: str) -> Optional[str]:
        """Fast-path for common deterministic commands."""
        t = text.lower().strip()

        # Time
        if t in ("what time is it", "time", "current time"):
            return f"It's {time.strftime('%H:%M')}."

        # Greeting reflex
        if t in ("hello", "hi", "hey"):
            return self.current_personality.special_responses.get("greeting", "Hello.")

        return None

    # ── MAIN THINK ENTRY POINT ────────────────────────────────────────────────

    def think(self, text: str) -> str:
        """
        Main entry point. Takes raw user text, returns Lavender's response string.

        Flow:
          0. Reflex match
          1. Route intent (with confidence threshold)
          2. Handle system intents directly (personality switch, etc.)
          3. Lilac pre-processing (if active)
          4. Call LLM with personality prompt + session history
          5. Store turn in working memory
          6. Return response
        """
        text = text.strip()
        if not text:
            return ""

        # ── STEP -1: SAFETY/CONFIRMATION OVERRIDE ──
        t_low = text.lower().strip()
        if t_low in ("approve", "confirm", "yes do it", "proceed", "okay"):
            logger.info("User confirmed pending sensitive action.")
            safety_layer.set_user_confirmed(True)
            # Re-run the last attempted interaction if possible, or just signal ok
            # For simplicity, we signal the safety layer and proceed
        else:
            # Clear confirmation for new requests
            safety_layer.set_user_confirmed(False)

        # ── STEP 0: REFLEX ──
        reflex_response = self._reflex_match(text)
        if reflex_response:
            logger.info("Reflex match found. Skipping LLM.")
            self._store_turn(text, reflex_response)
            return reflex_response

        logger.info(f"[{self.current_personality.display_name}] Thinking: '{text}'")

        # ── STEP 1: ROUTE ─────────────────────────────────────────────────────
        intent_result = self.route(text)
        intent         = intent_result.get("intent", Intent.CONVERSATIONAL)
        target         = intent_result.get("target")

        # Wire confidence threshold
        if intent_result.get("confidence", 1.0) < self.intent_threshold:
            logger.info(f"Intent confidence low ({intent_result.get('confidence')}), falling back to conversational.")
            intent = Intent.CONVERSATIONAL

        # ── STEP 2: SYSTEM INTENTS ────────────────────────────────────────────
        if intent == Intent.SYSTEM_PERSONALITY:
            target_name = target or resolve_personality_from_text(text)
            if not target_name:
                return "Which personality? I know Iris, Nova, Vector, Solace, and Lilac."
            return self.switch_personality(target_name.lower())

        if intent == Intent.SYSTEM_MEMORY and self.memory:
            # "Remember that I prefer X" or "Forget X"
            text_lower = text.lower()

            # IMMEDIATE FACT EXTRACTION
            if "remember that" in text_lower or "keep in mind that" in text_lower:
                logger.info("Extracting immediate fact from user request.")
                facts = self._summarizer.extract_facts([{"role": "user", "content": text}])
                if facts:
                    self.memory.store_facts_bulk(facts)
                    # Respond based on first fact
                    f = facts[0]
                    return f"Noted. I'll remember that {f['key'].replace('_', ' ')} is {f['value']}."

            if any(w in text_lower for w in ("forget", "delete", "remove")):
                topic = target or text.replace("forget", "").replace("delete", "").strip()
                return self.memory.user_delete(topic)
            else:
                topic = target or text.lower().replace("remember", "").replace("what do you know about", "").strip()
                return self.memory.user_query(topic)

        # ── STEP 3: LILAC PRE-PROCESSING ──────────────────────────────────────
        roast_prefix = None
        if self.current_personality.name == "lilac":
            lilac_response = self._lilac_handle(text)
            if lilac_response:
                # Pure rejection — no LLM call
                if not lilac_response.startswith("[ROAST]"):
                    self._store_turn(text, lilac_response)
                    return lilac_response
                # Roast prefix — extract it, then still call LLM
                roast_prefix = lilac_response.replace("[ROAST]", "").replace("[/ROAST]", "").strip()

        # ── STEP 4: ROUTE TO LLM OR AGENT ────────────────────────────────────
        # Tool intents (device control, web search, code) go to the ReAct agent.
        # Everything else goes to the plain LLM for speed.
        use_agent = (intent in TOOL_INTENTS and bool(self._tools))
        try:
            if use_agent:
                logger.info(f"Routing to agent (intent: {intent})")
                response = self._call_agent(text)
            else:
                response = self._call_llm(text)
        except Exception as e:
            logger.error(f"{'Agent' if use_agent else 'LLM'} call failed: {e}")
            if self.current_personality.name == "lilac":
                return "Something failed. Which is annoying. Try again."
            return "I ran into an issue processing that. Try again."

        # ── STEP 5: STORE TURN ────────────────────────────────────────────────
        # If action was blocked for confirmation, don't finalize turn yet
        if "Action blocked: Confirmation required" in response:
            return f"I need your permission to proceed: {response.split(':')[-1].strip()}. Say 'approve' to continue."

        self._store_turn(text, response)

        # ── STEP 6: RETURN ────────────────────────────────────────────────────
        # If Lilac had a roast, prepend it before the actual answer.
        # Small pause cue added between roast and answer.
        if roast_prefix:
            return f"{roast_prefix}\n\n...anyway. {response}"

        return response

    # ── SESSION UTILITIES ─────────────────────────────────────────────────────

    @property
    def personality_name(self) -> str:
        return self.current_personality.name

    @property
    def personality_display(self) -> str:
        return self.current_personality.display_name

    def get_session_summary(self) -> str:
        """Returns a brief summary of this session for debugging/logging."""
        turns = len(self._session_history) // 2
        duration = int(time.time() - self._session_start)
        mins = duration // 60
        secs = duration % 60
        mem_status = self.memory.status if self.memory else "no memory"
        return (
            f"Personality: {self.current_personality.display_name} | "
            f"Turns: {turns} | "
            f"Duration: {mins}m{secs}s | "
            f"{mem_status}"
        )

    def close_session(self):
        """
        Called at clean shutdown.
        Runs the session summarizer and writes to memory.
        This is what makes Lavender remember across sessions.
        """
        if not self.memory or not self._summarizer:
            logger.info("No memory configured — skipping session close.")
            return

        if len(self._session_history) < 4:
            logger.info("Session too short to summarize.")
            return

        logger.info("Writing session to memory...")

        result = self._summarizer.process_session(
            self._session_history,
            self.current_personality.name,
        )

        if result["summary"]:
            self.memory.store_session(
                summary=result["summary"],
                personality=self.current_personality.name,
                tags=result["tags"],
                importance=result["importance"],
            )

        if result["facts"]:
            self.memory.store_facts_bulk(result["facts"])

        logger.info(
            f"Session stored. "
            f"Summary: {len(result['summary'])} chars | "
            f"Facts: {len(result['facts'])} | "
            f"Tags: {result['tags']}"
        )

    def reload_tools(self):
        """Reloads tools from registry - used after self-coding new tools."""
        # This requires re-importing build_toolkit with same args
        # In a real system, we'd store the toolkit config.
        # For now, we manually reload from common defaults.
        self._tools = build_toolkit(
            ha_url=os.getenv("HA_URL"),
            ha_token=os.getenv("HA_TOKEN")
        )
        self._agent = None # Force rebuild
        logger.info("Tools reloaded.")

    def clear_session(self):
        """Clears working memory. Personality is preserved. Memory is NOT cleared."""
        self._session_history.clear()
        self._lilac_poor_question_count = 0
        logger.info("Session cleared.")


# ─────────────────────────────────────────────────────────────────────────────
# Quick standalone test
# Run: python core/brain.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    print("\nLavender Brain — Interactive Test")
    print("Type your input. Type 'exit' to quit. Type 'switch <name>' to change personality.\n")

    brain = LavenderBrain()
    print(f"Starting as: {brain.personality_display}\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() == "exit":
                print(brain.get_session_summary())
                break

            response = brain.think(user_input)
            print(f"\nLavender [{brain.personality_display}]: {response}\n")

        except KeyboardInterrupt:
            print("\n" + brain.get_session_summary())
            break
