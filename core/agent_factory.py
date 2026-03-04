"""
LAVENDER — Dynamic Agent Factory
Creates specialized agents for specific tasks on demand.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, Optional, List, Any
from enum import Enum
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama

logger = logging.getLogger("lavender.agent_factory")

class AgentType(Enum):
    DATABASE = "database"
    WEB = "web"
    FILE = "file"
    RESEARCH = "research"
    CUSTOM = "custom"

@dataclass
class SpawnedAgent:
    id: str
    name: str
    purpose: str
    graph: Any # LangGraph instance
    type: AgentType

class AgentFactory:
    """
    Spawns and manages specialized sub-agents.
    """

    def __init__(self, brain):
        self.brain = brain
        self._spawned: Dict[str, SpawnedAgent] = {}

    def spawn(self, name: str, purpose: str, agent_type: AgentType = AgentType.CUSTOM) -> SpawnedAgent:
        logger.info(f"Spawning specialized agent: {name} ({agent_type.value})")

        # Build specialized LLM
        specialized_llm = ChatOllama(
            model=self.brain.llm.model,
            base_url=self.brain.llm.base_url,
            temperature=0.2
        )

        # Select subset of tools based on type
        tools = self._select_tools(agent_type)

        # Create ReAct agent
        graph = create_react_agent(specialized_llm, tools)

        agent = SpawnedAgent(
            id=f"agent_{len(self._spawned)}",
            name=name,
            purpose=purpose,
            graph=graph,
            type=agent_type
        )

        self._spawned[agent.id] = agent
        return agent

    def _select_tools(self, agent_type: AgentType) -> List[Any]:
        all_tools = self.brain._tools
        if agent_type == AgentType.WEB:
            return [t for t in all_tools if t.name in ("search_web", "fetch_page", "get_weather")]
        elif agent_type == AgentType.FILE:
            return [t for t in all_tools if t.name in ("file_ops", "run_python")]
        # Default: all tools
        return all_tools

    def get_agent(self, agent_id: str) -> Optional[SpawnedAgent]:
        return self._spawned.get(agent_id)
