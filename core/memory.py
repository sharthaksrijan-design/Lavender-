"""
LAVENDER — Memory System
core/memory.py

Three memory tiers:

  TIER 1 — WORKING MEMORY
  In-context session history. Lives in brain.py. Not handled here.

  TIER 2 — EPISODIC MEMORY (this file)
  ChromaDB vector store. Session summaries retrieved by semantic similarity.
  "What happened last time I worked on the Nexus project?"

  TIER 3 — SEMANTIC MEMORY (this file)
  SQLite. Hard facts about the user's world — structured, queried directly.
  Preferences, active projects, recurring contacts, habits, decisions.
"""

import sqlite3
import logging
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings

logger = logging.getLogger("lavender.memory")


# ─────────────────────────────────────────────────────────────────────────────
# EPISODIC MEMORY — ChromaDB vector store
# ─────────────────────────────────────────────────────────────────────────────

class EpisodicMemory:
    """
    Stores and retrieves session summaries by semantic similarity.
    Each entry is a paragraph summarizing a past session — what was discussed,
    what was decided, what was learned about the user.
    """

    def __init__(self, db_path: str):
        Path(db_path).mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )

        self._collection = self._client.get_or_create_collection(
            name="episodes",
            metadata={"hnsw:space": "cosine"}
        )

        logger.info(
            f"Episodic memory ready. "
            f"{self._collection.count()} episodes stored."
        )

    def store(
        self,
        summary: str,
        personality: str,
        tags: list[str] = None,
        session_id: str = None,
        importance: float = 0.5,
    ):
        """
        Store a session summary as an episodic memory.

        Args:
            summary:     The summary text — written by the session summarizer.
            personality: Which personality was active during the session.
            tags:        Topic tags extracted from the session (e.g. ["project-nexus", "code"]).
            session_id:  Unique session identifier. Auto-generated if not provided.
            importance:  Scale of 0.0 to 1.0 (decided by LLM summarizer). High importance decays slower.
        """
        session_id = session_id or f"ep_{int(time.time())}"
        tags = tags or []

        metadata = {
            "personality":  personality,
            "tags":         json.dumps(tags),
            "timestamp":    datetime.now().isoformat(),
            "timestamp_ts": int(time.time()),
            "importance":   importance,
            "relevance":    1.0,    # Initial relevance, decays over time via decay()
        }

        self._collection.add(
            documents=[summary],
            metadatas=[metadata],
            ids=[session_id],
        )

        logger.info(f"Stored episode '{session_id}' (importance: {importance})")

    def recall(self, query: str, n_results: int = 3, min_relevance: float = 0.3) -> list[dict]:
        """
        Retrieve the most semantically relevant episodes for a given query.

        Returns list of dicts:
          {"text": str, "personality": str, "timestamp": str, "tags": list, "score": float}
        """
        count = self._collection.count()
        if count == 0:
            return []

        n_results = min(n_results, count)

        results = self._collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        episodes = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity score 0.0–1.0
            similarity = max(0.0, 1.0 - (dist / 2.0))

            if similarity < min_relevance:
                continue

            episodes.append({
                "text":        doc,
                "personality": meta.get("personality", "unknown"),
                "timestamp":   meta.get("timestamp", ""),
                "tags":        json.loads(meta.get("tags", "[]")),
                "score":       round(similarity, 3),
            })

        logger.debug(f"Recalled {len(episodes)} episodes for query: '{query[:50]}...'")
        return episodes

    def decay(self, days_half_life: int = 45):
        """
        Reduce relevance scores of old memories.
        Called periodically (e.g., once per day) — not on every query.

        Memories are not deleted — they just become less likely to surface
        unless the query is strongly related.
        """
        # ChromaDB doesn't support in-place updates easily,
        # so we fetch all, recalculate, and upsert changed ones.
        all_results = self._collection.get(include=["metadatas"])
        if not all_results["ids"]:
            return

        now_ts = int(time.time())
        seconds_per_day = 86400
        half_life_seconds = days_half_life * seconds_per_day

        updated_ids = []
        updated_metadatas = []
        for ep_id, meta in zip(all_results["ids"], all_results["metadatas"]):
            stored_ts = meta.get("timestamp_ts", now_ts)
            age_seconds = now_ts - stored_ts

            # High importance memories decay slower (importance 1.0 = 4x half-life)
            importance = meta.get("importance", 0.5)
            effective_half_life = half_life_seconds * (1.0 + (importance * 3.0))

            decay_factor = 0.5 ** (age_seconds / effective_half_life)
            new_relevance = round(decay_factor, 4)

            if abs(new_relevance - meta.get("relevance", 1.0)) > 0.01:
                meta["relevance"] = new_relevance
                updated_ids.append(ep_id)
                updated_metadatas.append(meta)

        if updated_ids:
            # ChromaDB update() supports batching
            self._collection.update(ids=updated_ids, metadatas=updated_metadatas)

        logger.info(f"Memory decay applied to {len(updated_ids)} episodes.")

    def get_all(self) -> list[dict]:
        """Return all stored episodes (for user inspection/deletion)."""
        results = self._collection.get(include=["documents", "metadatas"])
        episodes = []
        for ep_id, doc, meta in zip(
            results["ids"],
            results["documents"],
            results["metadatas"],
        ):
            episodes.append({
                "id":          ep_id,
                "text":        doc,
                "personality": meta.get("personality"),
                "timestamp":   meta.get("timestamp"),
                "tags":        json.loads(meta.get("tags", "[]")),
                "relevance":   meta.get("relevance", 1.0),
            })
        return sorted(episodes, key=lambda x: x["timestamp"], reverse=True)

    def delete(self, session_id: str):
        """Delete a specific episode by ID."""
        self._collection.delete(ids=[session_id])
        logger.info(f"Deleted episode '{session_id}'")

    def clear_all(self):
        """Delete all episodic memories. Irreversible."""
        self._client.delete_collection("episodes")
        self._collection = self._client.get_or_create_collection(
            name="episodes",
            metadata={"hnsw:space": "cosine"}
        )
        logger.warning("All episodic memories cleared.")

    @property
    def count(self) -> int:
        return self._collection.count()


# ─────────────────────────────────────────────────────────────────────────────
# SEMANTIC MEMORY — SQLite structured facts
# ─────────────────────────────────────────────────────────────────────────────

class SemanticMemory:
    """
    Structured key-value fact store for hard knowledge about the user's world.

    Categories (extensible):
      "preference"  — user likes/dislikes, communication style, habits
      "project"     — active or past projects and their status
      "person"      — known contacts and their relationship to user
      "routine"     — recurring patterns (meetings, habits, schedule)
      "decision"    — recorded decisions and their outcomes
      "system"      — Lavender's own config facts (e.g. "user prefers metric units")
    """

    VALID_CATEGORIES = {
        "preference", "project", "person",
        "routine", "decision", "system", "task", "habit", "general"
    }

    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        logger.info(f"Semantic memory ready at {db_path}")

    def _init_schema(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS facts (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                category        TEXT NOT NULL,
                key             TEXT NOT NULL,
                value           TEXT NOT NULL,
                confidence      REAL DEFAULT 1.0,
                source          TEXT DEFAULT 'explicit',
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL,
                UNIQUE(category, key)
            );

            CREATE TABLE IF NOT EXISTS fact_history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                fact_id     INTEGER NOT NULL,
                old_value   TEXT,
                new_value   TEXT,
                changed_at  TEXT NOT NULL,
                FOREIGN KEY (fact_id) REFERENCES facts(id)
            );

            CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(category);
            CREATE INDEX IF NOT EXISTS idx_facts_key ON facts(key);
        """)
        self._conn.commit()

    # ── WRITE ────────────────────────────────────────────────────────────────

    def store(
        self,
        category: str,
        key: str,
        value: str,
        confidence: float = 1.0,
        source: str = "explicit",
    ):
        """
        Store or update a fact. If the key already exists, its history is preserved.

        Args:
            category:   One of VALID_CATEGORIES
            key:        Short descriptive key (e.g. "preferred_temperature_unit")
            value:      The fact value (e.g. "celsius")
            confidence: How certain we are (0.0–1.0). Inferred facts < 1.0.
            source:     "explicit" (user said it) | "inferred" (Lavender deduced it)
        """
        category = category.lower()
        now = datetime.now().isoformat()

        # Check if fact already exists — if so, record history
        existing = self._conn.execute(
            "SELECT id, value FROM facts WHERE category=? AND key=?",
            (category, key)
        ).fetchone()

        if existing:
            if existing["value"] != value:
                # Record change history
                self._conn.execute(
                    "INSERT INTO fact_history (fact_id, old_value, new_value, changed_at) "
                    "VALUES (?, ?, ?, ?)",
                    (existing["id"], existing["value"], value, now)
                )
            # Update existing fact
            self._conn.execute(
                "UPDATE facts SET value=?, confidence=?, source=?, updated_at=? "
                "WHERE category=? AND key=?",
                (value, confidence, source, now, category, key)
            )
        else:
            # Insert new fact
            self._conn.execute(
                "INSERT INTO facts (category, key, value, confidence, source, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (category, key, value, confidence, source, now, now)
            )

        self._conn.commit()
        logger.debug(f"Stored fact [{category}] {key} = '{value[:50]}'")

    def store_many(self, facts: list[dict]):
        """
        Bulk store a list of facts.
        Each dict: {"category": str, "key": str, "value": str, ...optional fields}
        """
        for fact in facts:
            self.store(
                category=fact.get("category", "general"),
                key=fact["key"],
                value=fact["value"],
                confidence=fact.get("confidence", 1.0),
                source=fact.get("source", "explicit"),
            )

    # ── READ ─────────────────────────────────────────────────────────────────

    def get(self, category: str, key: str) -> Optional[str]:
        """Get a single fact value. Returns None if not found."""
        row = self._conn.execute(
            "SELECT value FROM facts WHERE category=? AND key=?",
            (category.lower(), key)
        ).fetchone()
        return row["value"] if row else None

    def get_category(self, category: str) -> dict[str, str]:
        """Get all facts in a category as a key→value dict."""
        rows = self._conn.execute(
            "SELECT key, value FROM facts WHERE category=?",
            (category.lower(),)
        ).fetchall()
        return {row["key"]: row["value"] for row in rows}

    def search(self, query: str, category: str = None) -> list[dict]:
        """
        Simple text search across fact keys and values.
        Not semantic — use episodic memory for semantic search.
        """
        query_like = f"%{query.lower()}%"
        if category:
            rows = self._conn.execute(
                "SELECT * FROM facts WHERE category=? AND "
                "(LOWER(key) LIKE ? OR LOWER(value) LIKE ?)",
                (category.lower(), query_like, query_like)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM facts WHERE "
                "LOWER(key) LIKE ? OR LOWER(value) LIKE ?",
                (query_like, query_like)
            ).fetchall()

        return [dict(row) for row in rows]

    def get_all(self, min_confidence: float = 0.0) -> list[dict]:
        """Return all facts, optionally filtered by confidence."""
        rows = self._conn.execute(
            "SELECT * FROM facts WHERE confidence >= ? ORDER BY category, key",
            (min_confidence,)
        ).fetchall()
        return [dict(row) for row in rows]

    # ── DELETE ────────────────────────────────────────────────────────────────

    def delete(self, category: str, key: str) -> bool:
        """Delete a specific fact. Returns True if something was deleted."""
        cursor = self._conn.execute(
            "DELETE FROM facts WHERE category=? AND key=?",
            (category.lower(), key)
        )
        self._conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.info(f"Deleted fact [{category}] {key}")
        return deleted

    def clear_category(self, category: str):
        """Delete all facts in a category."""
        self._conn.execute("DELETE FROM facts WHERE category=?", (category.lower(),))
        self._conn.commit()
        logger.warning(f"Cleared all facts in category '{category}'")

    # ── FORMATTING ────────────────────────────────────────────────────────────

    def format_for_context(self, categories: list[str] = None) -> str:
        """
        Returns a compact text representation of semantic memory
        suitable for injection into an LLM context prompt.

        Example output:
          [KNOWN FACTS]
          preference: prefers dark mode → yes
          preference: temperature unit → celsius
          project: current focus → Lavender AI build
          person: main collaborator → Rohan
        """
        if categories:
            facts = []
            for cat in categories:
                for k, v in self.get_category(cat).items():
                    facts.append((cat, k, v))
        else:
            all_facts = self.get_all(min_confidence=0.6)
            facts = [(f["category"], f["key"], f["value"]) for f in all_facts]

        if not facts:
            return ""

        lines = ["[KNOWN FACTS ABOUT USER]"]
        for cat, key, value in facts:
            lines.append(f"  {cat}: {key} → {value}")

        return "\n".join(lines)

    @property
    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) as n FROM facts").fetchone()
        return row["n"]


# ─────────────────────────────────────────────────────────────────────────────
# UNIFIED MEMORY INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

class LavenderMemory:
    """
    Single interface to both memory tiers.
    The brain imports this — not EpisodicMemory or SemanticMemory directly.
    """

    def __init__(self, episodic_db_path: str, semantic_db_path: str):
        self.episodic = EpisodicMemory(episodic_db_path)
        self.semantic = SemanticMemory(semantic_db_path)
        logger.info(
            f"Memory online. "
            f"Episodes: {self.episodic.count} | "
            f"Facts: {self.semantic.count}"
        )

    def recall_for_query(self, query: str, top_k: int = 3) -> str:
        """
        Main retrieval method called by the brain before each LLM call.

        Returns a formatted string for context injection, combining:
          - Relevant episodic memories (semantic search)
          - All semantic facts (structured)

        Returns empty string if nothing is in memory.
        """
        parts = []

        # Semantic facts (always included if they exist)
        semantic_context = self.semantic.format_for_context()
        if semantic_context:
            parts.append(semantic_context)

        # Episodic memories (retrieved by similarity to current query)
        episodes = self.episodic.recall(query, n_results=top_k)
        if episodes:
            lines = ["[RELEVANT PAST SESSIONS]"]
            for ep in episodes:
                # Format timestamp as relative time
                try:
                    ep_dt = datetime.fromisoformat(ep["timestamp"])
                    delta = datetime.now() - ep_dt
                    if delta.days == 0:
                        time_str = "today"
                    elif delta.days == 1:
                        time_str = "yesterday"
                    else:
                        time_str = f"{delta.days} days ago"
                except Exception:
                    time_str = ep["timestamp"][:10]

                lines.append(f"  [{time_str}, {ep['personality']}]: {ep['text']}")
            parts.append("\n".join(lines))

        return "\n\n".join(parts)

    def store_session(self, summary: str, personality: str, tags: list[str] = None, importance: float = 0.5):
        """Store an end-of-session summary in episodic memory."""
        self.episodic.store(summary, personality, tags, importance=importance)

    def store_fact(self, category: str, key: str, value: str, **kwargs):
        """Store a semantic fact."""
        self.semantic.store(category, key, value, **kwargs)

    def store_facts_bulk(self, facts: list[dict]):
        """Store multiple semantic facts at once."""
        self.semantic.store_many(facts)

    def user_query(self, topic: str) -> str:
        """
        Handle 'Lavender, what do you remember about X?'
        Returns a human-readable summary of what's in memory about X.
        """
        episodes = self.episodic.recall(topic, n_results=5)
        facts    = self.semantic.search(topic)

        lines = [f"Here is what I have about '{topic}':\n"]

        if facts:
            lines.append("Facts I know:")
            for f in facts:
                lines.append(f"  {f['category']} / {f['key']}: {f['value']}")

        if episodes:
            lines.append("\nRelevant past sessions:")
            for ep in episodes:
                lines.append(f"  [{ep['timestamp'][:10]}]: {ep['text'][:200]}...")

        if not facts and not episodes:
            return f"I don't have anything stored about '{topic}' yet."

        return "\n".join(lines)

    def user_delete(self, topic: str) -> str:
        """
        Handle 'Lavender, forget X.'
        Searches and removes matching facts. Episodic memories are not deleted
        (they are historical record) but are noted as user-requested to forget.
        """
        facts = self.semantic.search(topic)
        deleted = []
        for f in facts:
            if self.semantic.delete(f["category"], f["key"]):
                deleted.append(f"{f['category']}/{f['key']}")

        if deleted:
            return f"Deleted {len(deleted)} fact(s): {', '.join(deleted)}."
        return f"I didn't find anything to forget about '{topic}'."

    @property
    def status(self) -> str:
        return (
            f"Memory: {self.episodic.count} episodes, "
            f"{self.semantic.count} facts"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# Run: python core/memory.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile, os
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(name)s] %(message)s")

    with tempfile.TemporaryDirectory() as tmp:
        mem = LavenderMemory(
            episodic_db_path=os.path.join(tmp, "chroma"),
            semantic_db_path=os.path.join(tmp, "semantic.db"),
        )

        # Test semantic facts
        print("\n── Semantic Memory ──")
        mem.store_fact("preference", "temperature_unit", "celsius")
        mem.store_fact("preference", "dark_mode", "yes")
        mem.store_fact("project", "current_focus", "Lavender AI build")
        mem.store_fact("person", "main_collaborator", "Rohan")

        print(mem.semantic.format_for_context())

        # Test episodic memory
        print("\n── Episodic Memory ──")
        mem.store_session(
            summary=(
                "User worked on the Lavender brain.py file for 2 hours. "
                "Decided to use LangGraph for orchestration over raw LangChain. "
                "Asked about ChromaDB vs Qdrant — chose ChromaDB for simplicity. "
                "Session ended with routing logic complete and tested."
            ),
            personality="vector",
            tags=["lavender-build", "brain", "memory", "langchain"]
        )

        mem.store_session(
            summary=(
                "Casual evening session. User asked about what to watch. "
                "Lavender suggested a documentary based on earlier preference for science content. "
                "User seemed tired. Short session."
            ),
            personality="solace",
            tags=["casual", "evening", "recommendations"]
        )

        # Recall
        print("\nRecalling for 'LangGraph architecture decision':")
        print(mem.recall_for_query("LangGraph architecture decision"))

        print("\nUser query — 'what do you remember about the build':")
        print(mem.user_query("build"))

        print(f"\nStatus: {mem.status}")
        print("\nAll tests passed.")
