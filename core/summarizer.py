"""
LAVENDER — Session Summarizer
core/summarizer.py

Runs at the end of every session (or periodically for long sessions).

Does two things:
  1. Writes an episodic memory — a paragraph summarizing what happened,
     what was decided, what was interesting or important.

  2. Extracts semantic facts — specific, structured facts about the user
     that should be remembered indefinitely (preferences, projects, people).

Both outputs are written to LavenderMemory by brain.py on session close.
"""

import json
import logging
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger("lavender.summarizer")


# ── PROMPTS ───────────────────────────────────────────────────────────────────

EPISODE_SUMMARY_PROMPT = """You are summarizing a session between a user and Lavender,
their personal AI system.

Write a single paragraph (4–8 sentences) capturing:
- What the user was working on or thinking about
- Any decisions that were made
- Anything notable about the user's state (focused, tired, creative, frustrated)
- What was resolved and what was left open
- Any preferences or patterns you noticed

Write in third person. Be specific. Omit small talk. Focus on what a future
Lavender instance would actually need to know to be useful in a related future session.

DO NOT include generic filler. If the session was trivial, say so briefly.

SESSION TRANSCRIPT:
{transcript}

PERSONALITY ACTIVE DURING SESSION: {personality}

Write the summary paragraph now:"""


FACT_EXTRACTION_PROMPT = """You are extracting structured facts from a conversation
between a user and Lavender, their personal AI system.

Extract ONLY facts that are:
- Explicit (user stated them directly) OR highly confident inferences
- Worth remembering long-term (not just relevant to this session)
- About the user, their world, preferences, or ongoing projects

Return ONLY valid JSON. No explanation. No markdown.

Categories:
- "preference": things the user likes, dislikes, or prefers
- "project": ongoing work, its status, key decisions
- "person": people mentioned, their relationship to the user
- "routine": recurring patterns, schedule, habits
- "decision": important decisions made and why
- "system": things about how the user wants Lavender to behave

SESSION TRANSCRIPT:
{transcript}

Return a JSON array. Each item:
{{
  "category": "<category>",
  "key": "<short snake_case key>",
  "value": "<concise value string>",
  "confidence": <0.5–1.0>,
  "source": "explicit" or "inferred"
}}

If no facts worth storing, return an empty array: []

Return JSON array now:"""


TAG_EXTRACTION_PROMPT = """Extract 2–5 topic tags from this session transcript.
Tags should be lowercase, hyphenated if multi-word, specific to actual content.

SESSION: {transcript_snippet}

Return ONLY a JSON array of strings. Example: ["project-lavender", "llm-routing", "architecture"]

Return JSON now:"""


# ─────────────────────────────────────────────────────────────────────────────

class SessionSummarizer:
    """
    Uses the primary LLM to extract episodic summaries and semantic facts
    from a session's conversation history.
    """

    def __init__(
        self,
        model: str = "llama3.1:70b-instruct-q4_K_M",
        ollama_base_url: str = "http://localhost:11434",
    ):
        # Use a slightly lower temperature for extraction — we want accuracy
        self.llm = ChatOllama(
            model=model,
            base_url=ollama_base_url,
            temperature=0.3,
            num_ctx=8192,
        )
        logger.info("Session summarizer initialized.")

    # ── TRANSCRIPT FORMATTING ─────────────────────────────────────────────────

    def _format_transcript(self, session_history: list[dict]) -> str:
        """
        Converts session history (list of {role, content} dicts) into
        a readable transcript string for the LLM to process.
        """
        lines = []
        for turn in session_history:
            role = "User" if turn["role"] == "user" else "Lavender"
            content = turn["content"].strip()
            # Truncate very long turns (e.g., code outputs)
            if len(content) > 800:
                content = content[:800] + "... [truncated]"
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    # ── EPISODIC SUMMARY ──────────────────────────────────────────────────────

    def write_episode(self, session_history: list[dict], personality: str) -> str:
        """
        Generates a 4–8 sentence episodic memory paragraph from session history.
        Returns the summary string.
        """
        if len(session_history) < 4:
            # Session too short to be worth summarizing
            logger.info("Session too short to summarize (<4 turns). Skipping.")
            return ""

        transcript = self._format_transcript(session_history)
        prompt = EPISODE_SUMMARY_PROMPT.format(
            transcript=transcript,
            personality=personality.upper()
        )

        try:
            response = self.llm.invoke([
                SystemMessage(content="You are a precise, factual summarizer. "
                                      "Be specific. Never fabricate details."),
                HumanMessage(content=prompt)
            ])
            summary = response.content.strip()
            logger.info(f"Episode summary written ({len(summary)} chars).")
            return summary
        except Exception as e:
            logger.error(f"Episode summary failed: {e}")
            return ""

    # ── FACT EXTRACTION ───────────────────────────────────────────────────────

    def extract_facts(self, session_history: list[dict]) -> list[dict]:
        """
        Extracts structured semantic facts from session history.
        Returns a list of fact dicts ready to pass to SemanticMemory.store_many().
        """
        if len(session_history) < 2:
            return []

        transcript = self._format_transcript(session_history)
        prompt = FACT_EXTRACTION_PROMPT.format(transcript=transcript)

        try:
            response = self.llm.invoke([
                SystemMessage(content="You are a precise fact extractor. "
                                      "Return only valid JSON arrays. "
                                      "Never fabricate facts not present in the transcript."),
                HumanMessage(content=prompt)
            ])

            raw = response.content.strip()

            # Strip markdown code blocks if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            facts = json.loads(raw)

            # Validate structure
            valid_facts = []
            for f in facts:
                if all(k in f for k in ("category", "key", "value")):
                    valid_facts.append(f)
                else:
                    logger.warning(f"Malformed fact skipped: {f}")

            logger.info(f"Extracted {len(valid_facts)} facts from session.")
            return valid_facts

        except json.JSONDecodeError as e:
            logger.error(f"Fact extraction JSON parse failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Fact extraction failed: {e}")
            return []

    # ── TAG EXTRACTION ────────────────────────────────────────────────────────

    def extract_tags(self, session_history: list[dict]) -> list[str]:
        """
        Extracts 2–5 topic tags from the session for episodic metadata.
        """
        if not session_history:
            return []

        # Use just the first 1000 chars of transcript for tags — cheap operation
        transcript_snippet = self._format_transcript(session_history[:8])[:1000]
        prompt = TAG_EXTRACTION_PROMPT.format(transcript_snippet=transcript_snippet)

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            raw = response.content.strip()

            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            tags = json.loads(raw)
            tags = [str(t).lower().strip() for t in tags if t][:5]
            logger.debug(f"Session tags: {tags}")
            return tags

        except Exception as e:
            logger.warning(f"Tag extraction failed: {e}. Using empty tags.")
            return []

    # ── FULL SESSION CLOSE ────────────────────────────────────────────────────

    def process_session(
        self,
        session_history: list[dict],
        personality: str,
    ) -> dict:
        """
        Full session processing pipeline. Call this at session end.

        Returns dict:
        {
          "summary": str,       # The episodic summary paragraph
          "facts": list[dict],  # Extracted semantic facts
          "tags": list[str],    # Topic tags
        }

        The caller (brain.py) is responsible for writing these to LavenderMemory.
        """
        logger.info(
            f"Processing session close. "
            f"{len(session_history) // 2} turns, personality: {personality}"
        )

        summary = self.write_episode(session_history, personality)
        facts   = self.extract_facts(session_history)
        tags    = self.extract_tags(session_history)

        return {
            "summary": summary,
            "facts":   facts,
            "tags":    tags,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# Run: python core/summarizer.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    summarizer = SessionSummarizer()

    # Fake session history
    fake_session = [
        {"role": "user",      "content": "Hey Lavender, I want to start using Python for data analysis. I've been using Excel but it's getting slow."},
        {"role": "assistant", "content": "Good move. pandas is the standard for that kind of work. What are you analyzing?"},
        {"role": "user",      "content": "Sales data for my startup. About 50,000 rows. We sell furniture online."},
        {"role": "assistant", "content": "50k rows is where Excel starts struggling. pandas will handle it fine. Do you have the data in CSV format?"},
        {"role": "user",      "content": "Yeah CSV. My name is Arjun by the way."},
        {"role": "assistant", "content": "Nice to meet you Arjun. Let's start with loading and exploring the data."},
        {"role": "user",      "content": "Sounds good. I prefer short explanations, I pick things up fast."},
        {"role": "assistant", "content": "Noted. Here's the bare minimum to get started..."},
        {"role": "user",      "content": "Perfect. Can we do this again tomorrow? Same time around 9pm?"},
        {"role": "assistant", "content": "Absolutely. We'll pick up from here."},
    ]

    print("\nProcessing test session...\n")
    result = summarizer.process_session(fake_session, personality="vector")

    print("── EPISODE SUMMARY ──")
    print(result["summary"])

    print("\n── EXTRACTED FACTS ──")
    for f in result["facts"]:
        print(f"  [{f['category']}] {f['key']} = {f['value']} (confidence: {f.get('confidence', '?')})")

    print(f"\n── TAGS ──")
    print(result["tags"])
