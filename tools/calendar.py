"""
LAVENDER — Calendar Tool
tools/calendar.py

Local calendar using iCalendar format (.ics file).
No Google account required. Syncs with any CalDAV client (Thunderbird, iOS, etc.)
"""
import json
import re
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict
import dateparser

logger = logging.getLogger("lavender.calendar")

CALENDAR_PATH = Path(__file__).parent.parent / "memory" / "lavender.ics"

def _load_events() -> List[Dict]:
    if not CALENDAR_PATH.exists():
        return []
    try:
        from icalendar import Calendar
        cal = Calendar.from_ical(CALENDAR_PATH.read_bytes())
        events = []
        for component in cal.walk():
            if component.name == "VEVENT":
                start = component.get("DTSTART").dt
                # Handle both datetime and date objects
                if not hasattr(start, "isoformat"):
                    start = datetime.combine(start, datetime.min.time())

                end = component.get("DTEND")
                if end:
                    end = end.dt
                    if not hasattr(end, "isoformat"):
                        end = datetime.combine(end, datetime.min.time())
                else:
                    end = start + timedelta(hours=1)

                events.append({
                    "uid":     str(component.get("UID", "")),
                    "title":   str(component.get("SUMMARY", "")),
                    "start":   start.isoformat(),
                    "end":     end.isoformat(),
                    "notes":   str(component.get("DESCRIPTION", "")),
                })
        return sorted(events, key=lambda e: e["start"])
    except Exception as e:
        logger.error(f"Calendar load error: {e}")
        return []

def _save_event(title: str, start: datetime, end: datetime, notes: str = "") -> str:
    from icalendar import Calendar, Event
    import uuid as _uuid

    cal = Calendar()
    cal.add("prodid", "-//Lavender AI//EN")
    cal.add("version", "2.0")

    # Load existing events
    if CALENDAR_PATH.exists():
        try:
            existing = Calendar.from_ical(CALENDAR_PATH.read_bytes())
            for component in existing.walk():
                if component.name == "VEVENT":
                    cal.add_component(component)
        except Exception:
            pass

    event = Event()
    uid = str(_uuid.uuid4())
    event.add("uid",         uid)
    event.add("summary",     title)
    event.add("dtstart",     start)
    event.add("dtend",       end)
    event.add("description", notes)
    event.add("dtstamp",     datetime.now())
    cal.add_component(event)

    CALENDAR_PATH.parent.mkdir(parents=True, exist_ok=True)
    CALENDAR_PATH.write_bytes(cal.to_ical())
    return uid

def make_calendar_tools() -> List:
    from langchain_core.tools import tool
    from core.safety import instance as safety_layer

    # Simple tool wrapper for local logic since we can't use safe_tool here
    # due to import order. We'll rely on tool_registry's SafeTool.

    @tool
    def add_calendar_event(title: str, when: str, duration_minutes: int = 60, notes: str = "") -> str:
        """
        Add an event to the calendar.
        title: event name
        when: natural language time e.g. "tomorrow at 3pm", "next Monday at 10am"
        duration_minutes: how long (default 60)
        notes: optional description
        """
        try:
            start = dateparser.parse(when, settings={"PREFER_DATES_FROM": "future"})
            if not start:
                return f"Could not understand the time '{when}'. Try 'tomorrow at 3pm' or 'Friday at 10am'."
            end = start + timedelta(minutes=duration_minutes)
            uid = _save_event(title, start, end, notes)
            return (f"Added: '{title}' on {start.strftime('%A %d %B at %H:%M')} "
                    f"({duration_minutes} min). ID: {uid[:8]}")
        except Exception as e:
            return f"Failed to add event: {e}"

    @tool
    def list_calendar_events(when: str = "today") -> str:
        """
        List upcoming calendar events.
        when: 'today', 'tomorrow', 'this week', 'next 7 days'
        """
        try:
            events = _load_events()
            if not events:
                return "No events in calendar."

            now = datetime.now()
            if when.lower() == "today":
                window_start = now.replace(hour=0, minute=0, second=0)
                window_end   = now.replace(hour=23, minute=59, second=59)
            elif when.lower() == "tomorrow":
                tomorrow = now + timedelta(days=1)
                window_start = tomorrow.replace(hour=0, minute=0)
                window_end   = tomorrow.replace(hour=23, minute=59)
            else:
                window_start = now
                window_end   = now + timedelta(days=7)

            relevant = []
            for e in events:
                try:
                    estart = datetime.fromisoformat(e["start"])
                    if window_start <= estart <= window_end:
                        relevant.append(e)
                except Exception:
                    pass

            if not relevant:
                return f"No events {when}."

            lines = [f"Events {when}:"]
            for e in relevant:
                try:
                    t = datetime.fromisoformat(e["start"]).strftime("%H:%M")
                    lines.append(f"  {t} — {e['title']}")
                    if e["notes"]:
                        lines.append(f"         {e['notes'][:80]}")
                except Exception:
                    lines.append(f"  {e['title']}")
            return "\n".join(lines)

        except Exception as e:
            return f"Calendar error: {e}"

    @tool
    def delete_calendar_event(title_or_id: str) -> str:
        """
        Delete a calendar event by title (partial match) or UID prefix.
        """
        try:
            from icalendar import Calendar
            if not CALENDAR_PATH.exists():
                return "No calendar found."

            cal = Calendar.from_ical(CALENDAR_PATH.read_bytes())
            new_cal = Calendar()
            new_cal.add("prodid", "-//Lavender AI//EN")
            new_cal.add("version", "2.0")

            removed = []
            for component in cal.walk():
                if component.name != "VEVENT":
                    continue
                summary = str(component.get("SUMMARY", ""))
                uid     = str(component.get("UID", ""))
                if (title_or_id.lower() in summary.lower() or
                        uid.startswith(title_or_id)):
                    removed.append(summary)
                else:
                    new_cal.add_component(component)

            if not removed:
                return f"No event found matching '{title_or_id}'."

            CALENDAR_PATH.write_bytes(new_cal.to_ical())
            return f"Deleted: {', '.join(removed)}"

        except Exception as e:
            return f"Delete failed: {e}"

    return [add_calendar_event, list_calendar_events, delete_calendar_event]
