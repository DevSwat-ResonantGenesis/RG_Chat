"""
Google Calendar Integration Skill
===================================

Access Google Calendar: list upcoming events, create events,
check schedule, and manage meetings.

Requires: Google Calendar OAuth2 refresh token
API Docs: https://developers.google.com/calendar/api/v3/reference
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import httpx

from .base import BaseIntegrationSkill

logger = logging.getLogger(__name__)

CALENDAR_API = "https://www.googleapis.com/calendar/v3"

# Regex fragment matching a time like "8:00 PM", "8PM", "8 pm", "11:30AM"
_TIME = r"\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm)"

# Common venue/location suffix words for splitting "at Location NextTitle"
_LOCATION_SUFFIXES = {
    "center", "club", "gallery", "theater", "theatre",
    "arena", "hall", "park", "square", "hotel", "stadium",
    "museum", "auditorium", "coliseum", "building", "room",
    "garden", "gardens", "lounge", "bar", "restaurant",
    "francisco", "angeles", "york", "chicago", "boston",
    "fillmore", "freight", "palace", "pub", "cafe",
}

# Words that are preamble / not part of the event title
_PREAMBLE_RE = re.compile(
    r"^(?:can\s+(?:u|you)\s+)?"
    r"(?:please\s+)?"
    r"(?:also\s+)?"
    r"(?:oh\s+and\s+)?"
    r"(?:add|put|create|schedule|book|set\s+up)\s+"
    r"(?:another\s+|an?\s+|one\s+more\s+)?"
    r"(?:new\s+)?"
    r"(?:event\s+|meeting\s+|reminder\s+)?"
    r"(?:called\s+|named\s+|titled\s+)?"
    r"(?:to\s+my\s+(?:google\s+)?calendar\s+)?"
    r"(?:on\s+my\s+(?:google\s+)?calendar\s+)?",
    re.IGNORECASE,
)


class GoogleCalendarSkill(BaseIntegrationSkill):
    skill_id = "google_calendar"
    skill_name = "Google Calendar"
    api_key_names = [
        "google-calendar", "google_calendar", "gcalendar",
        "google-calendar-token",
    ]
    intent_keywords = [
        "google calendar", "my calendar", "my schedule",
        "upcoming events", "upcoming meetings", "next meeting",
        "list events", "show events", "show my events",
        "create event", "schedule meeting", "add event",
        "calendar events", "what's on my calendar",
        "free time", "availability", "am i free",
        "book meeting", "schedule call",
        "tomorrow's schedule", "today's schedule",
        "this week's events", "next week's events",
        "access calendar", "connect calendar", "open calendar",
    ]

    # ── action detection ──────────────────────────────────────────────

    def _detect_action(self, message: str) -> str:
        msg = message.lower()
        create_patterns = [
            "create event", "schedule meeting", "add event", "add events",
            "add this event", "add these event", "add this to",
            "add all this", "add all these", "add them to",
            "book meeting", "schedule call", "new event",
            "put on my calendar", "put these on", "put this on",
            "add to my calendar", "add to calendar",
        ]
        if any(k in msg for k in create_patterns):
            return "create_event"
        if re.search(r'\badd\b.{0,40}\b(calendar|event)', msg):
            return "create_event"
        return "list_events"

    # ── time parsing ──────────────────────────────────────────────────

    @staticmethod
    def _parse_time(time_str: str, date) -> datetime:
        """Parse '8:00 PM', '8PM', '8 pm', '11:30AM' into a datetime."""
        ts = time_str.strip().upper()
        # Normalize: ensure space before AM/PM → "8PM" → "8 PM"
        ts = re.sub(r"(\d)\s*(AM|PM)", r"\1 \2", ts)

        pacific = timezone(timedelta(hours=-7))
        for fmt in ("%I:%M %p", "%I %p"):
            try:
                t = datetime.strptime(ts, fmt).time()
                return datetime.combine(date, t, tzinfo=pacific)
            except ValueError:
                continue
        # Last resort
        t = datetime.strptime("8:00 PM", "%I:%M %p").time()
        return datetime.combine(date, t, tzinfo=pacific)

    # ── parse multiple events from natural-language message ───────────

    @staticmethod
    def _split_location_title(text: str):
        """Split 'Chase Center Charlie Hunter Trio' into
        (title='Charlie Hunter Trio', location='Chase Center').

        Uses common venue suffix words to find the boundary.
        """
        words = text.split()
        if len(words) <= 1:
            return text, text

        best_split = 2  # default: first 2 words = location
        for i, word in enumerate(words):
            if word.lower().rstrip(".,;:") in _LOCATION_SUFFIXES:
                best_split = i + 1
                break

        if best_split >= len(words):
            best_split = max(1, len(words) - 1)

        location = " ".join(words[:best_split])
        title = " ".join(words[best_split:])
        return title, location

    def _parse_events_from_message(self, message: str) -> List[Dict[str, Any]]:
        """Parse multiple events from a message.

        Handles three formats:
        1. Structured:  'Nine Inch Nails - 8:00 PM at Chase Center'
        2. Run-on:      '...Chase Center Charlie Hunter Trio - 6:00 PM at The Freight'
        3. Natural-lang: 'add event to my calendar for 10:30 pm to buy beer'
        4. Short times:  'from 1am to 4am work on project'
        """
        events: List[Dict[str, Any]] = []
        today = datetime.now(timezone(timedelta(hours=-7))).date()

        # ── Phase 1: Structured format with " - TIME" ────────────────
        time_re = re.compile(
            r"\s+-\s+(" + _TIME + r")"
            r"(?:\s+-\s+(" + _TIME + r"))?",
            re.IGNORECASE,
        )
        markers = list(time_re.finditer(message))

        if markers:
            prev_end = 0
            for idx, m in enumerate(markers):
                before_text = message[prev_end : m.start()].strip()
                title = before_text

                if idx > 0 and before_text.lower().startswith("at "):
                    after_at = before_text[3:]
                    title, prev_location = self._split_location_title(after_at)
                    if events:
                        events[-1]["location"] = prev_location
                elif idx == 0:
                    preamble = re.match(
                        r"(?:can\s+(?:u|you)\s+)?(?:add|put|create|schedule)\s+"
                        r"(?:this|these|all\s+this|all\s+these|the)?\s*"
                        r"(?:events?\s+)?(?:to\s+)?(?:my\s+)?(?:google\s+)?(?:calendar\s+)?",
                        title,
                        re.IGNORECASE,
                    )
                    if preamble:
                        title = title[preamble.end() :].strip()

                start_time_str = m.group(1).strip()
                end_time_str = m.group(2).strip() if m.group(2) else None

                if title:
                    start_dt = self._parse_time(start_time_str, today)
                    end_dt = (
                        self._parse_time(end_time_str, today)
                        if end_time_str
                        else start_dt + timedelta(hours=2)
                    )
                    events.append(
                        {
                            "title": title,
                            "start": start_dt,
                            "end": end_dt,
                            "location": "",
                        }
                    )
                prev_end = m.end()

            # Last event's location
            remaining = message[prev_end:].strip()
            if remaining.lower().startswith("at ") and events:
                events[-1]["location"] = remaining[3:].strip()

            return events

        # ── Phase 2: Natural language — "for 10:30 pm to buy beer" ────
        nl_result = self._parse_natural_language(message, today)
        if nl_result:
            return nl_result

        return []

    def _parse_natural_language(
        self, message: str, today
    ) -> List[Dict[str, Any]]:
        """Parse natural-language event requests like:
        - 'add event to my calendar for 10:30 pm to buy beer'
        - 'schedule dinner with Sarah at 7:00 PM'
        - 'from 1am to 4am work on project'
        - 'add event at 11pm night club'
        """
        # Find time with preposition: "for 10:30pm", "at 1am", "from 2pm"
        time_re = re.compile(
            r"(?:for|at|@|from)\s+(" + _TIME + r")"
            r"(?:\s*(?:-|to|until|till)\s+(" + _TIME + r"))?",
            re.IGNORECASE,
        )
        m = time_re.search(message)

        # Also try standalone time
        if not m:
            standalone_re = re.compile(
                r"(" + _TIME + r")"
                r"(?:\s*(?:-|to|until|till)\s+(" + _TIME + r"))?",
                re.IGNORECASE,
            )
            m = standalone_re.search(message)

        if not m:
            return []

        start_str = m.group(1).strip()
        end_str = m.group(2).strip() if m.group(2) else None
        start_dt = self._parse_time(start_str, today)
        end_dt = (
            self._parse_time(end_str, today)
            if end_str
            else start_dt + timedelta(hours=1)
        )

        # ── Extract title ──────────────────────────────────────────────
        before_time = message[: m.start()].strip()
        after_time = message[m.end() :].strip()
        title = ""
        location = ""

        # 1) Check text AFTER the time: "to buy beer", "night club", "work on project"
        if after_time:
            cleaned = re.sub(
                r"^(?:to\s+|for\s+|[-–:]\s*|i\s+need\s+to\s+)",
                "",
                after_time,
                flags=re.IGNORECASE,
            ).strip()
            # Strip trailing calendar references
            cleaned = re.sub(
                r"\s+(?:to|on|in)\s+(?:my\s+)?(?:google\s+)?calendar\s*$",
                "",
                cleaned,
                flags=re.IGNORECASE,
            ).strip()
            if cleaned:
                title = cleaned

        # 2) If no title after time, check text BEFORE the time
        if not title and before_time:
            stripped = _PREAMBLE_RE.sub("", before_time).strip()
            # Remove trailing prepositions left over
            stripped = re.sub(
                r"\s+(?:for|at|@|from)\s*$", "", stripped, flags=re.IGNORECASE
            ).strip()
            if stripped:
                title = stripped

        # 3) Check if there's a location (after "at" in the title)
        if title:
            at_match = re.search(r"\bat\s+(.+)", title, re.IGNORECASE)
            if at_match:
                potential_loc = at_match.group(1).strip()
                if len(potential_loc.split()) <= 4:
                    location = potential_loc
                    title = title[: at_match.start()].strip()

        if not title:
            title = "New Event"
        else:
            # Title-case if all lowercase
            if title == title.lower():
                title = title.title()

        return [
            {
                "title": title,
                "start": start_dt,
                "end": end_dt,
                "location": location,
            }
        ]

    def _parse_event_details(self, message: str) -> Dict[str, Any]:
        """Extract single event title from natural language (legacy fallback)."""
        title = ""
        title_match = re.search(
            r"(?:titled?|called|named)\s+['\"]([^'\"]+)['\"]",
            message,
            re.IGNORECASE,
        )
        if title_match:
            title = title_match.group(1)
        else:
            t2 = re.search(
                r"(?:create event|schedule meeting|add event|book meeting)"
                r"\s+(.+?)(?:\s+(?:on|at|for|tomorrow|today|$))",
                message,
                re.IGNORECASE,
            )
            if t2:
                title = t2.group(1).strip().strip("'\"")
        if not title:
            title = "New Event"
        return {"title": title}

    # ── OAuth refresh ─────────────────────────────────────────────────

    async def _refresh_access_token(self, refresh_token: str) -> str:
        """Exchange a Google OAuth refresh token for a fresh access token."""
        client_id = os.getenv("GOOGLE_CLIENT_ID")
        client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
        if not client_id or not client_secret:
            raise ValueError(
                "GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET not configured"
            )

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                },
            )
            resp.raise_for_status()
            return resp.json()["access_token"]

    # ── main execute ──────────────────────────────────────────────────

    async def execute(
        self, message: str, user_id: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        token = self.get_credentials(context)
        if not token:
            return self._no_credentials_error()

        # If token looks like a refresh token (starts with 1//), exchange it
        if token.startswith("1//"):
            try:
                token = await self._refresh_access_token(token)
            except Exception as e:
                logger.error(f"Google Calendar OAuth refresh failed: {e}")
                return {
                    "success": False,
                    "action": "google_calendar",
                    "error": (
                        f"Failed to refresh Google Calendar token: {e}. "
                        "Please reconnect in **Settings \u2192 Connect Profiles**."
                    ),
                }

        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }
        action = self._detect_action(message)

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                if action == "create_event":
                    return await self._create_events(client, headers, message)
                else:
                    return await self._list_events(client, headers)

        except httpx.HTTPStatusError as e:
            if e.response.status_code in (401, 403):
                return {
                    "success": False,
                    "action": "google_calendar",
                    "error": (
                        f"Google Calendar access denied (HTTP {e.response.status_code}). "
                        f"Detail: {e.response.text[:200]}"
                    ),
                }
            return {
                "success": False,
                "action": "google_calendar",
                "error": f"Calendar API error ({e.response.status_code}): {e.response.text[:200]}",
            }
        except Exception as e:
            return {
                "success": False,
                "action": "google_calendar",
                "error": str(e)[:300],
            }

    # ── create (batch-aware) ──────────────────────────────────────────

    async def _create_events(
        self, client: httpx.AsyncClient, headers: Dict, message: str
    ) -> Dict[str, Any]:
        """Create one or more events from the user message."""
        parsed = self._parse_events_from_message(message)

        if not parsed:
            # Last-resort fallback — should rarely hit now
            details = self._parse_event_details(message)
            now = datetime.now(timezone.utc)
            start = now + timedelta(hours=1)
            end = start + timedelta(hours=1)

            event_body = {
                "summary": details["title"],
                "start": {"dateTime": start.isoformat(), "timeZone": "UTC"},
                "end": {"dateTime": end.isoformat(), "timeZone": "UTC"},
            }
            resp = await client.post(
                f"{CALENDAR_API}/calendars/primary/events",
                headers={**headers, "Content-Type": "application/json"},
                json=event_body,
            )
            resp.raise_for_status()
            event = resp.json()
            link = event.get("htmlLink", "")
            summary = (
                f"\u2705 **Event Created: {details['title']}**\n\n"
                f"- **When**: {start.strftime('%Y-%m-%d %H:%M')} UTC\n"
                f"- **Duration**: 1 hour\n"
            )
            if link:
                summary += f"- [Open in Calendar]({link})\n"
            return {
                "success": True,
                "action": "create_event",
                "summary": summary,
                "event": event,
            }

        # Batch create
        created = []
        failed = []
        for ev in parsed:
            event_body = {
                "summary": ev["title"],
                "start": {
                    "dateTime": ev["start"].isoformat(),
                    "timeZone": "America/Los_Angeles",
                },
                "end": {
                    "dateTime": ev["end"].isoformat(),
                    "timeZone": "America/Los_Angeles",
                },
            }
            if ev.get("location"):
                event_body["location"] = ev["location"]

            try:
                resp = await client.post(
                    f"{CALENDAR_API}/calendars/primary/events",
                    headers={**headers, "Content-Type": "application/json"},
                    json=event_body,
                )
                resp.raise_for_status()
                created.append({"title": ev["title"], "event": resp.json()})
            except Exception as exc:
                failed.append({"title": ev["title"], "error": str(exc)[:100]})

        # Build summary
        lines = []
        if created:
            lines.append(
                f"\u2705 **{len(created)} event(s) added to Google Calendar:**\n"
            )
            for c in created:
                link = c["event"].get("htmlLink", "")
                start_raw = (
                    c["event"]
                    .get("start", {})
                    .get("dateTime", "")[:16]
                    .replace("T", " ")
                )
                loc = c["event"].get("location", "")
                line = f"- \U0001f4c5 **{c['title']}** \u2014 {start_raw}"
                if loc:
                    line += f" \u2014 \U0001f4cd {loc}"
                if link:
                    line += f" \u2014 [Open]({link})"
                lines.append(line)

        if failed:
            lines.append(f"\n\u274c **{len(failed)} event(s) failed:**\n")
            for f_ev in failed:
                lines.append(f"- {f_ev['title']}: {f_ev['error']}")

        return {
            "success": len(created) > 0,
            "action": "create_events",
            "summary": "\n".join(lines),
            "created_count": len(created),
            "failed_count": len(failed),
        }

    # ── list ──────────────────────────────────────────────────────────

    async def _list_events(
        self, client: httpx.AsyncClient, headers: Dict
    ) -> Dict[str, Any]:
        """List upcoming events for the next 7 days."""
        now = datetime.now(timezone.utc)
        time_min = now.isoformat()
        time_max = (now + timedelta(days=7)).isoformat()

        resp = await client.get(
            f"{CALENDAR_API}/calendars/primary/events",
            headers=headers,
            params={
                "timeMin": time_min,
                "timeMax": time_max,
                "maxResults": 25,
                "singleEvents": "true",
                "orderBy": "startTime",
            },
        )
        resp.raise_for_status()
        events = resp.json().get("items", [])

        if not events:
            summary = (
                "**No upcoming events** in the next 7 days. Your calendar is clear!"
            )
        else:
            summary = (
                f"**Upcoming Events** (next 7 days, {len(events)} events)\n\n"
            )
            for ev in events[:25]:
                start = ev.get("start", {})
                start_dt = (
                    start.get("dateTime", start.get("date", ""))[:16].replace(
                        "T", " "
                    )
                )
                title = ev.get("summary", "No title")
                location = ev.get("location", "")
                link = ev.get("htmlLink", "")
                summary += f"- \U0001f4c5 **{title}** \u2014 {start_dt}"
                if location:
                    summary += f" \u2014 \U0001f4cd {location}"
                if link:
                    summary += f" \u2014 [Open]({link})"
                summary += "\n"

        return {
            "success": True,
            "action": "list_events",
            "summary": summary,
            "events": events,
        }
