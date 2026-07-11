"""
Navigation Detection Service
==============================

Extracts navigation intents (page routes, URLs) from user messages
and returns structured ToolResultData for the frontend to handle.

Also handles current-time tool results and time-only query detection.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import List, Optional
from zoneinfo import ZoneInfo

from ..routers.chat_models import ToolResultData


def extract_navigation_tool_results(user_message: str) -> List[ToolResultData]:
    msg = (user_message or "").strip()
    if not msg:
        return []

    msg_lower = msg.lower()
    if not any(k in msg_lower for k in ["open ", "go to ", "navigate ", "navigate to ", "visit "]):
        return []

    url_match = re.search(r"(https?://[^\s\)\]>'\"]+)", msg)
    if url_match:
        url = url_match.group(1).rstrip(".,;!?")
        return [ToolResultData(tool_name="navigation", success=True, result={"action": "navigate", "url": url})]

    path_match = re.search(r"(^|\s)(/[A-Za-z0-9\-_/]+)", msg)
    if path_match:
        path = path_match.group(2).rstrip(".,;!?")
        return [ToolResultData(tool_name="navigation", success=True, result={"action": "navigate", "url": path})]

    # Common internal page navigation
    page_routes: List[tuple[str, str, str]] = [
        (r"\bagents?\s+(?:os|page|panel)\b", "/agents", "agents"),
        (r"\bagents?\b", "/agents", "agents"),
        (r"\bagent\s+teams?\b", "/agent-teams", "agent-teams"),
        (r"\bteam\s+dashboard\b", "/agent-teams", "agent-teams"),
        (r"\bresonant\s+chat\b", "/resonant-chat-next", "resonant-chat"),
        (r"\bdashboard\b", "/dashboard", "dashboard"),
        (r"\bpricing\b", "/pricing", "pricing"),
        (r"\baccount\b", "/dashboard", "dashboard"),
        (r"\bide\b", "/ide", "ide"),
        (r"\bmarketplace\b", "/marketplace", "marketplace"),
        (r"\bcode\s*visual", "/code-visualizer", "code-visualizer"),
        (r"\bstate\s*physics\b", "/state-physics", "state-physics"),
        (r"\bresonant\s+memory\b", "/resonant-memory", "resonant-memory"),
        (r"\bmemory\s+(?:page|library|panel)\b", "/resonant-memory", "resonant-memory"),
        (r"\brabbit\b", "/rabbit", "rabbit"),
        (r"\bcommunity\b", "/rabbit", "rabbit"),
        (r"\bprofile\b", "/profile", "profile"),
        (r"\bsettings\b", "/profile", "profile"),
        (r"\bconnect.?profiles?\b", "/connect-profiles", "connect-profiles"),
        (r"\bintegrations?\b", "/connect-profiles", "connect-profiles"),
        (r"\bapi\s*keys?\b", "/connect-profiles", "connect-profiles"),
        (r"\bbuild\b", "/build", "build"),
        (r"\bproject\s*builder\b", "/build", "build"),
        (r"\bwallet\b", "/wallet", "wallet"),
    ]

    for pattern, path, page in page_routes:
        if re.search(pattern, msg_lower):
            return [
                ToolResultData(
                    tool_name="navigation",
                    success=True,
                    result={"action": "navigate", "url": path, "page": page},
                )
            ]

    return []


def extract_current_time_tool_results(user_message: str, default_timezone: Optional[str] = None) -> List[ToolResultData]:
    msg = (user_message or "").strip()
    if not msg:
        return []

    msg_lower = msg.lower()
    time_trigger = bool(
        re.search(r"\b(time\s+now|current\s+time|what\s+time|exact\s+time|time\s+in)\b", msg_lower)
        or ("time" in msg_lower and "now" in msg_lower)
        or ("time" in msg_lower and ("san francisco" in msg_lower or re.search(r"\bsf\b", msg_lower)))
    )
    if not time_trigger:
        return []

    tz: Optional[str] = None
    if "san francisco" in msg_lower or re.search(r"\bsf\b", msg_lower):
        tz = "America/Los_Angeles"
    elif "pacific" in msg_lower or "pst" in msg_lower or "pdt" in msg_lower:
        tz = "America/Los_Angeles"

    # Default to the client's browser timezone if user asks for "current time" without specifying location
    if not tz:
        tz = default_timezone or "America/Los_Angeles"

    try:
        now_local = datetime.now(ZoneInfo(tz))
    except Exception:
        tz = "America/Los_Angeles"
        now_local = datetime.now(ZoneInfo(tz))
    now_utc = datetime.utcnow()

    return [
        ToolResultData(
            tool_name="time",
            success=True,
            result={
                "action": "current_time",
                "timezone": tz,
                "iso": now_local.isoformat(),
                "local": now_local.strftime("%Y-%m-%d %H:%M:%S %Z"),
                "utc": now_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
            },
        )
    ]


def is_time_only_query(user_message: str) -> bool:
    msg = (user_message or "").strip().lower()
    if not msg:
        return False
    if not re.search(r"\b(time\s+now|current\s+time|what\s+time|exact\s+time)\b", msg):
        return False

    # If user is also asking for other time-sensitive info (events/weather/etc), don't short-circuit
    blockers = [
        "events", "weather", "restaurants", "things to do",
        "news", "price", "stock", "crypto", "bitcoin", "ethereum",
        # Business hours — "what time does X open/close"
        "open", "opens", "close", "closes", "closing", "opening",
        "store", "shop", "mall", "pharmacy", "target", "walmart",
        "costco", "safeway", "walgreens", "starbucks", "mcdonalds",
        # Transport schedules
        "flight", "train", "bus", "ferry", "departure", "arrival",
        "game", "match", "show", "concert", "movie",
    ]
    if any(b in msg for b in blockers):
        return any(msg == p for p in ["time now", "current time", "what time", "exact time"])
    return True
