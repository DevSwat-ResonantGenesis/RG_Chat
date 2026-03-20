"""
Temporal Thread Rebuilder (TTR)
=================================

Patch #46: Restores temporal continuity.

Ported from old backend: ResonantGraphAIV0.1/backend/fastapi_app/services/temporal_thread_engine.py
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class TemporalThreadEngine:
    """
    Temporal Thread Rebuilder
    
    Reconstructs chronological reasoning threads from chat history
    to enable temporal continuity and understanding.
    """
    
    def rebuild_thread(
        self,
        messages: List[Any],
        max_messages: int = 12
    ) -> str:
        """Reconstruct chronological reasoning thread from chat history."""
        try:
            msg_list = []
            for msg in messages:
                timestamp = None
                if hasattr(msg, 'created_at'):
                    timestamp = msg.created_at
                elif hasattr(msg, 'timestamp'):
                    timestamp = msg.timestamp
                elif isinstance(msg, dict):
                    timestamp_str = msg.get('timestamp') or msg.get('created_at')
                    if timestamp_str:
                        try:
                            if isinstance(timestamp_str, str):
                                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                            else:
                                timestamp = timestamp_str
                        except Exception:
                            timestamp = None
                
                content = None
                if hasattr(msg, 'content'):
                    content = msg.content
                elif isinstance(msg, dict):
                    content = msg.get('content') or msg.get('text')
                
                role = "user"
                if hasattr(msg, 'role'):
                    role = msg.role
                elif isinstance(msg, dict):
                    role = msg.get('role', 'user')
                
                if content and timestamp:
                    msg_list.append({
                        "timestamp": timestamp,
                        "role": role,
                        "content": content
                    })
            
            msg_list.sort(key=lambda m: m["timestamp"])
            recent_messages = msg_list[-max_messages:] if len(msg_list) > max_messages else msg_list
            
            timeline = []
            now = datetime.utcnow()
            
            for msg in recent_messages:
                ts = msg["timestamp"]
                role = msg["role"]
                content = msg["content"][:200]
                
                delta = now - ts if isinstance(ts, datetime) else timedelta(0)
                if delta.days > 0:
                    relative_time = f"{delta.days} day{'s' if delta.days > 1 else ''} ago"
                elif delta.seconds > 3600:
                    hours = delta.seconds // 3600
                    relative_time = f"{hours} hour{'s' if hours > 1 else ''} ago"
                elif delta.seconds > 60:
                    minutes = delta.seconds // 60
                    relative_time = f"{minutes} minute{'s' if minutes > 1 else ''} ago"
                else:
                    relative_time = "just now"
                
                if isinstance(ts, datetime):
                    ts_str = ts.strftime("%Y-%m-%d %H:%M")
                else:
                    ts_str = str(ts)
                
                timeline.append(f"[{ts_str}] ({relative_time}) {role}: {content}")
            
            return "\n".join(timeline) if timeline else "No temporal thread available."
            
        except Exception as e:
            logger.error(f"Error rebuilding temporal thread: {e}", exc_info=True)
            return "Error building temporal thread."
    
    def get_temporal_context(
        self,
        messages: List[Any],
        reference_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get temporal context information."""
        if not reference_time:
            reference_time = datetime.utcnow()
        
        try:
            timestamps = []
            for msg in messages:
                if hasattr(msg, 'created_at'):
                    timestamps.append(msg.created_at)
                elif hasattr(msg, 'timestamp'):
                    timestamps.append(msg.timestamp)
                elif isinstance(msg, dict):
                    ts = msg.get('timestamp') or msg.get('created_at')
                    if ts:
                        try:
                            if isinstance(ts, str):
                                timestamps.append(datetime.fromisoformat(ts.replace('Z', '+00:00')))
                            else:
                                timestamps.append(ts)
                        except Exception:
                            pass
            
            if not timestamps:
                return {
                    "message_count": 0,
                    "oldest": None,
                    "newest": None,
                    "span_days": 0
                }
            
            timestamps.sort()
            oldest = timestamps[0]
            newest = timestamps[-1]
            span = (newest - oldest).days if isinstance(newest, datetime) and isinstance(oldest, datetime) else 0
            
            return {
                "message_count": len(timestamps),
                "oldest": oldest.isoformat() if isinstance(oldest, datetime) else str(oldest),
                "newest": newest.isoformat() if isinstance(newest, datetime) else str(newest),
                "span_days": span
            }
            
        except Exception as e:
            logger.error(f"Error getting temporal context: {e}", exc_info=True)
            return {
                "message_count": 0,
                "error": str(e)
            }
    
    def get_system_prompt(self, messages: List[Any]) -> str:
        """Generate system prompt with temporal thread context."""
        thread = self.rebuild_thread(messages, max_messages=6)
        if thread and thread != "No temporal thread available." and thread != "Error building temporal thread.":
            return f"TEMPORAL CONTEXT (conversation timeline):\n{thread}"
        return ""


# Global instance
temporal_thread_engine = TemporalThreadEngine()
