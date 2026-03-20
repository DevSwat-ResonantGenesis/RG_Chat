"""
Export Router
==============

Provides export functionality for conversations in various formats.
"""
from __future__ import annotations

import logging
import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, Request, Query
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_session
from ..models import ResonantChat, ResonantChatMessage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/resonant-chat", tags=["export"])


class ExportFormat(str, Enum):
    JSON = "json"
    MARKDOWN = "markdown"
    TXT = "txt"
    HTML = "html"
    CSV = "csv"


class ExportRequest(BaseModel):
    format: ExportFormat = ExportFormat.JSON
    include_metadata: bool = True
    include_hashes: bool = False
    include_xyz: bool = False


def _format_timestamp(dt: Optional[datetime]) -> str:
    """Format datetime for export."""
    if dt:
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    return ""


def _export_to_json(
    chat: ResonantChat,
    messages: List[ResonantChatMessage],
    include_metadata: bool = True,
    include_hashes: bool = False,
    include_xyz: bool = False,
) -> str:
    """Export conversation to JSON format."""
    data = {
        "conversation": {
            "id": str(chat.id),
            "title": chat.title,
            "created_at": _format_timestamp(chat.created_at),
            "status": chat.status,
        },
        "messages": [],
        "export_info": {
            "exported_at": datetime.utcnow().isoformat(),
            "format": "json",
            "message_count": len(messages),
        }
    }
    
    for msg in messages:
        msg_data = {
            "role": msg.role,
            "content": msg.content,
            "timestamp": _format_timestamp(msg.created_at),
        }
        
        if include_metadata and msg.role == "assistant":
            msg_data["provider"] = msg.ai_provider
            msg_data["resonance_score"] = msg.resonance_score
        
        if include_hashes:
            msg_data["hash"] = msg.hash
        
        if include_xyz and msg.xyz_x is not None:
            msg_data["xyz"] = [msg.xyz_x, msg.xyz_y, msg.xyz_z]
        
        data["messages"].append(msg_data)
    
    return json.dumps(data, indent=2, ensure_ascii=False)


def _export_to_markdown(
    chat: ResonantChat,
    messages: List[ResonantChatMessage],
    include_metadata: bool = True,
    include_hashes: bool = False,
    include_xyz: bool = False,
) -> str:
    """Export conversation to Markdown format."""
    lines = []
    
    # Header
    lines.append(f"# {chat.title or 'Conversation'}")
    lines.append("")
    lines.append(f"**Created:** {_format_timestamp(chat.created_at)}")
    lines.append(f"**Messages:** {len(messages)}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Messages
    for msg in messages:
        if msg.role == "user":
            lines.append("## 👤 User")
        else:
            provider = msg.ai_provider or "AI"
            lines.append(f"## 🤖 Assistant ({provider})")
        
        lines.append("")
        lines.append(msg.content)
        lines.append("")
        
        if include_metadata and msg.role == "assistant":
            lines.append(f"*Resonance Score: {msg.resonance_score or 0:.4f}*")
            lines.append("")
        
        if include_hashes and msg.hash:
            lines.append(f"*Hash: `{msg.hash[:16]}...`*")
            lines.append("")
        
        lines.append(f"*{_format_timestamp(msg.created_at)}*")
        lines.append("")
        lines.append("---")
        lines.append("")
    
    # Footer
    lines.append("")
    lines.append(f"*Exported from Resonant Chat on {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}*")
    
    return "\n".join(lines)


def _export_to_txt(
    chat: ResonantChat,
    messages: List[ResonantChatMessage],
    include_metadata: bool = True,
    include_hashes: bool = False,
    include_xyz: bool = False,
) -> str:
    """Export conversation to plain text format."""
    lines = []
    
    # Header
    lines.append(f"Conversation: {chat.title or 'Untitled'}")
    lines.append(f"Created: {_format_timestamp(chat.created_at)}")
    lines.append(f"Messages: {len(messages)}")
    lines.append("=" * 60)
    lines.append("")
    
    # Messages
    for msg in messages:
        role = "USER" if msg.role == "user" else "ASSISTANT"
        lines.append(f"[{role}] {_format_timestamp(msg.created_at)}")
        lines.append("-" * 40)
        lines.append(msg.content)
        lines.append("")
        
        if include_metadata and msg.role == "assistant":
            lines.append(f"Provider: {msg.ai_provider or 'unknown'}")
            lines.append(f"Resonance: {msg.resonance_score or 0:.4f}")
            lines.append("")
        
        lines.append("")
    
    # Footer
    lines.append("=" * 60)
    lines.append(f"Exported: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    
    return "\n".join(lines)


def _export_to_html(
    chat: ResonantChat,
    messages: List[ResonantChatMessage],
    include_metadata: bool = True,
    include_hashes: bool = False,
    include_xyz: bool = False,
) -> str:
    """Export conversation to HTML format."""
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{chat.title or 'Conversation'} - Resonant Chat Export</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .message {{
            background: white;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .message.user {{
            border-left: 4px solid #667eea;
        }}
        .message.assistant {{
            border-left: 4px solid #764ba2;
        }}
        .role {{
            font-weight: bold;
            margin-bottom: 10px;
            color: #333;
        }}
        .content {{
            white-space: pre-wrap;
            line-height: 1.6;
        }}
        .metadata {{
            font-size: 12px;
            color: #666;
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid #eee;
        }}
        .footer {{
            text-align: center;
            color: #666;
            font-size: 12px;
            margin-top: 20px;
        }}
        code {{
            background: #f0f0f0;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: monospace;
        }}
        pre {{
            background: #2d2d2d;
            color: #f8f8f2;
            padding: 15px;
            border-radius: 8px;
            overflow-x: auto;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{chat.title or 'Conversation'}</h1>
        <p>Created: {_format_timestamp(chat.created_at)}</p>
        <p>Messages: {len(messages)}</p>
    </div>
"""
    
    for msg in messages:
        role_class = "user" if msg.role == "user" else "assistant"
        role_label = "👤 User" if msg.role == "user" else f"🤖 Assistant ({msg.ai_provider or 'AI'})"
        
        # Escape HTML in content
        content = msg.content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        # Convert code blocks
        import re
        content = re.sub(
            r'```(\w+)?\n(.*?)```',
            r'<pre><code>\2</code></pre>',
            content,
            flags=re.DOTALL
        )
        content = re.sub(r'`([^`]+)`', r'<code>\1</code>', content)
        # Convert newlines
        content = content.replace("\n", "<br>")
        
        html += f"""
    <div class="message {role_class}">
        <div class="role">{role_label}</div>
        <div class="content">{content}</div>
"""
        
        if include_metadata and msg.role == "assistant":
            html += f"""
        <div class="metadata">
            Resonance Score: {msg.resonance_score or 0:.4f}
            | {_format_timestamp(msg.created_at)}
        </div>
"""
        else:
            html += f"""
        <div class="metadata">{_format_timestamp(msg.created_at)}</div>
"""
        
        html += "    </div>\n"
    
    html += f"""
    <div class="footer">
        Exported from Resonant Chat on {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
    </div>
</body>
</html>
"""
    
    return html


def _export_to_csv(
    chat: ResonantChat,
    messages: List[ResonantChatMessage],
    include_metadata: bool = True,
    include_hashes: bool = False,
    include_xyz: bool = False,
) -> str:
    """Export conversation to CSV format."""
    import csv
    import io
    
    output = io.StringIO()
    
    # Define columns
    columns = ["timestamp", "role", "content"]
    if include_metadata:
        columns.extend(["provider", "resonance_score"])
    if include_hashes:
        columns.append("hash")
    if include_xyz:
        columns.extend(["xyz_x", "xyz_y", "xyz_z"])
    
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    
    for msg in messages:
        row = {
            "timestamp": _format_timestamp(msg.created_at),
            "role": msg.role,
            "content": msg.content.replace("\n", "\\n"),  # Escape newlines
        }
        
        if include_metadata:
            row["provider"] = msg.ai_provider or ""
            row["resonance_score"] = msg.resonance_score or ""
        
        if include_hashes:
            row["hash"] = msg.hash or ""
        
        if include_xyz:
            row["xyz_x"] = msg.xyz_x if msg.xyz_x is not None else ""
            row["xyz_y"] = msg.xyz_y if msg.xyz_y is not None else ""
            row["xyz_z"] = msg.xyz_z if msg.xyz_z is not None else ""
        
        writer.writerow(row)
    
    return output.getvalue()


@router.post("/export/{chat_id}")
async def export_conversation(
    chat_id: str,
    export_request: ExportRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """
    Export a conversation in the specified format.
    
    Supported formats:
    - json: Structured JSON with all data
    - markdown: Human-readable Markdown
    - txt: Plain text
    - html: Styled HTML page
    - csv: Spreadsheet-compatible CSV
    """
    user_id = request.headers.get("x-user-id")
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID required")
    
    # Get chat
    try:
        result = await session.execute(
            select(ResonantChat).where(ResonantChat.id == UUID(chat_id))
        )
        chat = result.scalar_one_or_none()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid chat ID format")
    
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    if chat.user_id != UUID(user_id):
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Get messages
    result = await session.execute(
        select(ResonantChatMessage)
        .where(ResonantChatMessage.chat_id == UUID(chat_id))
        .order_by(ResonantChatMessage.created_at.asc())
    )
    messages = result.scalars().all()
    
    # Export based on format
    format_handlers = {
        ExportFormat.JSON: (_export_to_json, "application/json", "json"),
        ExportFormat.MARKDOWN: (_export_to_markdown, "text/markdown", "md"),
        ExportFormat.TXT: (_export_to_txt, "text/plain", "txt"),
        ExportFormat.HTML: (_export_to_html, "text/html", "html"),
        ExportFormat.CSV: (_export_to_csv, "text/csv", "csv"),
    }
    
    handler, content_type, extension = format_handlers[export_request.format]
    
    content = handler(
        chat=chat,
        messages=messages,
        include_metadata=export_request.include_metadata,
        include_hashes=export_request.include_hashes,
        include_xyz=export_request.include_xyz,
    )
    
    # Generate filename
    title_slug = (chat.title or "conversation")[:30].lower()
    title_slug = "".join(c if c.isalnum() else "_" for c in title_slug)
    filename = f"resonant_chat_{title_slug}_{chat_id[:8]}.{extension}"
    
    return Response(
        content=content,
        media_type=content_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        }
    )


@router.get("/export/{chat_id}/json")
async def export_json(
    chat_id: str,
    request: Request,
    include_metadata: bool = True,
    session: AsyncSession = Depends(get_session),
):
    """Quick export to JSON format."""
    export_request = ExportRequest(
        format=ExportFormat.JSON,
        include_metadata=include_metadata,
    )
    return await export_conversation(chat_id, export_request, request, session)


@router.get("/export/{chat_id}/markdown")
async def export_markdown(
    chat_id: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Quick export to Markdown format."""
    export_request = ExportRequest(format=ExportFormat.MARKDOWN)
    return await export_conversation(chat_id, export_request, request, session)


@router.get("/export/{chat_id}/html")
async def export_html(
    chat_id: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Quick export to HTML format."""
    export_request = ExportRequest(format=ExportFormat.HTML)
    return await export_conversation(chat_id, export_request, request, session)
