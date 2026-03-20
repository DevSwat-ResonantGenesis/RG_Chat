from typing import List, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .db import get_session
from .models import Conversation, Message


router = APIRouter(prefix="/chat", tags=["chat"])


class ConversationCreate(BaseModel):
    title: Optional[str] = None


class ConversationResponse(BaseModel):
    id: str
    user_id: Optional[str]
    title: Optional[str]

    class Config:
        from_attributes = True

    @classmethod
    def from_orm(cls, obj):
        return cls(
            id=str(obj.id),
            user_id=str(obj.user_id) if obj.user_id else None,
            title=obj.title,
        )


class MessageCreate(BaseModel):
    role: str
    content: str


class MessageResponse(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str

    class Config:
        from_attributes = True

    @classmethod
    def from_orm(cls, obj):
        return cls(
            id=str(obj.id),
            conversation_id=str(obj.conversation_id),
            role=obj.role,
            content=obj.content,
        )


class ChatContextResponse(BaseModel):
    conversation: ConversationResponse
    messages: List[MessageResponse]
    memories: List[dict]


@router.post("/conversations", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    payload: ConversationCreate,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    user_id = request.headers.get("x-user-id")
    conv = Conversation(user_id=user_id, title=payload.title)
    session.add(conv)
    await session.commit()
    await session.refresh(conv)
    return ConversationResponse.from_orm(conv)


@router.get("/conversations", response_model=List[ConversationResponse])
async def list_conversations(
    request: Request,
    limit: int = 50,
    session: AsyncSession = Depends(get_session),
):
    """List conversations for the current user."""
    user_id = request.headers.get("x-user-id")
    
    stmt = select(Conversation).order_by(Conversation.created_at.desc())
    if user_id:
        stmt = stmt.where(Conversation.user_id == user_id)
    
    result = await session.execute(stmt.limit(limit))
    convs = result.scalars().all()
    return [ConversationResponse.from_orm(c) for c in convs]


@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(conversation_id: str, session: AsyncSession = Depends(get_session)):
    result = await session.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    conv = result.scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return ConversationResponse.from_orm(conv)


@router.post(
    "/conversations/{conversation_id}/messages",
    response_model=MessageResponse,
    status_code=status.HTTP_201_CREATED,
)
async def add_message(
    conversation_id: str,
    payload: MessageCreate,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    # Ensure conversation exists
    result = await session.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    conv = result.scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    msg = Message(
        conversation_id=conversation_id,
        role=payload.role,
        content=payload.content,
    )
    session.add(msg)
    await session.commit()
    await session.refresh(msg)

    # Fire-and-forget calls to memory and cognitive services for enrichment
    user_id = request.headers.get("x-user-id")

    async with httpx.AsyncClient() as client:
        # Ingest message into memory service
        try:
            await client.post(
                "http://memory_service:8000/memory/ingest",
                json={
                    "chat_id": str(conversation_id),
                    "user_id": user_id,
                    "source": "chat",
                    "content": payload.content,
                },
                timeout=2.0,
            )
        except httpx.RequestError:
            # Best-effort; do not fail chat on memory issues
            pass

        # Record a cognitive tick for this message
        try:
            await client.post(
                "http://cognitive_service:8000/cognitive/ticks",
                json={
                    "agent_id": None,
                    "kind": "chat_message",
                    "payload": payload.content,
                },
                timeout=2.0,
            )
        except httpx.RequestError:
            # Also best-effort; avoid impacting chat latency
            pass

    return MessageResponse.from_orm(msg)


@router.get(
    "/conversations/{conversation_id}/messages",
    response_model=List[MessageResponse],
)
async def list_messages(conversation_id: str, session: AsyncSession = Depends(get_session)):
    # Ensure conversation exists
    result = await session.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    conv = result.scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    result = await session.execute(
        select(Message).where(Message.conversation_id == conversation_id).order_by(Message.created_at)
    )
    messages = result.scalars().all()
    return [MessageResponse.from_orm(m) for m in messages]


@router.get(
    "/conversations/{conversation_id}/context",
    response_model=ChatContextResponse,
)
async def get_conversation_context(
    conversation_id: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    # Fetch conversation
    result = await session.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    conv = result.scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Fetch messages
    result = await session.execute(
        select(Message).where(Message.conversation_id == conversation_id).order_by(Message.created_at)
    )
    messages = result.scalars().all()

    # Fetch memories from memory_service (best-effort)
    user_id = request.headers.get("x-user-id")
    memories: List[dict] = []
    try:
        async with httpx.AsyncClient() as client:
            mem_resp = await client.post(
                "http://memory_service:8000/memory/retrieve",
                json={
                    "chat_id": str(conversation_id),
                    "user_id": user_id,
                    "query": "",
                    "limit": 10,
                },
                timeout=3.0,
            )
        if mem_resp.status_code == 200:
            memories = mem_resp.json()
    except httpx.RequestError:
        # If memory service is unavailable, return empty memories list
        memories = []

    return ChatContextResponse(
        conversation=ConversationResponse.from_orm(conv),
        messages=[MessageResponse.from_orm(m) for m in messages],
        memories=memories,
    )


@router.get("/health")
async def health():
    return {"service": "chat", "status": "ok"}
