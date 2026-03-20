"""Chat Service Models - Full old backend compatibility.

Includes:
- ResonantChat: Chat conversation container with Hash Sphere support
- ResonantChatMessage: Individual message with resonance hashing and 3D coordinates
- Conversation/Message: Legacy models for backwards compatibility
"""
from datetime import datetime
import uuid

from sqlalchemy import Column, DateTime, Float, String, Text, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from .db import Base


class ResonantChat(Base):
    """Chat conversation container with Hash Sphere support.
    
    Ported from old backend: ResonantGraphAIV0.1/backend/fastapi_app/models/governance/resonant_chat.py
    """
    __tablename__ = "resonant_chats"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), index=True, nullable=False)
    org_id = Column(UUID(as_uuid=True), index=True, nullable=False)
    # Hash Sphere universe isolation
    universe_id = Column(String(32), index=True, nullable=True)
    title = Column(String(255), default="New Chat")
    status = Column(String(50), default="active")  # active, archived, deleted
    
    # Shared Agent Support: agent_hash for shared memory
    agent_hash = Column(String(64), index=True, nullable=True)
    
    meta_data = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class ResonantChatMessage(Base):
    """Individual message in a chat with resonance hashing and 3D coordinates.
    
    Ported from old backend: ResonantGraphAIV0.1/backend/fastapi_app/models/governance/resonant_chat.py
    """
    __tablename__ = "resonant_chat_messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chat_id = Column(UUID(as_uuid=True), index=True, nullable=False)
    role = Column(String(20), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    
    # AI Provider tracking
    ai_provider = Column(String(50), nullable=True)  # chatgpt, claude, gemini, resonant-brain
    
    # Hash Sphere fields
    hash = Column(String(255), index=True, nullable=True)  # Resonance hash
    resonance_score = Column(Float, nullable=True)  # 0-1
    
    # 3D semantic space coordinates
    xyz_x = Column(Float, nullable=True)
    xyz_y = Column(Float, nullable=True)
    xyz_z = Column(Float, nullable=True)
    
    # Shared Agent Support
    agent_hash = Column(String(64), index=True, nullable=True)
    
    meta_data = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


# Legacy models for backwards compatibility
class Conversation(Base):
    """Legacy conversation model - use ResonantChat for new code."""
    __tablename__ = "conversations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), index=True, nullable=True)
    org_id = Column(UUID(as_uuid=True), index=True, nullable=True)
    title = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class Message(Base):
    """Legacy message model - use ResonantChatMessage for new code."""
    __tablename__ = "messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), index=True, nullable=False)
    role = Column(String(32), nullable=False)  # user / assistant / system
    content = Column(Text, nullable=False)
    
    # Hash Sphere fields (added for compatibility)
    hash = Column(String(255), index=True, nullable=True)
    resonance_score = Column(Float, nullable=True)
    xyz_x = Column(Float, nullable=True)
    xyz_y = Column(Float, nullable=True)
    xyz_z = Column(Float, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class AgentFeedback(Base):
    """User feedback for agent responses - persisted for learning.
    
    This table stores thumbs up/down feedback to:
    1. Track agent performance over time
    2. Influence agent selection for future responses
    3. Enable continuous improvement of agent quality
    """
    __tablename__ = "agent_feedback"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    message_id = Column(UUID(as_uuid=True), index=True, nullable=False)
    user_id = Column(UUID(as_uuid=True), index=True, nullable=False)
    agent_type = Column(String(100), index=True, nullable=False)  # reasoning, code, review, etc.
    feedback_type = Column(String(20), nullable=False)  # positive, negative
    
    # Context for learning
    task_preview = Column(String(500), nullable=True)  # First 500 chars of user message
    response_preview = Column(String(1000), nullable=True)  # First 1000 chars of response
    comment = Column(String(500), nullable=True)  # Optional user comment
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class AgentPerformanceScore(Base):
    """Aggregated agent performance scores - updated on each feedback.
    
    Stores computed scores for quick lookup during agent routing.
    """
    __tablename__ = "agent_performance_scores"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_type = Column(String(100), unique=True, index=True, nullable=False)
    
    # Aggregated stats
    positive_count = Column(Float, default=0)
    negative_count = Column(Float, default=0)
    total_count = Column(Float, default=0)
    satisfaction_rate = Column(Float, default=0.5)  # 0-1
    quality_score = Column(Float, default=0.5)  # Bayesian smoothed score
    
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class HallucinationSettings(Base):
    """Per-user hallucination detection settings - persisted across restarts."""
    __tablename__ = "hallucination_settings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(255), unique=True, index=True, nullable=False)
    system_prompt_grounding = Column(Float, default=1)  # 1=on, 0=off
    llm_as_judge = Column(Float, default=0)
    knowledge_base_check = Column(Float, default=0)
    settings_updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class KnowledgeBaseEntryDB(Base):
    """User-uploaded knowledge base entries for hallucination cross-referencing."""
    __tablename__ = "knowledge_base_entries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(255), index=True, nullable=False)
    title = Column(String(500), nullable=False)
    content = Column(Text, nullable=False)
    entry_type = Column(String(50), default="fact")  # fact, document, data, book_excerpt
    file_name = Column(String(500), nullable=True)  # original filename if uploaded
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
