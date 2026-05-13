"""
Resonant Chat — Request/Response Models
=========================================

All Pydantic models used by resonant_chat.py endpoints.
Extracted to reduce file size and improve navigability.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


# ── Core Message Pipeline ──

class SendMessageRequest(BaseModel):
    message: str
    chat_id: Optional[str] = None
    preferred_provider: Optional[str] = None
    preferred_model: Optional[str] = None
    agent_hash: Optional[str] = None
    teamId: Optional[str] = None
    attached_files: Optional[List[Dict[str, Any]]] = None
    images: Optional[List[Dict[str, Any]]] = None  # Base64 images for vision models: [{type, data, name}]
    code_selection: Optional[Dict[str, Any]] = None
    isolate_anchors: Optional[bool] = False
    enabled_tool_ids: Optional[List[str]] = None  # Frontend tool toggles — overrides server defaults when provided
    # IDE Chat Integration
    execute_mode: Optional[bool] = False  # When True: skip explanations, return structured code changes
    project_context: Optional[Dict[str, Any]] = None  # IDE project context (files, structure)


class MessageData(BaseModel):
    id: str
    role: str
    content: str
    timestamp: str
    aiProvider: Optional[str] = None
    llmProvider: Optional[str] = None
    model: Optional[str] = None
    preferredProvider: Optional[str] = None
    wasFallback: Optional[bool] = None
    fallbackChain: Optional[List[Dict[str, Any]]] = None
    tokenUsage: Optional[Dict[str, Any]] = None
    hash: Optional[str] = None
    resonanceScore: Optional[float] = None
    xyz: Optional[List[float]] = None


class GeneratedImageData(BaseModel):
    """Generated image data for response."""
    url: Optional[str] = None
    base64_data: Optional[str] = None
    revised_prompt: Optional[str] = None
    model: str = "dall-e-3"
    size: str = "1024x1024"


class WebSearchResultData(BaseModel):
    """Web search result data for response."""
    title: str
    url: str
    snippet: str
    source: str = "unknown"


class ToolResultData(BaseModel):
    tool_name: str
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ResonantChatResponse(BaseModel):
    message: MessageData
    anchors: List[str] = []
    hash: Optional[str] = None
    resonanceScore: float = 0.5
    aiProvider: str = "unknown"
    llmProvider: Optional[str] = None
    memoryUpdated: bool = False
    chatId: str
    evidenceGraph: Optional[Dict[str, Any]] = None
    generatedImages: Optional[List[GeneratedImageData]] = None
    webSearchResults: Optional[List[WebSearchResultData]] = None
    toolResults: Optional[List[ToolResultData]] = None


class ConversationMessageRequest(BaseModel):
    """Compatibility request model for adding a message to a conversation."""
    role: str
    content: str


class SaveAgenticRequest(BaseModel):
    """Request model for saving agentic-chat messages into resonant pipeline."""
    user_message: str
    assistant_response: str
    chat_id: Optional[str] = None  # Existing resonant chat ID, or null to create new
    tool_results: Optional[List[Dict[str, Any]]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    model: Optional[str] = None
    tokens_used: Optional[int] = 0
    loops: Optional[int] = 0


class CreateChatRequest(BaseModel):
    title: Optional[str] = None
    agent_hash: Optional[str] = None


class CreateChatResponse(BaseModel):
    chatId: str
    title: str


# ── Hallucination & Knowledge Base ──

class HallucinationSettingsRequest(BaseModel):
    """Request to update hallucination detection settings."""
    system_prompt_grounding: Optional[bool] = None
    llm_as_judge: Optional[bool] = None
    knowledge_base_check: Optional[bool] = None


class KnowledgeBaseAddRequest(BaseModel):
    """Request to add a knowledge base entry."""
    title: str
    content: str
    entry_type: str = "fact"  # 'fact', 'document', 'data', 'book_excerpt'


# ── Internal Route Query ──

class InternalRouteQueryRequest(BaseModel):
    """Request for internal route-query endpoint."""
    message: str
    context: Optional[List[Dict[str, Any]]] = None
    model: Optional[str] = None
    preferred_provider: Optional[str] = None
    user_api_keys: Optional[Dict[str, str]] = None


# ── Phase 5 Endpoints ──

class FeedbackRequest(BaseModel):
    message_id: str
    is_positive: bool
    agent_type: str = ""
    comment: Optional[str] = None


class CreateChainRequest(BaseModel):
    name: str
    description: str
    steps: List[Dict[str, Any]]


class ExecuteChainRequest(BaseModel):
    chain_id: str
    task: str
    context: List[Dict[str, Any]] = []


class ExecuteCodeRequest(BaseModel):
    code: str
    language: Optional[str] = None
    test_input: str = ""


class AnalyzeRequest(BaseModel):
    response: str
    task: str = ""
    agent_type: str = ""


class ValidateRequest(BaseModel):
    response: str
    task: str
    agent_type: str
    context: List[Dict[str, Any]] = []


class VotingRequest(BaseModel):
    task: str
    context: List[Dict[str, Any]] = []
    candidate_agents: Optional[List[str]] = None
    voter_agents: Optional[List[str]] = None


class ProjectContextRequest(BaseModel):
    project_name: str


class ChunkingInfoRequest(BaseModel):
    text: str


class ProcessChunkedRequest(BaseModel):
    text: str
    task_prompt: str = "Process and summarize this content"
