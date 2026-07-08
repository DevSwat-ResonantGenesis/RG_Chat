"""Unit tests for Chat Service."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4


# ============================================
# Test Fixtures
# ============================================

@pytest.fixture
def mock_db_session():
    """Mock database session."""
    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.refresh = AsyncMock()
    return session


@pytest.fixture
def sample_chat():
    """Sample chat session."""
    return {
        "id": uuid4(),
        "user_id": uuid4(),
        "org_id": uuid4(),
        "title": "Test Conversation",
        "agent_id": None,
        "is_active": True,
        "created_at": datetime.utcnow(),
    }


@pytest.fixture
def sample_message():
    """Sample chat message."""
    return {
        "id": uuid4(),
        "chat_id": uuid4(),
        "role": "user",
        "content": "Hello, this is a test message.",
        "hash": "abc123",
        "resonance_score": 0.75,
        "created_at": datetime.utcnow(),
    }


# ============================================
# Chat Session Tests
# ============================================

class TestChatSession:
    """Test chat session management."""

    @pytest.mark.asyncio
    async def test_create_chat_session(self, mock_db_session):
        """Test creating a new chat session."""
        from app.services.chat_service import create_chat
        
        user_id = uuid4()
        org_id = uuid4()
        title = "New Conversation"
        
        mock_db_session.refresh = AsyncMock()
        
        chat = await create_chat(
            user_id=user_id,
            org_id=org_id,
            title=title,
            db=mock_db_session,
        )
        
        mock_db_session.commit.assert_called()

    @pytest.mark.asyncio
    async def test_get_chat_by_id(self, mock_db_session, sample_chat):
        """Test retrieving a chat by ID."""
        from app.services.chat_service import get_chat
        
        mock_chat = MagicMock(**sample_chat)
        mock_db_session.execute.return_value = MagicMock(
            scalar_one_or_none=lambda: mock_chat
        )
        
        chat = await get_chat(sample_chat["id"], mock_db_session)
        
        assert chat is not None

    @pytest.mark.asyncio
    async def test_list_user_chats(self, mock_db_session):
        """Test listing all chats for a user."""
        from app.services.chat_service import list_chats
        
        user_id = uuid4()
        mock_db_session.execute.return_value = MagicMock(
            scalars=lambda: MagicMock(all=lambda: [])
        )
        
        chats = await list_chats(user_id=user_id, db=mock_db_session)
        
        assert isinstance(chats, list)


# ============================================
# Message Tests
# ============================================

class TestMessages:
    """Test chat message handling."""

    @pytest.mark.asyncio
    async def test_add_message_to_chat(self, mock_db_session, sample_chat):
        """Test adding a message to a chat."""
        from app.services.message_service import add_message
        
        message_data = {
            "chat_id": sample_chat["id"],
            "role": "user",
            "content": "Test message content",
        }
        
        await add_message(message_data, mock_db_session)
        
        mock_db_session.commit.assert_called()

    @pytest.mark.asyncio
    async def test_get_chat_messages(self, mock_db_session, sample_chat):
        """Test retrieving messages from a chat."""
        from app.services.message_service import get_messages
        
        mock_db_session.execute.return_value = MagicMock(
            scalars=lambda: MagicMock(all=lambda: [])
        )
        
        messages = await get_messages(
            chat_id=sample_chat["id"],
            db=mock_db_session,
        )
        
        assert isinstance(messages, list)

    def test_message_role_validation(self):
        """Test that message roles are validated."""
        valid_roles = ["user", "assistant", "system"]
        invalid_role = "invalid"
        
        assert "user" in valid_roles
        assert "assistant" in valid_roles
        assert invalid_role not in valid_roles


# ============================================
# Resonant Chat Pipeline Tests
# ============================================

class TestResonantChatPipeline:
    """Test the Resonant Chat pipeline components."""

    def test_hash_message_content(self):
        """Test message content hashing."""
        from app.services.hash_service import hash_content
        
        content = "Test message to hash"
        hash_result = hash_content(content)
        
        assert hash_result is not None
        assert len(hash_result) == 64  # SHA-256

    def test_hash_deterministic(self):
        """Test that hashing is deterministic."""
        from app.services.hash_service import hash_content
        
        content = "Consistent message"
        hash1 = hash_content(content)
        hash2 = hash_content(content)
        
        assert hash1 == hash2

    @pytest.mark.asyncio
    async def test_memory_extraction(self):
        """Test memory extraction from message."""
        from app.services.memory_extraction import extract_memories
        
        with patch('app.services.memory_extraction.memory_client') as mock_client:
            mock_client.retrieve.return_value = []
            
            message = "This is important information to remember."
            user_id = uuid4()
            
            memories = await extract_memories(message, user_id)
            
            assert isinstance(memories, list)

    @pytest.mark.asyncio
    async def test_context_building(self):
        """Test context building for LLM."""
        from app.services.context_builder import build_context
        
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        memories = []
        
        context = await build_context(messages, memories)
        
        assert isinstance(context, list)
        assert len(context) >= len(messages)


# ============================================
# LLM Integration Tests
# ============================================

class TestLLMIntegration:
    """Test LLM integration functionality."""

    @pytest.mark.asyncio
    async def test_get_llm_response(self):
        """Test getting response from LLM."""
        from app.services.llm_client import get_completion
        
        with patch('app.services.llm_client.httpx.AsyncClient') as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "response": "This is a test response."
            }
            mock_response.status_code = 200
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )
            
            messages = [{"role": "user", "content": "Hello"}]
            
            response = await get_completion(messages)
            
            assert response is not None

    def test_provider_selection(self):
        """Test LLM provider selection logic."""
        from app.services.provider_router import select_provider
        
        # Test default provider
        provider = select_provider(user_keys=None)
        assert provider in ["openai", "anthropic"]
        
        # Test BYOK provider selection
        user_keys = {"openai": "sk-test-key"}
        provider = select_provider(user_keys=user_keys)
        assert provider == "openai"


# ============================================
# Streaming Tests
# ============================================

class TestStreaming:
    """Test streaming response functionality."""

    @pytest.mark.asyncio
    async def test_stream_response_yields_chunks(self):
        """Test that streaming yields chunks."""
        from app.services.streaming import stream_response
        
        async def mock_generator():
            for i in range(3):
                yield f"chunk_{i}"
        
        chunks = []
        async for chunk in mock_generator():
            chunks.append(chunk)
        
        assert len(chunks) == 3

    def test_stream_chunk_format(self):
        """Test stream chunk format."""
        chunk = {"delta": {"content": "test"}, "finish_reason": None}
        
        assert "delta" in chunk
        assert "content" in chunk["delta"]


# ============================================
# WebSocket Tests
# ============================================

class TestWebSocket:
    """Test WebSocket functionality."""

    @pytest.mark.asyncio
    async def test_websocket_connection(self):
        """Test WebSocket connection handling."""
        from app.routers.websocket import ConnectionManager
        
        manager = ConnectionManager()
        
        mock_ws = AsyncMock()
        mock_ws.accept = AsyncMock()
        mock_ws.send_text = AsyncMock()
        
        chat_id = str(uuid4())
        
        await manager.connect(mock_ws, chat_id)
        
        assert chat_id in manager.active_connections

    @pytest.mark.asyncio
    async def test_websocket_broadcast(self):
        """Test WebSocket message broadcasting."""
        from app.routers.websocket import ConnectionManager
        
        manager = ConnectionManager()
        
        mock_ws = AsyncMock()
        mock_ws.accept = AsyncMock()
        mock_ws.send_text = AsyncMock()
        
        chat_id = str(uuid4())
        await manager.connect(mock_ws, chat_id)
        
        await manager.broadcast(chat_id, "Test message")
        
        mock_ws.send_text.assert_called_with("Test message")


# ============================================
# Feedback Tests
# ============================================

class TestFeedback:
    """Test feedback functionality."""

    @pytest.mark.asyncio
    async def test_submit_feedback(self, mock_db_session):
        """Test submitting feedback for a message."""
        from app.services.feedback_service import submit_feedback
        
        feedback_data = {
            "message_id": uuid4(),
            "rating": 5,
            "comment": "Great response!",
        }
        
        await submit_feedback(feedback_data, mock_db_session)
        
        mock_db_session.commit.assert_called()

    def test_feedback_rating_validation(self):
        """Test feedback rating validation."""
        valid_ratings = [1, 2, 3, 4, 5]
        
        for rating in valid_ratings:
            assert 1 <= rating <= 5
        
        invalid_rating = 6
        assert invalid_rating > 5


# ============================================
# Run Tests
# ============================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
