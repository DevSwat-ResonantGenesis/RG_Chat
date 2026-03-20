"""
Tests for Credit Deduction Service - Phase 1.1 GTM

Tests token-based credit deduction integration.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from app.services.credit_deduction import (
    deduct_credits,
    check_sufficient_credits,
    deduct_credits_by_tokens,
    deduct_credits_from_llm_response,
    generate_idempotency_key,
    CREDIT_COSTS,
    BILLING_SERVICE_URL,
)


class TestCreditCosts:
    """Test credit cost constants."""
    
    def test_credit_costs_defined(self):
        """Test all credit costs are defined."""
        assert "chat_message" in CREDIT_COSTS
        assert "code_execution" in CREDIT_COSTS
        assert "agent_run" in CREDIT_COSTS
        assert "agent_step" in CREDIT_COSTS
        assert "codebase_analysis" in CREDIT_COSTS
        assert "workflow_run" in CREDIT_COSTS
        assert "memory_write" in CREDIT_COSTS
    
    def test_credit_costs_values(self):
        """Test credit cost values are correct."""
        assert CREDIT_COSTS["chat_message"] == 20
        assert CREDIT_COSTS["code_execution"] == 5
        assert CREDIT_COSTS["agent_run"] == 100
        assert CREDIT_COSTS["agent_step"] == 50
        assert CREDIT_COSTS["codebase_analysis"] == 200
        assert CREDIT_COSTS["workflow_run"] == 50
        assert CREDIT_COSTS["memory_write"] == 2


class TestIdempotencyKeyGeneration:
    """Test idempotency key generation."""
    
    def test_generate_key_basic(self):
        """Test basic key generation."""
        key = generate_idempotency_key("user123", "action", 100)
        
        assert key is not None
        assert len(key) == 32
        assert isinstance(key, str)
    
    def test_generate_key_with_reference(self):
        """Test key generation with reference ID."""
        key1 = generate_idempotency_key("user123", "action", 100, "ref1")
        key2 = generate_idempotency_key("user123", "action", 100, "ref2")
        
        assert key1 != key2
    
    def test_generate_key_deterministic_same_minute(self):
        """Test same inputs produce same key within same minute."""
        key1 = generate_idempotency_key("user123", "action", 100, "ref1")
        key2 = generate_idempotency_key("user123", "action", 100, "ref1")
        
        assert key1 == key2


class TestDeductCredits:
    """Test basic credit deduction."""
    
    @pytest.mark.asyncio
    async def test_deduct_credits_success(self):
        """Test successful credit deduction."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": "tx123",
            "balance_after": 980,
        }
        mock_response.raise_for_status = MagicMock()
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )
            
            result = await deduct_credits(
                user_id="user123",
                action="chat_message",
            )
        
        assert result["deducted"] == 20
        assert result["balance"] == 980
    
    @pytest.mark.asyncio
    async def test_deduct_credits_custom_amount(self):
        """Test credit deduction with custom amount."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": "tx123",
            "balance_after": 900,
        }
        mock_response.raise_for_status = MagicMock()
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )
            
            result = await deduct_credits(
                user_id="user123",
                action="custom",
                amount=100,
            )
        
        assert result["deducted"] == 100
    
    @pytest.mark.asyncio
    async def test_deduct_credits_zero_amount(self):
        """Test zero amount returns without API call."""
        result = await deduct_credits(
            user_id="user123",
            action="unknown_action",
        )
        
        assert result["deducted"] == 0
        assert result["balance"] is None
    
    @pytest.mark.asyncio
    async def test_deduct_credits_insufficient(self):
        """Test insufficient credits raises error."""
        mock_response = MagicMock()
        mock_response.status_code = 402
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Payment Required",
            request=MagicMock(),
            response=mock_response,
        )
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )
            
            with pytest.raises(httpx.HTTPStatusError):
                await deduct_credits(
                    user_id="user123",
                    action="chat_message",
                )


class TestCheckSufficientCredits:
    """Test credit sufficiency check."""
    
    @pytest.mark.asyncio
    async def test_check_sufficient_true(self):
        """Test returns True when sufficient."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"balance": 1000}
        mock_response.raise_for_status = MagicMock()
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )
            
            result = await check_sufficient_credits("user123", "chat_message")
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_check_sufficient_false(self):
        """Test returns False when insufficient."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"balance": 10}
        mock_response.raise_for_status = MagicMock()
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )
            
            result = await check_sufficient_credits("user123", "chat_message")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_check_sufficient_zero_cost(self):
        """Test returns True for zero cost action."""
        result = await check_sufficient_credits("user123", "unknown_action")
        assert result is True
    
    @pytest.mark.asyncio
    async def test_check_sufficient_error_allows(self):
        """Test returns True on error (fail open)."""
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=Exception("Network error")
            )
            
            result = await check_sufficient_credits("user123", "chat_message")
        
        # Should allow on error
        assert result is True


class TestDeductCreditsByTokens:
    """Test token-based credit deduction."""
    
    @pytest.mark.asyncio
    async def test_deduct_by_tokens_success(self):
        """Test successful token-based deduction."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": "tx123",
            "balance_after": 993,
        }
        mock_response.raise_for_status = MagicMock()
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )
            
            result = await deduct_credits_by_tokens(
                user_id="user123",
                input_tokens=100,
                output_tokens=200,
                model="gpt-4o",
                provider="openai",
            )
        
        assert "balance" in result
        assert "deducted" in result
        assert "token_usage" in result
        assert result["token_usage"]["input_tokens"] == 100
        assert result["token_usage"]["output_tokens"] == 200
        assert result["token_usage"]["total_tokens"] == 300
    
    @pytest.mark.asyncio
    async def test_deduct_by_tokens_with_reference(self):
        """Test token deduction with reference ID."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": "tx123",
            "balance_after": 990,
        }
        mock_response.raise_for_status = MagicMock()
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )
            
            result = await deduct_credits_by_tokens(
                user_id="user123",
                input_tokens=500,
                output_tokens=500,
                model="gpt-4o",
                provider="openai",
                reference_id="msg_123",
            )
        
        assert result["token_usage"]["model"] == "gpt-4o"
        assert result["token_usage"]["provider"] == "openai"
    
    @pytest.mark.asyncio
    async def test_deduct_by_tokens_error_handling(self):
        """Test error handling in token deduction."""
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=Exception("Network error")
            )
            
            result = await deduct_credits_by_tokens(
                user_id="user123",
                input_tokens=100,
                output_tokens=200,
                model="gpt-4o",
                provider="openai",
            )
        
        assert result["deducted"] == 0
        assert "error" in result


class TestDeductCreditsFromLLMResponse:
    """Test deduction from LLM API response."""
    
    @pytest.mark.asyncio
    async def test_deduct_from_openai_response(self):
        """Test deduction from OpenAI response format."""
        llm_response = {
            "choices": [{"message": {"content": "Hello!"}}],
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 100,
            }
        }
        
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": "tx123",
            "balance_after": 995,
        }
        mock_response.raise_for_status = MagicMock()
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )
            
            result = await deduct_credits_from_llm_response(
                user_id="user123",
                response=llm_response,
                model="gpt-4o",
                provider="openai",
            )
        
        assert "token_usage" in result
    
    @pytest.mark.asyncio
    async def test_deduct_from_anthropic_response(self):
        """Test deduction from Anthropic response format."""
        llm_response = {
            "content": [{"text": "Hello!"}],
            "usage": {
                "input_tokens": 50,
                "output_tokens": 100,
            }
        }
        
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": "tx123",
            "balance_after": 994,
        }
        mock_response.raise_for_status = MagicMock()
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )
            
            result = await deduct_credits_from_llm_response(
                user_id="user123",
                response=llm_response,
                model="claude-3",
                provider="anthropic",
            )
        
        assert "token_usage" in result


class TestBillingServiceURL:
    """Test billing service URL configuration."""
    
    def test_billing_service_url(self):
        """Test billing service URL is correct."""
        assert BILLING_SERVICE_URL == "http://billing_service:8000"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
