"""
Tests for Token Tracker Service - Phase 1.1 GTM

Tests actual LLM token counting and credit cost calculation.
"""

import pytest
from unittest.mock import patch, MagicMock

from app.services.token_tracker import (
    TokenTracker,
    TokenUsage,
    LLMProvider,
    token_tracker,
    count_tokens,
    calculate_credit_cost,
    track_llm_usage,
)


class TestTokenTracker:
    """Test TokenTracker class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.tracker = TokenTracker()
    
    def test_count_tokens_basic(self):
        """Test basic token counting."""
        text = "Hello, world!"
        tokens = self.tracker.count_tokens(text)
        assert tokens > 0
        assert isinstance(tokens, int)
    
    def test_count_tokens_empty(self):
        """Test empty string returns 0."""
        assert self.tracker.count_tokens("") == 0
        assert self.tracker.count_tokens(None) == 0
    
    def test_count_tokens_long_text(self):
        """Test longer text has more tokens."""
        short = "Hello"
        long = "Hello, this is a much longer piece of text that should have more tokens."
        
        short_tokens = self.tracker.count_tokens(short)
        long_tokens = self.tracker.count_tokens(long)
        
        assert long_tokens > short_tokens
    
    def test_count_tokens_different_models(self):
        """Test token counting works for different models."""
        text = "Test message for token counting"
        
        gpt4_tokens = self.tracker.count_tokens(text, "gpt-4")
        gpt4o_tokens = self.tracker.count_tokens(text, "gpt-4o")
        
        # Both should return valid counts
        assert gpt4_tokens > 0
        assert gpt4o_tokens > 0
    
    def test_calculate_cost_basic(self):
        """Test basic cost calculation."""
        # 1000 input + 1000 output tokens
        # Input: 1000/1000 * 10 = 10 credits
        # Output: 1000/1000 * 30 = 30 credits
        # Total: 40 credits
        cost = self.tracker.calculate_cost(1000, 1000, "gpt-4o", "openai")
        assert cost == 40
    
    def test_calculate_cost_small_amounts(self):
        """Test cost calculation with small token counts."""
        # 100 input + 200 output
        # Input: 100/1000 * 10 = 1 credit
        # Output: 200/1000 * 30 = 6 credits
        # Total: 7 credits (rounded)
        cost = self.tracker.calculate_cost(100, 200, "gpt-4o", "openai")
        assert cost == 7
    
    def test_calculate_cost_minimum(self):
        """Test minimum cost is 1 credit."""
        # Very small usage should still cost at least 1 credit
        cost = self.tracker.calculate_cost(1, 1, "gpt-4o", "openai")
        assert cost >= 1
    
    def test_calculate_cost_provider_multipliers(self):
        """Test provider multipliers affect cost."""
        input_tokens = 1000
        output_tokens = 1000
        
        openai_cost = self.tracker.calculate_cost(input_tokens, output_tokens, "gpt-4o", "openai")
        anthropic_cost = self.tracker.calculate_cost(input_tokens, output_tokens, "claude-3", "anthropic")
        groq_cost = self.tracker.calculate_cost(input_tokens, output_tokens, "llama", "groq")
        
        # OpenAI: 1.0x multiplier = 40 credits
        assert openai_cost == 40
        
        # Anthropic: 1.2x multiplier = 48 credits
        assert anthropic_cost == 48
        
        # Groq: 0.5x multiplier = 20 credits
        assert groq_cost == 20
    
    def test_track_usage_returns_token_usage(self):
        """Test track_usage returns TokenUsage object."""
        usage = self.tracker.track_usage(
            input_text="Hello",
            output_text="Hi there!",
            model="gpt-4o",
            provider="openai",
        )
        
        assert isinstance(usage, TokenUsage)
        assert usage.input_tokens > 0
        assert usage.output_tokens > 0
        assert usage.total_tokens == usage.input_tokens + usage.output_tokens
        assert usage.credit_cost >= 1
        assert usage.model == "gpt-4o"
        assert usage.provider == "openai"
    
    def test_track_usage_with_provided_tokens(self):
        """Test track_usage uses provided token counts."""
        usage = self.tracker.track_usage(
            input_text="ignored",
            output_text="ignored",
            model="gpt-4o",
            provider="openai",
            input_tokens=500,
            output_tokens=1000,
        )
        
        assert usage.input_tokens == 500
        assert usage.output_tokens == 1000
        assert usage.total_tokens == 1500
    
    def test_track_from_response_openai_format(self):
        """Test extracting usage from OpenAI response format."""
        response = {
            "choices": [{"message": {"content": "Hello!"}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
            }
        }
        
        usage = self.tracker.track_from_response(response, "gpt-4o", "openai")
        
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
    
    def test_track_from_response_anthropic_format(self):
        """Test extracting usage from Anthropic response format."""
        response = {
            "content": [{"text": "Hello!"}],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
            }
        }
        
        usage = self.tracker.track_from_response(response, "claude-3", "anthropic")
        
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
    
    def test_estimate_cost_preview(self):
        """Test cost estimation before execution."""
        estimate = self.tracker.estimate_cost_preview(
            input_text="What is the meaning of life?",
            estimated_output_tokens=500,
            model="gpt-4o",
            provider="openai",
        )
        
        assert "estimated_cost" in estimate
        assert "input_tokens" in estimate
        assert "estimated_output_tokens" in estimate
        assert estimate["estimated_output_tokens"] == 500
        assert estimate["confidence"] == "medium"
    
    def test_token_usage_to_dict(self):
        """Test TokenUsage.to_dict() method."""
        usage = TokenUsage(
            input_tokens=100,
            output_tokens=200,
            total_tokens=300,
            model="gpt-4o",
            provider="openai",
            credit_cost=10,
        )
        
        d = usage.to_dict()
        
        assert d["input_tokens"] == 100
        assert d["output_tokens"] == 200
        assert d["total_tokens"] == 300
        assert d["model"] == "gpt-4o"
        assert d["provider"] == "openai"
        assert d["credit_cost"] == 10


class TestConvenienceFunctions:
    """Test module-level convenience functions."""
    
    def test_count_tokens_function(self):
        """Test count_tokens convenience function."""
        tokens = count_tokens("Hello, world!")
        assert tokens > 0
    
    def test_calculate_credit_cost_function(self):
        """Test calculate_credit_cost convenience function."""
        cost = calculate_credit_cost(1000, 1000, "gpt-4o", "openai")
        assert cost == 40
    
    def test_track_llm_usage_function(self):
        """Test track_llm_usage convenience function."""
        usage = track_llm_usage("Hello", "Hi there!", "gpt-4o", "openai")
        assert isinstance(usage, TokenUsage)


class TestProviderMultipliers:
    """Test all provider multipliers."""
    
    def setup_method(self):
        self.tracker = TokenTracker()
        self.base_input = 1000
        self.base_output = 1000
        # Base cost: (1000/1000)*10 + (1000/1000)*30 = 40
    
    def test_openai_multiplier(self):
        """OpenAI: 1.0x multiplier."""
        cost = self.tracker.calculate_cost(self.base_input, self.base_output, "gpt-4o", "openai")
        assert cost == 40  # 40 * 1.0
    
    def test_anthropic_multiplier(self):
        """Anthropic: 1.2x multiplier."""
        cost = self.tracker.calculate_cost(self.base_input, self.base_output, "claude-3", "anthropic")
        assert cost == 48  # 40 * 1.2
    
    def test_google_multiplier(self):
        """Google: 0.8x multiplier."""
        cost = self.tracker.calculate_cost(self.base_input, self.base_output, "gemini", "google")
        assert cost == 32  # 40 * 0.8
    
    def test_groq_multiplier(self):
        """Groq: 0.5x multiplier."""
        cost = self.tracker.calculate_cost(self.base_input, self.base_output, "llama", "groq")
        assert cost == 20  # 40 * 0.5
    
    def test_local_multiplier(self):
        """Local: 0.1x multiplier."""
        cost = self.tracker.calculate_cost(self.base_input, self.base_output, "local-model", "local")
        assert cost == 4  # 40 * 0.1
    
    def test_unknown_provider_default(self):
        """Unknown provider defaults to 1.0x."""
        cost = self.tracker.calculate_cost(self.base_input, self.base_output, "model", "unknown_provider")
        assert cost == 40  # 40 * 1.0


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def setup_method(self):
        self.tracker = TokenTracker()
    
    def test_zero_tokens(self):
        """Test zero tokens returns minimum cost."""
        cost = self.tracker.calculate_cost(0, 0, "gpt-4o", "openai")
        assert cost >= 1  # Minimum 1 credit
    
    def test_very_large_token_count(self):
        """Test very large token counts."""
        cost = self.tracker.calculate_cost(100000, 100000, "gpt-4o", "openai")
        # (100000/1000)*10 + (100000/1000)*30 = 1000 + 3000 = 4000
        assert cost == 4000
    
    def test_unicode_text(self):
        """Test token counting with unicode."""
        text = "Hello 世界! 🌍 Привет мир"
        tokens = self.tracker.count_tokens(text)
        assert tokens > 0
    
    def test_multiline_text(self):
        """Test token counting with multiline text."""
        text = """Line 1
        Line 2
        Line 3"""
        tokens = self.tracker.count_tokens(text)
        assert tokens > 0
    
    def test_count_messages_tokens(self):
        """Test counting tokens for chat messages."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        
        tokens = self.tracker.count_messages_tokens(messages)
        assert tokens > 0
        
        # Should be more than just the content tokens due to overhead
        content_only = sum(
            self.tracker.count_tokens(m.get("content", ""))
            for m in messages
        )
        assert tokens > content_only


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
