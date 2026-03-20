"""
Token Tracker Service - Track actual LLM token usage for accurate billing.

This replaces path-based cost estimates with actual token counting
using tiktoken for OpenAI-compatible models.

Phase 1.1 of GTM Production Strategy.
"""

import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# Try to import tiktoken, fall back to estimation if not available
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not installed - using character-based estimation")


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    GROQ = "groq"
    LOCAL = "local"


@dataclass
class TokenUsage:
    """Token usage for a single LLM call."""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    model: str
    provider: str
    credit_cost: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "model": self.model,
            "provider": self.provider,
            "credit_cost": self.credit_cost,
        }


class TokenTracker:
    """
    Track actual LLM token usage for accurate billing.
    
    Credit costs (from pricing.yaml):
    - Input tokens: 10 credits per 1K tokens
    - Output tokens: 30 credits per 1K tokens
    
    Provider multipliers:
    - OpenAI: 1.0x
    - Anthropic: 1.2x
    - Google: 0.8x
    - Groq: 0.5x
    - Local: 0.1x
    """
    
    # Credit costs per 1K tokens
    INPUT_COST_PER_1K = 10
    OUTPUT_COST_PER_1K = 30
    
    # Provider cost multipliers
    PROVIDER_MULTIPLIERS = {
        LLMProvider.OPENAI: 1.0,
        LLMProvider.ANTHROPIC: 1.2,
        LLMProvider.GOOGLE: 0.8,
        LLMProvider.GROQ: 0.5,
        LLMProvider.LOCAL: 0.1,
        # String aliases
        "openai": 1.0,
        "anthropic": 1.2,
        "google": 0.8,
        "groq": 0.5,
        "local": 0.1,
    }
    
    # Model-specific encodings
    MODEL_ENCODINGS = {
        "gpt-4": "cl100k_base",
        "gpt-4-turbo": "cl100k_base",
        "gpt-4o": "o200k_base",
        "gpt-4o-mini": "o200k_base",
        "gpt-3.5-turbo": "cl100k_base",
        "claude-3": "cl100k_base",  # Approximation
        "claude-3.5": "cl100k_base",
        "gemini-pro": "cl100k_base",  # Approximation
    }
    
    def __init__(self):
        self._encoders: Dict[str, Any] = {}
    
    def _get_encoder(self, model: str):
        """Get or create encoder for a model."""
        if not TIKTOKEN_AVAILABLE:
            return None
        
        if model not in self._encoders:
            try:
                # Try model-specific encoding first
                self._encoders[model] = tiktoken.encoding_for_model(model)
            except KeyError:
                # Fall back to encoding name from our mapping
                encoding_name = self.MODEL_ENCODINGS.get(model, "cl100k_base")
                try:
                    self._encoders[model] = tiktoken.get_encoding(encoding_name)
                except Exception:
                    self._encoders[model] = tiktoken.get_encoding("cl100k_base")
        
        return self._encoders[model]
    
    def count_tokens(self, text: str, model: str = "gpt-4o") -> int:
        """
        Count tokens for a given text and model.
        
        Args:
            text: Text to count tokens for
            model: Model name for encoding selection
            
        Returns:
            Token count
        """
        if not text:
            return 0
        
        if TIKTOKEN_AVAILABLE:
            encoder = self._get_encoder(model)
            if encoder:
                try:
                    return len(encoder.encode(text))
                except Exception as e:
                    logger.warning(f"Token encoding failed: {e}, using estimation")
        
        # Fallback: estimate ~4 characters per token
        return len(text) // 4
    
    def count_messages_tokens(
        self,
        messages: list,
        model: str = "gpt-4o"
    ) -> int:
        """
        Count tokens for a list of chat messages.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name
            
        Returns:
            Total token count
        """
        total = 0
        
        # Add overhead per message (role, formatting)
        tokens_per_message = 4  # OpenAI standard
        
        for message in messages:
            total += tokens_per_message
            content = message.get("content", "")
            if content:
                total += self.count_tokens(content, model)
            
            # Role tokens
            role = message.get("role", "")
            if role:
                total += self.count_tokens(role, model)
        
        # Add reply priming tokens
        total += 3
        
        return total
    
    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str = "gpt-4o",
        provider: str = "openai"
    ) -> int:
        """
        Calculate credit cost based on actual tokens.
        
        Args:
            input_tokens: Number of input/prompt tokens
            output_tokens: Number of output/completion tokens
            model: Model name (for future model-specific pricing)
            provider: LLM provider name
            
        Returns:
            Credit cost (integer)
        """
        multiplier = self.PROVIDER_MULTIPLIERS.get(provider.lower(), 1.0)
        
        input_credits = (input_tokens / 1000) * self.INPUT_COST_PER_1K * multiplier
        output_credits = (output_tokens / 1000) * self.OUTPUT_COST_PER_1K * multiplier
        
        # Round up to ensure we never undercharge
        total = int(input_credits + output_credits + 0.5)
        
        # Minimum cost of 1 credit
        return max(1, total)
    
    def track_usage(
        self,
        input_text: str,
        output_text: str,
        model: str = "gpt-4o",
        provider: str = "openai",
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
    ) -> TokenUsage:
        """
        Track token usage for a complete LLM call.
        
        If input_tokens/output_tokens are provided (from API response),
        use those. Otherwise, count from text.
        
        Args:
            input_text: Input/prompt text
            output_text: Output/completion text
            model: Model name
            provider: LLM provider
            input_tokens: Optional pre-counted input tokens
            output_tokens: Optional pre-counted output tokens
            
        Returns:
            TokenUsage with all metrics
        """
        # Use provided token counts or calculate
        if input_tokens is None:
            input_tokens = self.count_tokens(input_text, model)
        if output_tokens is None:
            output_tokens = self.count_tokens(output_text, model)
        
        total_tokens = input_tokens + output_tokens
        credit_cost = self.calculate_cost(input_tokens, output_tokens, model, provider)
        
        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            model=model,
            provider=provider,
            credit_cost=credit_cost,
        )
        
        logger.info(
            f"📊 Token usage: {input_tokens} in + {output_tokens} out = {total_tokens} total "
            f"({model}/{provider}) → {credit_cost} credits"
        )
        
        return usage
    
    def track_from_response(
        self,
        response: Dict[str, Any],
        model: str = "gpt-4o",
        provider: str = "openai",
        input_text: Optional[str] = None,
    ) -> TokenUsage:
        """
        Track usage from an LLM API response.
        
        Extracts token counts from response if available,
        otherwise falls back to counting.
        
        Args:
            response: API response dict
            model: Model name
            provider: LLM provider
            input_text: Optional input text for fallback counting
            
        Returns:
            TokenUsage with all metrics
        """
        # Try to extract usage from response (OpenAI format)
        usage = response.get("usage", {})
        input_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")
        
        # Anthropic format
        if input_tokens is None:
            input_tokens = usage.get("input_tokens")
        if output_tokens is None:
            output_tokens = usage.get("output_tokens")
        
        # Extract output text for fallback
        output_text = ""
        choices = response.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            output_text = message.get("content", "")
        
        # Anthropic format
        if not output_text:
            content = response.get("content", [])
            if content and isinstance(content, list):
                output_text = content[0].get("text", "")
        
        return self.track_usage(
            input_text=input_text or "",
            output_text=output_text,
            model=model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
    
    def estimate_cost_preview(
        self,
        input_text: str,
        estimated_output_tokens: int = 500,
        model: str = "gpt-4o",
        provider: str = "openai",
    ) -> Dict[str, Any]:
        """
        Estimate cost before execution (for cost preview API).
        
        Args:
            input_text: Input text
            estimated_output_tokens: Expected output tokens (default 500)
            model: Model name
            provider: LLM provider
            
        Returns:
            Cost estimate dict
        """
        input_tokens = self.count_tokens(input_text, model)
        estimated_cost = self.calculate_cost(
            input_tokens, estimated_output_tokens, model, provider
        )
        
        return {
            "estimated_cost": estimated_cost,
            "input_tokens": input_tokens,
            "estimated_output_tokens": estimated_output_tokens,
            "model": model,
            "provider": provider,
            "confidence": "medium",
            "note": "Actual cost depends on response length",
        }


# Global instance
token_tracker = TokenTracker()


# Convenience functions
def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count tokens for text."""
    return token_tracker.count_tokens(text, model)


def calculate_credit_cost(
    input_tokens: int,
    output_tokens: int,
    model: str = "gpt-4o",
    provider: str = "openai"
) -> int:
    """Calculate credit cost from token counts."""
    return token_tracker.calculate_cost(input_tokens, output_tokens, model, provider)


def track_llm_usage(
    input_text: str,
    output_text: str,
    model: str = "gpt-4o",
    provider: str = "openai",
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
) -> TokenUsage:
    """Track LLM usage and return TokenUsage."""
    return token_tracker.track_usage(
        input_text, output_text, model, provider, input_tokens, output_tokens
    )
