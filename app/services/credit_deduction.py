"""
Credit Deduction Service for Chat Service
Integrates with billing_service to deduct credits on each action.

Phase 1.1 GTM: Now supports actual token-based billing via TokenTracker.
"""
import logging
import httpx
import hashlib
from typing import Optional, Dict, Any
from datetime import datetime

from .token_tracker import token_tracker, TokenUsage

logger = logging.getLogger(__name__)

# Billing service URL (Docker service name)
BILLING_SERVICE_URL = "http://billing_service:8000"

# Default credit costs per action (from pricing.yaml)
# These are fallbacks when actual token counts aren't available
CREDIT_COSTS = {
    "chat_message": 20,
    "code_execution": 5,
    "agent_run": 100,
    "agent_step": 50,
    "codebase_analysis": 200,
    "workflow_run": 50,
    "memory_write": 2,
}

# Idempotency cache TTL (24 hours)
IDEMPOTENCY_TTL = 86400


async def deduct_credits(
    user_id: str,
    action: str,
    amount: Optional[int] = None,
    description: Optional[str] = None,
    user_role: Optional[str] = None,
    is_superuser: bool = False,
    unlimited_credits: bool = False,
) -> dict:
    """
    Deduct credits from user's balance via billing service.
    
    Args:
        user_id: User ID
        action: Action type (e.g., "chat_message", "code_execution")
        amount: Credit amount (if None, uses CREDIT_COSTS[action])
        description: Optional description
        user_role: User role for unlimited credits check
        is_superuser: Whether user is a superuser (unlimited credits)
        
    Returns:
        dict with new balance and transaction info
        
    Raises:
        httpx.HTTPStatusError: If insufficient credits or other error
    """
    if amount is None:
        amount = CREDIT_COSTS.get(action, 0)
    
    if amount == 0:
        logger.warning(f"No credits to deduct for action: {action}")
        return {"balance": None, "deducted": 0}
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BILLING_SERVICE_URL}/billing/credits/deduct",
                json={
                    "amount": amount,
                    "reference_type": action,
                    "description": description or f"Used for {action}",
                },
                headers={
                    "X-User-Id": user_id,
                    "X-User-Role": user_role or "user",
                    "X-Is-Superuser": str(is_superuser).lower(),
                    "X-Unlimited-Credits": str(unlimited_credits).lower(),
                    "Content-Type": "application/json",
                },
                timeout=5.0,
            )
            response.raise_for_status()
            result = response.json()
            
            logger.info(
                f"💳 Deducted {amount} credits from user {user_id[:8]}... "
                f"for {action}. New balance: {result.get('balance_after', 'unknown')}"
            )
            
            return {
                "balance": result.get("balance_after"),
                "deducted": amount,
                "transaction_id": result.get("id"),
            }
            
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 402:
            # Insufficient credits
            logger.warning(f"❌ Insufficient credits for user {user_id[:8]}... action: {action}")
            raise
        else:
            logger.error(f"❌ Credit deduction failed: {e}")
            raise
    except Exception as e:
        logger.error(f"❌ Credit deduction error: {e}", exc_info=True)
        # Don't block the request if billing service is down
        # Just log the error and continue
        return {"balance": None, "deducted": 0, "error": str(e)}


async def check_sufficient_credits(user_id: str, action: str, amount: Optional[int] = None) -> bool:
    """
    Check if user has sufficient credits for an action.
    
    Args:
        user_id: User ID
        action: Action type
        amount: Credit amount (if None, uses CREDIT_COSTS[action])
        
    Returns:
        True if user has sufficient credits, False otherwise
    """
    if amount is None:
        amount = CREDIT_COSTS.get(action, 0)
    
    if amount == 0:
        return True
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BILLING_SERVICE_URL}/billing/credits/balance/{user_id}",
                timeout=5.0,
            )
            response.raise_for_status()
            data = response.json()
            balance = data.get("balance", 0)
            
            return balance >= amount
            
    except Exception as e:
        logger.error(f"❌ Failed to check credit balance: {e}")
        # If billing service is down, allow the request
        return True


def generate_idempotency_key(
    user_id: str,
    action: str,
    amount: int,
    reference_id: Optional[str] = None,
) -> str:
    """
    Generate idempotency key for billing operations.
    
    Prevents duplicate charges from retries or network issues.
    """
    timestamp_bucket = int(datetime.utcnow().timestamp() // 60)  # 1-minute buckets
    data = f"{user_id}:{action}:{amount}:{reference_id or ''}:{timestamp_bucket}"
    return hashlib.sha256(data.encode()).hexdigest()[:32]


async def deduct_credits_by_tokens(
    user_id: str,
    input_tokens: int,
    output_tokens: int,
    model: str = "gpt-4o",
    provider: str = "openai",
    reference_id: Optional[str] = None,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Deduct credits based on actual token usage.
    
    This is the preferred method for LLM calls as it provides
    accurate billing based on actual consumption.
    
    Args:
        user_id: User ID
        input_tokens: Number of input/prompt tokens
        output_tokens: Number of output/completion tokens
        model: LLM model name
        provider: LLM provider (openai, anthropic, etc.)
        reference_id: Optional reference ID for idempotency
        description: Optional description
        
    Returns:
        dict with balance, deducted amount, and token details
    """
    # Calculate credit cost from actual tokens
    credit_cost = token_tracker.calculate_cost(
        input_tokens, output_tokens, model, provider
    )
    
    # Generate idempotency key
    idempotency_key = generate_idempotency_key(
        user_id, "llm_tokens", credit_cost, reference_id
    )
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BILLING_SERVICE_URL}/billing/credits/deduct",
                json={
                    "amount": credit_cost,
                    "reference_type": "llm_tokens",
                    "reference_id": reference_id,
                    "idempotency_key": idempotency_key,
                    "description": description or f"LLM usage: {input_tokens} in + {output_tokens} out ({model}/{provider})",
                    "metadata": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens,
                        "model": model,
                        "provider": provider,
                    },
                },
                headers={
                    "X-User-Id": user_id,
                    "Content-Type": "application/json",
                },
                timeout=5.0,
            )
            response.raise_for_status()
            result = response.json()
            
            logger.info(
                f"💳 Token-based deduction: {credit_cost} credits from user {user_id[:8]}... "
                f"({input_tokens} in + {output_tokens} out = {input_tokens + output_tokens} tokens)"
            )
            
            return {
                "balance": result.get("balance_after"),
                "deducted": credit_cost,
                "transaction_id": result.get("id"),
                "token_usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "model": model,
                    "provider": provider,
                },
            }
            
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 402:
            logger.warning(
                f"❌ Insufficient credits for user {user_id[:8]}... "
                f"needed {credit_cost} credits for {input_tokens + output_tokens} tokens"
            )
            raise
        else:
            logger.error(f"❌ Token-based credit deduction failed: {e}")
            raise
    except Exception as e:
        logger.error(f"❌ Token-based credit deduction error: {e}", exc_info=True)
        return {
            "balance": None,
            "deducted": 0,
            "error": str(e),
            "token_usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        }


async def deduct_credits_from_llm_response(
    user_id: str,
    response: Dict[str, Any],
    model: str = "gpt-4o",
    provider: str = "openai",
    input_text: Optional[str] = None,
    reference_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Deduct credits from an LLM API response.
    
    Extracts token counts from the response and deducts accordingly.
    This is the most accurate method when the API provides usage stats.
    
    Args:
        user_id: User ID
        response: LLM API response dict
        model: Model name
        provider: LLM provider
        input_text: Optional input text for fallback counting
        reference_id: Optional reference ID
        
    Returns:
        dict with balance, deducted amount, and token details
    """
    # Track usage from response
    usage = token_tracker.track_from_response(
        response, model, provider, input_text
    )
    
    # Deduct based on actual tokens
    return await deduct_credits_by_tokens(
        user_id=user_id,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        model=model,
        provider=provider,
        reference_id=reference_id,
        description=f"Chat: {usage.total_tokens} tokens via {model}",
    )
