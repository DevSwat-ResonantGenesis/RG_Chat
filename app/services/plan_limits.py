"""
Plan Limits Service
Enforces conversation and message limits based on user's subscription plan.
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import httpx

logger = logging.getLogger(__name__)

# Plan limits configuration
PLAN_LIMITS = {
    "developer": {
        "conversations": 1000,
        "messages_per_day": 100,
        "agents": 3,
        "rag_documents": 5,
        "storage_mb": 100,
    },
    "free": {
        "conversations": 1000,
        "messages_per_day": 100,
        "agents": 3,
        "rag_documents": 5,
        "storage_mb": 100,
    },
    "plus": {
        "conversations": 1000,
        "messages_per_day": 1000,
        "agents": 20,
        "rag_documents": 100,
        "storage_mb": 5000,
    },
    "professional": {
        "conversations": 1000,
        "messages_per_day": 1000,
        "agents": 20,
        "rag_documents": 100,
        "storage_mb": 5000,
    },
    "enterprise": {
        "conversations": -1,  # Unlimited
        "messages_per_day": -1,
        "agents": -1,
        "rag_documents": -1,
        "storage_mb": -1,
    },
    "unlimited": {
        "conversations": -1,
        "messages_per_day": -1,
        "agents": -1,
        "rag_documents": -1,
        "storage_mb": -1,
    },
}


class PlanLimitsService:
    """Service to check and enforce plan limits."""
    
    async def get_user_plan(self, user_id: str) -> str:
        """Get user's current plan from billing service."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "http://billing_service:8000/billing/subscription",
                    headers={"x-user-id": user_id},
                    timeout=5.0,
                )
                if response.status_code == 200:
                    data = response.json()
                    plan = data.get("plan", "developer").lower()
                    # Handle dev users
                    if data.get("is_dev"):
                        return "unlimited"
                    return plan
        except Exception as e:
            logger.warning(f"Failed to get user plan: {e}")
        return "developer"  # Default to free tier
    
    def get_limits(self, plan: str) -> Dict:
        """Get limits for a plan."""
        return PLAN_LIMITS.get(plan.lower(), PLAN_LIMITS["developer"])
    
    async def check_conversation_limit(
        self, 
        user_id: str, 
        current_count: int,
        plan: Optional[str] = None
    ) -> Tuple[bool, str, int]:
        """
        Check if user can create a new conversation.
        
        Returns:
            Tuple of (allowed, message, limit)
        """
        if not plan:
            plan = await self.get_user_plan(user_id)
        
        limits = self.get_limits(plan)
        max_conversations = limits["conversations"]
        
        # Unlimited
        if max_conversations == -1:
            return True, "OK", -1
        
        if current_count >= max_conversations:
            return (
                False, 
                f"Conversation limit reached ({current_count}/{max_conversations}). "
                f"Upgrade to Plus for 1,000 conversations.",
                max_conversations
            )
        
        return True, "OK", max_conversations
    
    async def check_message_limit(
        self, 
        user_id: str, 
        messages_today: int,
        plan: Optional[str] = None
    ) -> Tuple[bool, str, int]:
        """
        Check if user can send a message today.
        
        Returns:
            Tuple of (allowed, message, limit)
        """
        if not plan:
            plan = await self.get_user_plan(user_id)
        
        limits = self.get_limits(plan)
        max_messages = limits["messages_per_day"]
        
        # Unlimited
        if max_messages == -1:
            return True, "OK", -1
        
        if messages_today >= max_messages:
            return (
                False, 
                f"Daily message limit reached ({messages_today}/{max_messages}). "
                f"Upgrade to Plus for 1,000 messages/day.",
                max_messages
            )
        
        return True, "OK", max_messages
    
    async def check_agent_limit(
        self, 
        user_id: str, 
        current_count: int,
        plan: Optional[str] = None
    ) -> Tuple[bool, str, int]:
        """
        Check if user can create a new agent.
        
        Returns:
            Tuple of (allowed, message, limit)
        """
        if not plan:
            plan = await self.get_user_plan(user_id)
        
        limits = self.get_limits(plan)
        max_agents = limits["agents"]
        
        # Unlimited
        if max_agents == -1:
            return True, "OK", -1
        
        if current_count >= max_agents:
            return (
                False, 
                f"Agent limit reached ({current_count}/{max_agents}). "
                f"Upgrade to Plus for 20 agents.",
                max_agents
            )
        
        return True, "OK", max_agents
    
    async def check_rag_document_limit(
        self, 
        user_id: str, 
        current_count: int,
        plan: Optional[str] = None
    ) -> Tuple[bool, str, int]:
        """
        Check if user can upload a new RAG document.
        
        Returns:
            Tuple of (allowed, message, limit)
        """
        if not plan:
            plan = await self.get_user_plan(user_id)
        
        limits = self.get_limits(plan)
        max_docs = limits["rag_documents"]
        
        # Unlimited
        if max_docs == -1:
            return True, "OK", -1
        
        if current_count >= max_docs:
            return (
                False, 
                f"RAG document limit reached ({current_count}/{max_docs}). "
                f"Upgrade to Plus for 100 documents.",
                max_docs
            )
        
        return True, "OK", max_docs
    
    async def get_usage_stats(self, user_id: str, session) -> Dict:
        """Get current usage statistics for a user."""
        from sqlalchemy import select, func
        from datetime import date
        from ..models import ResonantChat, ResonantChatMessage
        
        plan = await self.get_user_plan(user_id)
        limits = self.get_limits(plan)
        
        # Count conversations
        from uuid import UUID
        conv_result = await session.execute(
            select(func.count(ResonantChat.id)).where(
                ResonantChat.user_id == UUID(user_id)
            )
        )
        conversation_count = conv_result.scalar() or 0
        
        # Count messages today
        today_start = datetime.combine(date.today(), datetime.min.time())
        msg_result = await session.execute(
            select(func.count(ResonantChatMessage.id)).where(
                ResonantChatMessage.chat_id.in_(
                    select(ResonantChat.id).where(ResonantChat.user_id == UUID(user_id))
                ),
                ResonantChatMessage.role == "user",
                ResonantChatMessage.created_at >= today_start
            )
        )
        messages_today = msg_result.scalar() or 0
        
        return {
            "plan": plan,
            "conversations": {
                "used": conversation_count,
                "limit": limits["conversations"],
                "remaining": limits["conversations"] - conversation_count if limits["conversations"] > 0 else -1,
            },
            "messages_today": {
                "used": messages_today,
                "limit": limits["messages_per_day"],
                "remaining": limits["messages_per_day"] - messages_today if limits["messages_per_day"] > 0 else -1,
            },
            "limits": limits,
        }


# Singleton instance
plan_limits_service = PlanLimitsService()
