"""
Context Adapter for Multi-Provider LLM Calls
=============================================

Systems-level solution to handle provider API differences.

Problem:
- Different providers have different API formats
- Anthropic requires system messages in separate parameter
- OpenAI/Groq keep system messages in messages array
- Context loss occurs when not properly adapted

Solution:
- Centralized adapter that transforms context for each provider
- Preserves ALL context regardless of provider
- Maintains consistency across providers
- Single source of truth for provider-specific formatting

Author: Resonant Chat Systems Team
Date: December 26, 2025
"""

from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ContextAdapter:
    """
    Unified context adapter for all LLM providers.
    
    Handles provider-specific API format differences to ensure
    no context loss and consistent intelligence across providers.
    """
    
    def adapt_for_provider(
        self,
        message: str,
        context: Optional[List[Dict[str, Any]]],
        provider: str
    ) -> Tuple[str, Any]:
        """
        Adapt context for specific provider API format.
        
        Args:
            message: User message
            context: List of context messages (system, user, assistant)
            provider: Provider name (chatgpt, groq, gemini, claude)
            
        Returns:
            Tuple of (message, adapted_context)
            - For OpenAI-compatible: (message, messages_array)
            - For Anthropic: (message, {"system": str, "messages": list})
        """
        if not context:
            context = []
        
        provider_lower = provider.lower()
        
        if provider_lower in ["claude", "anthropic"]:
            return self._adapt_anthropic(message, context)
        elif provider_lower in ["chatgpt", "openai", "groq"]:
            return self._adapt_openai(message, context)
        elif provider_lower in ["gemini", "google"]:
            return self._adapt_gemini(message, context)
        else:
            # Default to OpenAI format
            logger.warning(f"Unknown provider {provider}, using OpenAI format")
            return self._adapt_openai(message, context)
    
    def _adapt_anthropic(
        self,
        message: str,
        context: List[Dict[str, Any]]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Adapt context for Anthropic Claude API.
        
        Anthropic requires:
        - System messages in separate 'system' parameter (single string)
        - Conversation messages in 'messages' array (user/assistant only)
        
        This is the CRITICAL fix for Anthropic context loss.
        """
        system_messages = []
        conversation_messages = []
        
        # Separate system messages from conversation
        for msg in context:
            if not isinstance(msg, dict):
                continue
                
            role = msg.get("role", "user")
            content = str(msg.get("content", ""))
            
            if not content:
                continue
            
            if role == "system":
                # Collect ALL system messages
                system_messages.append(content)
            elif role in ["user", "assistant"]:
                # Keep conversation messages
                conversation_messages.append({
                    "role": role,
                    "content": content
                })
            else:
                # Unknown role, treat as user
                logger.warning(f"Unknown role {role}, treating as user")
                conversation_messages.append({
                    "role": "user",
                    "content": content
                })
        
        # Add current user message
        conversation_messages.append({
            "role": "user",
            "content": message
        })
        
        # Anthropic requires strictly alternating user/assistant roles.
        # Merge any consecutive same-role messages to prevent 400 errors.
        merged = []
        for msg in conversation_messages:
            if merged and merged[-1]["role"] == msg["role"]:
                merged[-1]["content"] += "\n\n" + msg["content"]
            else:
                merged.append(dict(msg))
        conversation_messages = merged
        
        # Ensure the first message is from the user (Anthropic requirement)
        if conversation_messages and conversation_messages[0]["role"] != "user":
            conversation_messages.insert(0, {
                "role": "user",
                "content": "(continuing conversation)"
            })
        
        # Combine ALL system messages (this is the fix!)
        combined_system = "\n\n".join(system_messages) if system_messages else None
        
        logger.info(
            f"[ContextAdapter] Anthropic: "
            f"{len(system_messages)} system messages combined, "
            f"{len(conversation_messages)} conversation messages"
        )
        
        return message, {
            "system": combined_system,
            "messages": conversation_messages
        }
    
    def _adapt_openai(
        self,
        message: str,
        context: List[Dict[str, Any]]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Adapt context for OpenAI/Groq API.
        
        OpenAI format:
        - All messages (system, user, assistant) in single array
        - No special handling needed
        """
        messages = []
        
        # Add all context messages
        for msg in context:
            if not isinstance(msg, dict):
                continue
                
            role = msg.get("role", "user")
            content = str(msg.get("content", ""))
            
            if not content:
                continue
            
            messages.append({
                "role": role,
                "content": content
            })
        
        # Add current user message
        messages.append({
            "role": "user",
            "content": message
        })
        
        logger.info(
            f"[ContextAdapter] OpenAI: {len(messages)} total messages"
        )
        
        return message, messages
    
    def _adapt_gemini(
        self,
        message: str,
        context: List[Dict[str, Any]]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Adapt context for Google Gemini API.
        
        Gemini format:
        - System instruction in separate 'system_instruction' parameter
        - Conversation in 'contents' array with 'parts'
        """
        system_messages = []
        conversation_messages = []
        
        # Separate system from conversation
        for msg in context:
            if not isinstance(msg, dict):
                continue
                
            role = msg.get("role", "user")
            content = str(msg.get("content", ""))
            
            if not content:
                continue
            
            if role == "system":
                system_messages.append(content)
            else:
                # Gemini uses 'user' and 'model' roles
                gemini_role = "model" if role == "assistant" else "user"
                conversation_messages.append({
                    "role": gemini_role,
                    "parts": [{"text": content}]
                })
        
        # Add current user message
        conversation_messages.append({
            "role": "user",
            "parts": [{"text": message}]
        })
        
        # Combine system messages
        combined_system = "\n\n".join(system_messages) if system_messages else None
        
        logger.info(
            f"[ContextAdapter] Gemini: "
            f"{len(system_messages)} system messages combined, "
            f"{len(conversation_messages)} conversation messages"
        )
        
        return message, {
            "system_instruction": combined_system,
            "contents": conversation_messages
        }
    
    def validate_context(self, context: Optional[List[Dict[str, Any]]]) -> bool:
        """
        Validate context structure.
        
        Returns True if context is valid, False otherwise.
        """
        if context is None:
            return True
        
        if not isinstance(context, list):
            logger.error(f"Context must be a list, got {type(context)}")
            return False
        
        for i, msg in enumerate(context):
            if not isinstance(msg, dict):
                logger.error(f"Context message {i} must be dict, got {type(msg)}")
                return False
            
            if "role" not in msg:
                logger.error(f"Context message {i} missing 'role' field")
                return False
            
            if "content" not in msg:
                logger.error(f"Context message {i} missing 'content' field")
                return False
        
        return True
    
    def get_context_stats(self, context: Optional[List[Dict[str, Any]]]) -> Dict[str, int]:
        """
        Get statistics about context.
        
        Returns dict with counts of system, user, assistant messages.
        """
        if not context:
            return {"system": 0, "user": 0, "assistant": 0, "total": 0}
        
        stats = {"system": 0, "user": 0, "assistant": 0, "other": 0}
        
        for msg in context:
            if not isinstance(msg, dict):
                continue
            
            role = msg.get("role", "unknown")
            if role in stats:
                stats[role] += 1
            else:
                stats["other"] += 1
        
        stats["total"] = sum(stats.values())
        return stats


# Global instance
context_adapter = ContextAdapter()
