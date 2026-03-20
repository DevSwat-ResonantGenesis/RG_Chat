"""
User API Keys Service (BYOK - Bring Your Own Key)
==================================================

Retrieves and manages user-provided API keys for AI providers.

Ported from old backend: ResonantGraphAIV0.1/backend/fastapi_app/models/user_api_key.py
"""
from __future__ import annotations

import logging
import os
import httpx
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class UserApiKeyService:
    """
    User API Key Service
    
    Retrieves user's own API keys for BYOK functionality.
    Keys are stored encrypted in the auth service.
    """
    
    AUTH_SERVICE_URL = os.getenv("AUTH_SERVICE_URL", "http://green_auth_service:8000")
    
    async def get_user_api_keys(self, user_id: str) -> Dict[str, str]:
        """
        Retrieve user's API keys from auth service.
        
        Args:
            user_id: User ID
        
        Returns:
            Dictionary mapping provider names to API keys
        """
        try:
            # Use internal decrypted-key endpoint for BYOK service-to-service calls
            url = f"{self.AUTH_SERVICE_URL.rstrip('/')}/auth/internal/user-api-keys/{user_id}"
            logger.info(f"🔑 Fetching API keys from: {url} for user: {user_id}")

            internal_key = os.getenv("AUTH_INTERNAL_SERVICE_KEY") or os.getenv("INTERNAL_SERVICE_KEY")
            headers = {"x-user-id": user_id}
            if internal_key:
                headers["x-internal-service-key"] = internal_key
            
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    url, 
                    timeout=5.0,
                    headers=headers,
                )
                
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"🔑 Auth service response: {data}")
                    keys = {}
                    
                    for key_entry in data.get("keys", []):
                        provider = key_entry.get("provider")
                        api_key = key_entry.get("api_key")
                        
                        if provider and api_key:
                            keys[provider] = api_key
                            logger.info(f"🔑 Found key for provider: {provider}")
                    
                    logger.info(f"🔑 Retrieved {len(keys)} API keys for user {user_id}: {list(keys.keys())}")
                    return keys
                else:
                    logger.warning(f"🔑 Failed to retrieve API keys: {response.status_code} - {response.text}")
                    return {}
                    
        except httpx.RequestError as e:
            logger.warning(f"Error retrieving user API keys: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error retrieving API keys: {e}")
            return {}
    
    async def get_provider_key(self, user_id: str, provider: str) -> Optional[str]:
        """
        Get API key for a specific provider.
        
        Args:
            user_id: User ID
            provider: Provider name (openai, anthropic, google, groq)
        
        Returns:
            API key or None
        """
        keys = await self.get_user_api_keys(user_id)
        return keys.get(provider)
    
    async def validate_key(self, provider: str, api_key: str) -> bool:
        """
        Validate an API key by making a test request.
        
        Args:
            provider: Provider name
            api_key: API key to validate
        
        Returns:
            True if valid, False otherwise
        """
        try:
            if provider == "openai":
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        "https://api.openai.com/v1/models",
                        headers={"Authorization": f"Bearer {api_key}"},
                        timeout=5.0
                    )
                    return response.status_code == 200
            
            elif provider == "groq":
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        "https://api.groq.com/openai/v1/models",
                        headers={"Authorization": f"Bearer {api_key}"},
                        timeout=5.0
                    )
                    return response.status_code == 200
            
            elif provider == "google":
                # Gemini validation would go here
                return True
            
            elif provider == "anthropic":
                # Claude validation would go here
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Error validating {provider} key: {e}")
            return False
    
    def format_keys_for_router(self, keys: Dict[str, str]) -> Dict[str, str]:
        """
        Format keys for use with MultiAIRouter.
        
        Args:
            keys: Dictionary of provider -> API key
        
        Returns:
            Formatted dictionary for router
        
        Note: MultiAIRouter._get_api_key normalizes provider names to:
        openai, anthropic, google, groq - so we keep those names.
        """
        formatted = {}
        
        # Keep provider names as-is since MultiAIRouter normalizes them
        # The router expects: openai, anthropic, google, groq
        for provider, key in keys.items():
            # Normalize to lowercase
            formatted[provider.lower()] = key
        
        return formatted


# Global instance
user_api_key_service = UserApiKeyService()
