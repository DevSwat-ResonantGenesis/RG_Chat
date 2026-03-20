"""
Dynamic AI Provider Registry
============================

Allows adding new AI providers without code changes.
Providers can be configured via:
1. Environment variables (PROVIDER_CONFIG_JSON)
2. Database (future)
3. Admin API (future)

This enables BYOK for ANY provider on the market.
"""
import os
import json
import logging
import httpx
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ProviderType(str, Enum):
    """Supported provider API types."""
    OPENAI_COMPATIBLE = "openai_compatible"  # OpenAI API format (most common)
    ANTHROPIC = "anthropic"  # Anthropic's format
    GOOGLE = "google"  # Google Gemini format
    CUSTOM = "custom"  # Custom implementation


@dataclass
class ProviderConfig:
    """Configuration for an AI provider."""
    id: str  # Unique identifier (e.g., "openai", "mistral", "together")
    name: str  # Display name (e.g., "OpenAI", "Mistral AI")
    api_type: ProviderType  # API format type
    base_url: str  # API base URL
    default_model: str  # Default model to use
    models: List[str] = field(default_factory=list)  # Available models
    env_key_name: str = ""  # Environment variable name for API key
    headers: Dict[str, str] = field(default_factory=dict)  # Custom headers
    supports_vision: bool = False  # Supports image input
    supports_streaming: bool = True  # Supports streaming
    max_tokens: int = 4096  # Default max tokens
    enabled: bool = True  # Whether provider is enabled


# Default provider configurations
DEFAULT_PROVIDERS: Dict[str, ProviderConfig] = {
    "openai": ProviderConfig(
        id="openai",
        name="OpenAI",
        api_type=ProviderType.OPENAI_COMPATIBLE,
        base_url="https://api.openai.com/v1",
        default_model="gpt-4o",
        models=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        env_key_name="OPENAI_API_KEY",
        supports_vision=True,
    ),
    "anthropic": ProviderConfig(
        id="anthropic",
        name="Anthropic",
        api_type=ProviderType.ANTHROPIC,
        base_url="https://api.anthropic.com/v1",
        default_model="claude-3-haiku-20240307",
        models=["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
        env_key_name="ANTHROPIC_API_KEY",
        supports_vision=True,
    ),
    "google": ProviderConfig(
        id="google",
        name="Google Gemini",
        api_type=ProviderType.GOOGLE,
        base_url="https://generativelanguage.googleapis.com/v1beta",
        default_model="gemini-1.5-flash",
        models=["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"],
        env_key_name="GOOGLE_API_KEY",
        supports_vision=True,
    ),
    "groq": ProviderConfig(
        id="groq",
        name="Groq",
        api_type=ProviderType.OPENAI_COMPATIBLE,
        base_url="https://api.groq.com/openai/v1",
        default_model="llama-3.3-70b-versatile",
        models=["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"],
        env_key_name="GROQ_API_KEY",
        supports_vision=False,
    ),
    # ============================================
    # ADDITIONAL PROVIDERS (Easy to add!)
    # ============================================
    "mistral": ProviderConfig(
        id="mistral",
        name="Mistral AI",
        api_type=ProviderType.OPENAI_COMPATIBLE,
        base_url="https://api.mistral.ai/v1",
        default_model="mistral-large-latest",
        models=["mistral-large-latest", "mistral-medium-latest", "mistral-small-latest", "open-mixtral-8x22b"],
        env_key_name="MISTRAL_API_KEY",
        supports_vision=False,
    ),
    "together": ProviderConfig(
        id="together",
        name="Together AI",
        api_type=ProviderType.OPENAI_COMPATIBLE,
        base_url="https://api.together.xyz/v1",
        default_model="meta-llama/Llama-3-70b-chat-hf",
        models=["meta-llama/Llama-3-70b-chat-hf", "mistralai/Mixtral-8x7B-Instruct-v0.1", "togethercomputer/CodeLlama-34b-Instruct"],
        env_key_name="TOGETHER_API_KEY",
        supports_vision=False,
    ),
    "perplexity": ProviderConfig(
        id="perplexity",
        name="Perplexity",
        api_type=ProviderType.OPENAI_COMPATIBLE,
        base_url="https://api.perplexity.ai",
        default_model="llama-3.1-sonar-large-128k-online",
        models=["llama-3.1-sonar-large-128k-online", "llama-3.1-sonar-small-128k-online"],
        env_key_name="PERPLEXITY_API_KEY",
        supports_vision=False,
    ),
    "deepseek": ProviderConfig(
        id="deepseek",
        name="DeepSeek",
        api_type=ProviderType.OPENAI_COMPATIBLE,
        base_url="https://api.deepseek.com/v1",
        default_model="deepseek-chat",
        models=["deepseek-chat", "deepseek-coder"],
        env_key_name="DEEPSEEK_API_KEY",
        supports_vision=False,
    ),
    "fireworks": ProviderConfig(
        id="fireworks",
        name="Fireworks AI",
        api_type=ProviderType.OPENAI_COMPATIBLE,
        base_url="https://api.fireworks.ai/inference/v1",
        default_model="accounts/fireworks/models/llama-v3p1-70b-instruct",
        models=["accounts/fireworks/models/llama-v3p1-70b-instruct", "accounts/fireworks/models/mixtral-8x7b-instruct"],
        env_key_name="FIREWORKS_API_KEY",
        supports_vision=False,
    ),
    "openrouter": ProviderConfig(
        id="openrouter",
        name="OpenRouter",
        api_type=ProviderType.OPENAI_COMPATIBLE,
        base_url="https://openrouter.ai/api/v1",
        default_model="openai/gpt-4o",
        models=["openai/gpt-4o", "anthropic/claude-3-opus", "google/gemini-pro", "meta-llama/llama-3-70b-instruct"],
        env_key_name="OPENROUTER_API_KEY",
        headers={"HTTP-Referer": "https://resonantgenesis.com"},
        supports_vision=True,
    ),
    "cohere": ProviderConfig(
        id="cohere",
        name="Cohere",
        api_type=ProviderType.OPENAI_COMPATIBLE,
        base_url="https://api.cohere.ai/v1",
        default_model="command-r-plus",
        models=["command-r-plus", "command-r", "command"],
        env_key_name="COHERE_API_KEY",
        supports_vision=False,
    ),
    "anyscale": ProviderConfig(
        id="anyscale",
        name="Anyscale",
        api_type=ProviderType.OPENAI_COMPATIBLE,
        base_url="https://api.endpoints.anyscale.com/v1",
        default_model="meta-llama/Llama-3-70b-chat-hf",
        models=["meta-llama/Llama-3-70b-chat-hf", "mistralai/Mixtral-8x7B-Instruct-v0.1"],
        env_key_name="ANYSCALE_API_KEY",
        supports_vision=False,
    ),
}


class ProviderRegistry:
    """
    Dynamic provider registry that can be configured at runtime.
    
    Usage:
        registry = ProviderRegistry()
        
        # Get available providers
        providers = registry.get_available_providers()
        
        # Add a new provider at runtime
        registry.add_provider(ProviderConfig(
            id="my_provider",
            name="My Custom Provider",
            api_type=ProviderType.OPENAI_COMPATIBLE,
            base_url="https://api.myprovider.com/v1",
            default_model="my-model",
            env_key_name="MY_PROVIDER_API_KEY",
        ))
    """
    
    def __init__(self):
        self._providers: Dict[str, ProviderConfig] = {}
        self._user_keys: Dict[str, str] = {}
        self._load_default_providers()
        self._load_custom_providers_from_env()
    
    def _load_default_providers(self):
        """Load default provider configurations."""
        for provider_id, config in DEFAULT_PROVIDERS.items():
            self._providers[provider_id] = config
        logger.info(f"📦 Loaded {len(self._providers)} default providers")
    
    def _load_custom_providers_from_env(self):
        """
        Load custom providers from PROVIDER_CONFIG_JSON environment variable.
        
        Format:
        {
            "providers": [
                {
                    "id": "my_provider",
                    "name": "My Provider",
                    "api_type": "openai_compatible",
                    "base_url": "https://api.example.com/v1",
                    "default_model": "model-name",
                    "env_key_name": "MY_PROVIDER_API_KEY"
                }
            ]
        }
        """
        config_json = os.getenv("PROVIDER_CONFIG_JSON")
        if not config_json:
            return
        
        try:
            config = json.loads(config_json)
            for provider_data in config.get("providers", []):
                provider = ProviderConfig(
                    id=provider_data["id"],
                    name=provider_data["name"],
                    api_type=ProviderType(provider_data.get("api_type", "openai_compatible")),
                    base_url=provider_data["base_url"],
                    default_model=provider_data["default_model"],
                    models=provider_data.get("models", [provider_data["default_model"]]),
                    env_key_name=provider_data.get("env_key_name", f"{provider_data['id'].upper()}_API_KEY"),
                    headers=provider_data.get("headers", {}),
                    supports_vision=provider_data.get("supports_vision", False),
                    supports_streaming=provider_data.get("supports_streaming", True),
                    max_tokens=provider_data.get("max_tokens", 4096),
                    enabled=provider_data.get("enabled", True),
                )
                self._providers[provider.id] = provider
                logger.info(f"📦 Loaded custom provider: {provider.name}")
        except Exception as e:
            logger.error(f"Failed to load custom providers from env: {e}")
    
    def add_provider(self, config: ProviderConfig):
        """Add or update a provider configuration."""
        self._providers[config.id] = config
        logger.info(f"📦 Added provider: {config.name}")
    
    def remove_provider(self, provider_id: str):
        """Remove a provider."""
        if provider_id in self._providers:
            del self._providers[provider_id]
            logger.info(f"📦 Removed provider: {provider_id}")
    
    def get_provider(self, provider_id: str) -> Optional[ProviderConfig]:
        """Get provider configuration by ID."""
        # Handle aliases
        aliases = {
            "chatgpt": "openai",
            "gpt": "openai",
            "claude": "anthropic",
            "gemini": "google",
        }
        normalized_id = aliases.get(provider_id.lower(), provider_id.lower())
        return self._providers.get(normalized_id)
    
    def get_all_providers(self) -> List[ProviderConfig]:
        """Get all registered providers."""
        return list(self._providers.values())
    
    def get_available_providers(self, user_keys: Optional[Dict[str, str]] = None) -> List[ProviderConfig]:
        """
        Get providers that have API keys configured.
        
        Args:
            user_keys: User-provided API keys (BYOK)
        
        Returns:
            List of available providers
        """
        available = []
        for provider in self._providers.values():
            if not provider.enabled:
                continue
            
            # Check user key first (BYOK)
            if user_keys and user_keys.get(provider.id):
                available.append(provider)
                continue
            
            # Check environment key
            if provider.env_key_name and os.getenv(provider.env_key_name):
                available.append(provider)
        
        return available
    
    def get_api_key(self, provider_id: str, user_keys: Optional[Dict[str, str]] = None) -> Optional[str]:
        """
        Get API key for a provider.
        User keys take priority over environment keys.
        """
        provider = self.get_provider(provider_id)
        if not provider:
            return None
        
        # User key first (BYOK)
        if user_keys and user_keys.get(provider.id):
            return user_keys[provider.id]
        
        # Environment key
        if provider.env_key_name:
            return os.getenv(provider.env_key_name)
        
        return None
    
    def set_user_keys(self, keys: Dict[str, str]):
        """Set user-specific API keys for the current request."""
        self._user_keys = keys or {}
    
    async def call_provider(
        self,
        provider_id: str,
        message: str,
        context: Optional[List[Dict]] = None,
        model: Optional[str] = None,
        user_keys: Optional[Dict[str, str]] = None,
        images: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Call an AI provider using its configuration.
        
        This is a universal caller that works with any OpenAI-compatible API.
        """
        provider = self.get_provider(provider_id)
        if not provider:
            return {
                "provider": provider_id,
                "response": f"Error: Provider '{provider_id}' not found",
                "metadata": {"error": "Provider not found"}
            }
        
        api_key = self.get_api_key(provider_id, user_keys)
        if not api_key:
            return {
                "provider": provider_id,
                "response": f"Error: API key not configured for {provider.name}",
                "metadata": {"error": "API key missing"}
            }
        
        model_to_use = model or provider.default_model
        
        # Build messages
        messages = []
        if context:
            for msg in context:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    messages.append({
                        "role": msg.get("role", "user"),
                        "content": str(msg.get("content", ""))
                    })
        messages.append({"role": "user", "content": message})
        
        # Handle different API types
        if provider.api_type == ProviderType.OPENAI_COMPATIBLE:
            return await self._call_openai_compatible(provider, api_key, messages, model_to_use, images)
        elif provider.api_type == ProviderType.ANTHROPIC:
            return await self._call_anthropic(provider, api_key, messages, model_to_use, images)
        elif provider.api_type == ProviderType.GOOGLE:
            return await self._call_google(provider, api_key, messages, model_to_use, images)
        else:
            return {
                "provider": provider_id,
                "response": f"Error: Unsupported API type '{provider.api_type}'",
                "metadata": {"error": "Unsupported API type"}
            }
    
    async def _call_openai_compatible(
        self,
        provider: ProviderConfig,
        api_key: str,
        messages: List[Dict],
        model: str,
        images: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """Call OpenAI-compatible API (works with most providers)."""
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                **provider.headers,
            }
            
            payload = {
                "model": model,
                "messages": messages,
                "temperature": 0.5,
                "max_tokens": provider.max_tokens,
            }
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{provider.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                
                return {
                    "provider": provider.id,
                    "response": data["choices"][0]["message"]["content"],
                    "metadata": {
                        "model": data.get("model", model),
                        "usage": data.get("usage", {}),
                    }
                }
        except Exception as e:
            return {
                "provider": provider.id,
                "response": f"Error calling {provider.name}: {str(e)}",
                "metadata": {"error": str(e)}
            }
    
    async def _call_anthropic(
        self,
        provider: ProviderConfig,
        api_key: str,
        messages: List[Dict],
        model: str,
        images: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """Call Anthropic API."""
        try:
            # Extract system message
            system_content = ""
            user_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    system_content += msg["content"] + "\n"
                else:
                    user_messages.append(msg)
            
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            }
            
            payload = {
                "model": model,
                "max_tokens": provider.max_tokens,
                "messages": user_messages,
            }
            if system_content:
                payload["system"] = system_content.strip()
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{provider.base_url}/messages",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                
                return {
                    "provider": provider.id,
                    "response": data["content"][0]["text"],
                    "metadata": {
                        "model": data.get("model", model),
                        "usage": data.get("usage", {}),
                    }
                }
        except Exception as e:
            return {
                "provider": provider.id,
                "response": f"Error calling {provider.name}: {str(e)}",
                "metadata": {"error": str(e)}
            }
    
    async def _call_google(
        self,
        provider: ProviderConfig,
        api_key: str,
        messages: List[Dict],
        model: str,
        images: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """Call Google Gemini API."""
        try:
            # Convert to Gemini format
            contents = []
            for msg in messages:
                role = "user" if msg["role"] == "user" else "model"
                contents.append({
                    "role": role,
                    "parts": [{"text": msg["content"]}]
                })
            
            url = f"{provider.base_url}/models/{model}:generateContent?key={api_key}"
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    url,
                    json={"contents": contents},
                )
                response.raise_for_status()
                data = response.json()
                
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                
                return {
                    "provider": provider.id,
                    "response": text,
                    "metadata": {
                        "model": model,
                        "usage": data.get("usageMetadata", {}),
                    }
                }
        except Exception as e:
            return {
                "provider": provider.id,
                "response": f"Error calling {provider.name}: {str(e)}",
                "metadata": {"error": str(e)}
            }


# Global instance
provider_registry = ProviderRegistry()
