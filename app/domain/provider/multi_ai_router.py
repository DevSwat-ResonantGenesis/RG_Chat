"""
Multi-AI Routing Service
Routes queries to appropriate AI providers (ChatGPT, Gemini, Groq)
Ported from old backend: ResonantGraphAIV0.1/backend/fastapi_app/services/multi_ai_routing.py
"""
from __future__ import annotations

import os
import logging
from typing import Dict, List, Optional

import httpx
from openai import OpenAI, AsyncOpenAI

from .context_adapter import context_adapter

logger = logging.getLogger(__name__)


class MultiAIRouter:
    """Route queries to different AI providers with automatic fallback."""
    
    def __init__(self):
        # Initialize OpenAI client
        self.openai_client = None
        if os.getenv("OPENAI_API_KEY"):
            self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Gemini API keys - support multiple keys for rate limit handling
        gemini_key_1 = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        gemini_key_2 = os.getenv("GEMINI_API_KEY_2")
        self.gemini_api_keys = []
        if gemini_key_1:
            self.gemini_api_keys.append(gemini_key_1)
        if gemini_key_2:
            self.gemini_api_keys.append(gemini_key_2)
        self.gemini_api_key = self.gemini_api_keys[0] if self.gemini_api_keys else None
        self.gemini_key_index = 0
        self.gemini_base_url = "https://generativelanguage.googleapis.com/v1beta"
        
        # Groq API keys - support multiple keys for rate limit handling
        groq_key_1 = os.getenv("GROQ_API_KEY")
        groq_key_2 = os.getenv("GROQ_API_KEY_2")
        self.groq_api_keys = []
        if groq_key_1:
            for k in groq_key_1.split(","):
                k = k.strip()
                if k:
                    self.groq_api_keys.append(k)
        if groq_key_2:
            for k in groq_key_2.split(","):
                k = k.strip()
                if k:
                    self.groq_api_keys.append(k)
        self.groq_api_key = self.groq_api_keys[0] if self.groq_api_keys else None
        self.groq_key_index = 0
        self.groq_base_url = "https://api.groq.com/openai/v1"
        
        # Anthropic API key
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        
        # User-specific API keys (set per-request)
        self._user_api_keys: Dict[str, str] = {}
    
    def set_user_api_keys(self, keys: Dict[str, str]):
        """Set user-specific API keys for BYOK (Bring Your Own Key) users."""
        self._user_api_keys = keys or {}
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers based on configured keys."""
        providers = []
        if self._user_api_keys.get('openai') or self.openai_client:
            providers.append('openai')
        if self._user_api_keys.get('anthropic') or self.anthropic_api_key:
            providers.append('anthropic')
        if self._user_api_keys.get('google') or self.gemini_api_keys:
            providers.append('google')
        if self._user_api_keys.get('groq') or self.groq_api_keys:
            providers.append('groq')
        return providers
    
    def _get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for provider - user key takes priority."""
        provider_map = {
            'chatgpt': 'openai',
            'gpt': 'openai',
            'gemini': 'google',
            'claude': 'anthropic',
        }
        normalized = provider_map.get(provider.lower(), provider.lower())
        
        if self._user_api_keys.get(normalized):
            return self._user_api_keys[normalized]
        
        if normalized == 'openai':
            return os.getenv("OPENAI_API_KEY")
        elif normalized == 'anthropic':
            return self.anthropic_api_key
        elif normalized == 'google':
            return self.gemini_api_key
        elif normalized == 'groq':
            return self.groq_api_key
        
        return None
    
    async def route_query(
        self,
        message: str,
        context: Optional[List[Dict]] = None,
        preferred_provider: Optional[str] = None,
        images: Optional[List[Dict]] = None,  # Base64 images for vision models
    ) -> Dict:
        """
        Route query to appropriate AI provider with automatic fallback.
        
        Args:
            images: List of dicts with {type, data, name} for vision models
        
        Returns:
            {
                'provider': str,
                'response': str,
                'metadata': dict
            }
        """
        provider_map = {
            "openai": "chatgpt",
            "chatgpt": "chatgpt",
            "gpt": "chatgpt",
            "gemini": "gemini",
            "google": "gemini",
            "groq": "groq",
            "claude": "claude",
            "anthropic": "claude",
        }
        normalized_provider = provider_map.get(
            preferred_provider.lower() if preferred_provider else "", 
            preferred_provider
        )
        
        # Build list of available providers in priority order
        # Check both environment keys AND user-provided keys (BYOK)
        has_openai = self.openai_client or self._user_api_keys.get('openai')
        has_gemini = self.gemini_api_keys or self._user_api_keys.get('google')
        has_groq = self.groq_api_key or self._user_api_keys.get('groq')
        has_anthropic = self.anthropic_api_key or self._user_api_keys.get('anthropic')
        
        available_providers = []
        if normalized_provider:
            if normalized_provider == "chatgpt" and has_openai:
                available_providers.append("chatgpt")
            elif normalized_provider == "gemini" and has_gemini:
                available_providers.append("gemini")
            elif normalized_provider == "groq" and has_groq:
                available_providers.append("groq")
            elif normalized_provider == "claude" and has_anthropic:
                available_providers.append("claude")
        
        # Add fallbacks: Groq first (only provider with working quota),
        # then quality providers as they come back online.
        if "groq" not in available_providers and has_groq:
            available_providers.append("groq")
        if "chatgpt" not in available_providers and has_openai:
            available_providers.append("chatgpt")
        if "claude" not in available_providers and has_anthropic:
            available_providers.append("claude")
        if "gemini" not in available_providers and has_gemini:
            available_providers.append("gemini")
        
        if not normalized_provider:
            auto_provider = self._select_provider(message, context)
            if auto_provider not in available_providers:
                if auto_provider in ["chatgpt", "gemini", "groq"]:
                    available_providers.insert(0, auto_provider)
        
        logger.info(f"🔄 Routing query - preferred_provider={preferred_provider}, normalized={normalized_provider}")
        logger.info(f"🔍 Provider availability: groq={has_groq}, gemini={has_gemini}, claude={has_anthropic}, openai={has_openai}")
        logger.info(f"🔄 Final provider list (in order): {available_providers}")
        
        # If images are provided, prefer vision-capable providers (OpenAI GPT-4V, Gemini)
        if images and len(images) > 0:
            logger.info(f"🖼️ Images detected ({len(images)}), prioritizing vision-capable providers")
            vision_providers = ["chatgpt", "gemini"]  # Vision-capable
            # Move vision providers to front
            for vp in reversed(vision_providers):
                if vp in available_providers:
                    available_providers.remove(vp)
                    available_providers.insert(0, vp)
        
        last_error = None
        fallback_chain = []  # Track every provider attempt
        for provider in available_providers:
            logger.info(f"🔄 Trying provider: {provider}")
            try:
                if provider == "chatgpt":
                    result = await self._call_chatgpt(message, context, images)
                elif provider == "gemini":
                    result = await self._call_gemini(message, context, images)
                elif provider == "groq":
                    result = await self._call_groq(message, context)  # Groq doesn't support vision
                elif provider == "claude":
                    result = await self._call_anthropic(message, context, images)
                else:
                    continue
                
                response_text = result.get("response", "")
                metadata = result.get("metadata", {})
                is_error = (
                    (response_text and response_text.startswith("Error calling")) or
                    metadata.get("quota_exceeded", False) or
                    metadata.get("error") is not None
                )
                
                if is_error:
                    reason = metadata.get("error") or response_text[:120]
                    fallback_chain.append({"provider": provider, "status": "failed", "reason": reason})
                    logger.warning(f"⚠️ {provider} failed: {response_text[:100]}. Trying next...")
                    last_error = result
                    continue
                
                fallback_chain.append({"provider": provider, "status": "success"})
                logger.info(f"✅ {provider} succeeded!")
                # Inject fallback chain + model into metadata
                result.setdefault("metadata", {})
                result["metadata"]["fallback_chain"] = fallback_chain
                result["metadata"]["was_fallback"] = len(fallback_chain) > 1
                result["metadata"]["preferred_provider"] = preferred_provider
                return result
            except Exception as e:
                last_error = {
                    'provider': provider,
                    'response': f"Error calling {provider}: {str(e)}",
                    'metadata': {'error': str(e)}
                }
                continue
        
        if last_error:
            return {
                'provider': 'resonant-brain',
                'response': "I apologize, but all AI providers are currently unavailable. "
                           "Please check your API keys or try again later.",
                'metadata': {
                    'error': last_error.get('response', 'All providers failed'),
                    'all_providers_failed': True,
                }
            }
        
        return {
            'provider': 'resonant-brain',
            'response': "No AI providers configured. Please set API keys.",
            'metadata': {'note': 'No AI providers configured'}
        }
    
    async def route_query_async(
        self,
        message: str,
        context: Optional[List[Dict]] = None,
        preferred_provider: Optional[str] = None,
    ) -> Dict:
        """Alias for route_query (now natively async)."""
        return await self.route_query(message, context, preferred_provider)
    
    def _select_provider(self, message: str, context: Optional[List[Dict]] = None) -> str:
        """
        Select best AI provider using Layer 8 intelligent routing.
        
        Uses task complexity analysis and cost optimization to select
        the optimal provider for each request.
        """
        try:
            from ...services.intelligent_router import select_optimal_provider
            
            # Build available providers list
            available = []
            if self.groq_api_key or self._user_api_keys.get('groq'):
                available.append("groq")
            if self.gemini_api_keys or self._user_api_keys.get('google'):
                available.append("gemini")
            if self.openai_client or self._user_api_keys.get('openai'):
                available.append("chatgpt")
            if self.anthropic_api_key or self._user_api_keys.get('anthropic'):
                available.append("claude")
            
            if not available:
                return "resonant-brain"
            
            # Use intelligent router for selection
            decision = select_optimal_provider(
                message=message,
                context=context,
                available_providers=available,
                optimize_for="balanced",
            )
            
            logger.info(f"[Layer8] Selected: {decision.provider} | Reason: {decision.reason}")
            return decision.provider
            
        except ImportError:
            # Fallback to simple selection if intelligent router not available
            logger.warning("Intelligent router not available, using fallback selection")
            if self.groq_api_key:
                return "groq"
            elif self.openai_client:
                return "chatgpt"
            elif self.gemini_api_keys:
                return "gemini"
            else:
                return "resonant-brain"
    
    async def _call_chatgpt(self, message: str, context: Optional[List[Dict]] = None, images: Optional[List[Dict]] = None) -> Dict:
        """Call OpenAI ChatGPT with optional vision support."""
        try:
            api_key = self._get_api_key('openai')
            if not api_key:
                return {
                    'provider': 'chatgpt',
                    'response': "Error calling ChatGPT: API key not configured",
                    'metadata': {'error': 'API key missing'}
                }
            
            client = AsyncOpenAI(api_key=api_key)
            
            messages = []
            if context:
                for msg in context:
                    if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                        messages.append({
                            "role": msg.get("role", "user"),
                            "content": str(msg.get("content", ""))
                        })
            
            # Build user message with optional images for vision
            if images and len(images) > 0:
                # GPT-4V format: content is array of text and image_url objects
                content_parts = [{"type": "text", "text": str(message)}]
                for img in images:
                    if img.get('data'):
                        # data is already base64 with data URI prefix from frontend
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": img['data']}
                        })
                        logger.info(f"🖼️ Added image to GPT-4V request: {img.get('name', 'image')}")
                messages.append({"role": "user", "content": content_parts})
            else:
                messages.append({"role": "user", "content": str(message)})
            
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7,
                max_tokens=4096
            )
            
            return {
                'provider': 'chatgpt',
                'response': response.choices[0].message.content,
                'metadata': {
                    'model': response.model,
                    'usage': {
                        'prompt_tokens': response.usage.prompt_tokens,
                        'completion_tokens': response.usage.completion_tokens,
                        'total_tokens': response.usage.total_tokens
                    }
                }
            }
        except Exception as e:
            error_str = str(e).lower()
            is_quota_error = any(x in error_str for x in ['429', 'quota', 'rate limit', 'billing'])
            return {
                'provider': 'chatgpt',
                'response': f"Error calling ChatGPT: {str(e)}",
                'metadata': {'error': str(e), 'quota_exceeded': is_quota_error}
            }
    
    async def _call_anthropic(self, message: str, context: Optional[List[Dict]] = None, images: Optional[List[Dict]] = None) -> Dict:
        """Call Anthropic Claude API with Context Adapter and vision support."""
        api_key = self._get_api_key('anthropic')
        if not api_key:
            return {
                'provider': 'claude',
                'response': "Error calling Claude: API key not configured",
                'metadata': {'error': 'API key missing'}
            }
        
        try:
            # Use Context Adapter to properly format context for Anthropic
            # This ensures ALL system messages are combined (fixes context loss)
            _, adapted = context_adapter.adapt_for_provider(
                message=message,
                context=context,
                provider="anthropic"
            )
            
            system_message = adapted.get("system")
            messages = adapted.get("messages", [])
            
            # context_adapter already appends the user message.
            # If images are provided, replace the last user message's
            # content with a multimodal content array for Claude Vision.
            if images and len(images) > 0:
                content_parts = []
                for img in images:
                    if img.get('data'):
                        data = img['data']
                        if ',' in data:
                            data = data.split(',')[1]
                        mime_type = img.get('type', 'image/png')
                        content_parts.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": data
                            }
                        })
                        logger.info(f"🖼️ Added image to Claude request: {img.get('name', 'image')}")
                content_parts.append({"type": "text", "text": message})
                # Replace the last user message with multimodal content
                if messages and messages[-1].get("role") == "user":
                    messages[-1]["content"] = content_parts
                else:
                    messages.append({"role": "user", "content": content_parts})
            
            # Log context stats for debugging
            stats = context_adapter.get_context_stats(context)
            logger.info(
                f"[Anthropic] Context stats: {stats['system']} system, "
                f"{stats['user']} user, {stats['assistant']} assistant messages. "
                f"System combined: {bool(system_message)}"
            )
            
            async with httpx.AsyncClient() as client:
                payload = {
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 4096,
                    "messages": messages,
                }
                if system_message:
                    payload["system"] = system_message
                
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json"
                    },
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()
                data = response.json()
                
                return {
                    'provider': 'claude',
                    'response': data['content'][0]['text'],
                    'metadata': {
                        'model': data.get('model', 'claude-3-haiku-20240307'),
                        'usage': data.get('usage', {}),
                    }
                }
        except httpx.HTTPStatusError as he:
            body = ""
            try:
                body = he.response.text
            except Exception:
                pass
            logger.error(f"[Claude] HTTP {he.response.status_code} error body: {body[:800]}")
            combined = (str(he) + " " + body).lower()
            is_quota_error = any(x in combined for x in [
                '429', 'quota', 'rate limit', 'billing',
                'credit balance', 'purchase credits', 'overloaded',
                'credit_balance_too_low', 'insufficient_quota',
            ])
            return {
                'provider': 'claude',
                'response': f"Error calling Claude: {str(he)} | {body[:200]}",
                'metadata': {'error': body or str(he), 'quota_exceeded': is_quota_error}
            }
        except Exception as e:
            error_str = str(e).lower()
            is_quota_error = any(x in error_str for x in [
                '429', 'quota', 'rate limit', 'billing',
                'credit balance', 'purchase credits',
            ])
            return {
                'provider': 'claude',
                'response': f"Error calling Claude: {str(e)}",
                'metadata': {'error': str(e), 'quota_exceeded': is_quota_error}
            }
    
    async def _call_groq(self, message: str, context: Optional[List[Dict]] = None) -> Dict:
        """Call Groq API with automatic key rotation."""
        user_groq_key = self._user_api_keys.get('groq')
        api_keys_to_try = []
        
        if user_groq_key:
            api_keys_to_try.append(user_groq_key)
        api_keys_to_try.extend(self.groq_api_keys)
        
        if not api_keys_to_try:
            return {
                'provider': 'groq',
                'response': "Error calling Groq: API key not configured",
                'metadata': {'error': 'API key missing'}
            }
        
        last_error = None
        for key_index, api_key in enumerate(api_keys_to_try):
            try:
                messages = []
                if context:
                    for msg in context:
                        if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                            role = msg.get("role", "user")
                            if role not in ["user", "assistant", "system"]:
                                role = "user"
                            messages.append({
                                "role": role,
                                "content": str(msg.get("content", ""))
                            })
                messages.append({"role": "user", "content": str(message)})
                
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.groq_base_url}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": "llama-3.3-70b-versatile",
                            "messages": messages,
                            "temperature": 0.7,
                            "max_tokens": 4096
                        },
                        timeout=30.0
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                    if 'choices' not in data or not data['choices']:
                        if key_index < len(api_keys_to_try) - 1:
                            continue
                        return {
                            'provider': 'groq',
                            'response': "Error calling Groq: Invalid response",
                            'metadata': {'error': 'Invalid response'}
                        }
                    
                    self.groq_key_index = (key_index + 1) % len(self.groq_api_keys) if self.groq_api_keys else 0
                    
                    return {
                        'provider': 'groq',
                        'response': data['choices'][0]['message']['content'],
                        'metadata': {
                            'model': data.get('model', 'llama-3.3-70b-versatile'),
                            'usage': data.get('usage', {}),
                        }
                    }
            except httpx.HTTPStatusError as e:
                error_detail = f"HTTP {e.response.status_code}"
                try:
                    error_data = e.response.json()
                    if isinstance(error_data, dict) and 'error' in error_data:
                        error_detail = str(error_data['error'].get('message', error_data['error']))
                except:
                    pass
                
                is_rate_limit = e.response.status_code == 429 or 'rate limit' in error_detail.lower()
                if is_rate_limit and key_index < len(api_keys_to_try) - 1:
                    last_error = error_detail
                    continue
                
                last_error = error_detail
                if key_index < len(api_keys_to_try) - 1:
                    continue
                return {
                    'provider': 'groq',
                    'response': f"Error calling Groq: {error_detail}",
                    'metadata': {'error': error_detail, 'rate_limit': is_rate_limit}
                }
            except Exception as e:
                last_error = str(e)
                if key_index < len(api_keys_to_try) - 1:
                    continue
                return {
                    'provider': 'groq',
                    'response': f"Error calling Groq: {str(e)}",
                    'metadata': {'error': str(e)}
                }
        
        return {
            'provider': 'groq',
            'response': f"Error calling Groq: All keys failed. {last_error}",
            'metadata': {'error': last_error, 'all_keys_failed': True}
        }
    
    async def _call_gemini(self, message: str, context: Optional[List[Dict]] = None, images: Optional[List[Dict]] = None) -> Dict:
        """Call Google Gemini with automatic key rotation and vision support."""
        # Build list of API keys to try - user keys first, then environment keys
        api_keys_to_try = []
        user_google_key = self._user_api_keys.get('google')
        if user_google_key:
            api_keys_to_try.append(user_google_key)
        api_keys_to_try.extend(self.gemini_api_keys)
        
        if not api_keys_to_try:
            return {
                'provider': 'gemini',
                'response': "Error calling Gemini: API key not configured",
                'metadata': {'error': 'API key missing'}
            }
        
        # Build content - Gemini doesn't support system role
        contents = []
        system_messages = []
        
        if context:
            for msg in context:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                
                if role == "system":
                    system_messages.append(content)
                else:
                    contents.append({
                        "role": "user" if role == "user" else "model",
                        "parts": [{"text": content}]
                    })
        
        # Merge system messages into user message
        merged_message = message
        if system_messages:
            system_context = "\n\n".join(system_messages)
            merged_message = f"{system_context}\n\n{message}"
        
        # Build parts with optional images for Gemini Vision
        parts = [{"text": merged_message}]
        if images and len(images) > 0:
            for img in images:
                if img.get('data'):
                    # Extract base64 data (remove data:image/xxx;base64, prefix)
                    data = img['data']
                    if ',' in data:
                        data = data.split(',')[1]
                    mime_type = img.get('type', 'image/png')
                    parts.append({
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": data
                        }
                    })
                    logger.info(f"🖼️ Added image to Gemini request: {img.get('name', 'image')}")
        
        contents.append({
            "role": "user",
            "parts": parts
        })
        
        last_error = None
        for key_index, api_key in enumerate(api_keys_to_try):
            try:
                async with httpx.AsyncClient() as client:
                    models_to_try = ['gemini-2.0-flash', 'gemini-1.5-flash']
                    
                    for model_name in models_to_try:
                        try:
                            response = await client.post(
                                f"{self.gemini_base_url}/models/{model_name}:generateContent?key={api_key}",
                                json={
                                    "contents": contents,
                                    "generationConfig": {
                                        "temperature": 0.7,
                                        "maxOutputTokens": 4096,
                                    },
                                },
                                timeout=30.0
                            )
                            response.raise_for_status()
                            data = response.json()
                            
                            self.gemini_key_index = (key_index + 1) % len(self.gemini_api_keys)
                            
                            return {
                                'provider': 'gemini',
                                'response': data['candidates'][0]['content']['parts'][0]['text'],
                                'metadata': {
                                    'model': model_name,
                                    'usage': data.get('usageMetadata', {}),
                                }
                            }
                        except httpx.HTTPStatusError as e:
                            if e.response.status_code == 404:
                                continue  # Try next model
                            raise
                        except Exception:
                            continue
                    
                    if key_index < len(self.gemini_api_keys) - 1:
                        continue
                    return {
                        'provider': 'gemini',
                        'response': f"Error calling Gemini: All models failed",
                        'metadata': {'error': 'All models failed'}
                    }
            except Exception as e:
                last_error = str(e)
                if key_index < len(self.gemini_api_keys) - 1:
                    continue
                return {
                    'provider': 'gemini',
                    'response': f"Error calling Gemini: {str(e)}",
                    'metadata': {'error': str(e)}
                }
        
        return {
            'provider': 'gemini',
            'response': f"Error calling Gemini: All keys failed. {last_error}",
            'metadata': {'error': last_error, 'all_keys_failed': True}
        }
