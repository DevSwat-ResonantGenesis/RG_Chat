"""Image Generation Service for Resonant Chat.

Provides AI image generation capabilities using:
- TokenRouter image models (primary): openai/gpt-5-image, openai/gpt-5-image-mini, google/gemini-3.1-flash-image-preview
- OpenAI DALL-E 3 (fallback)
- OpenAI DALL-E 2 (legacy fallback)
"""
import os
import re
import logging
import httpx
import base64
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

from rg_llm import UnifiedLLMClient, LLMRequest, TOKENROUTER_IMAGE_MODELS

logger = logging.getLogger(__name__)

_llm_client = UnifiedLLMClient()


class ImageSize(str, Enum):
    """Supported image sizes."""
    SQUARE_1024 = "1024x1024"
    LANDSCAPE_1792 = "1792x1024"
    PORTRAIT_1024 = "1024x1792"
    SQUARE_512 = "512x512"  # DALL-E 2 only
    SQUARE_256 = "256x256"  # DALL-E 2 only


class ImageQuality(str, Enum):
    """Image quality options."""
    STANDARD = "standard"
    HD = "hd"


class ImageStyle(str, Enum):
    """Image style options (DALL-E 3 only)."""
    VIVID = "vivid"
    NATURAL = "natural"


class GeneratedImage:
    """Represents a generated image."""
    def __init__(
        self,
        url: Optional[str] = None,
        base64_data: Optional[str] = None,
        revised_prompt: Optional[str] = None,
        model: str = "dall-e-3",
        size: str = "1024x1024",
        created_at: Optional[datetime] = None,
    ):
        self.url = url
        self.base64_data = base64_data
        self.revised_prompt = revised_prompt
        self.model = model
        self.size = size
        self.created_at = created_at or datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "base64_data": self.base64_data,
            "revised_prompt": self.revised_prompt,
            "model": self.model,
            "size": self.size,
            "created_at": self.created_at.isoformat(),
        }


class ImageGenerationService:
    """Image generation service using TokenRouter image models + DALL-E fallback."""
    
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.stability_api_key = os.getenv("STABILITY_API_KEY")
        self.timeout = 60.0  # Image generation can take time
        self.base_url = "https://api.openai.com/v1"
        self._user_keys: Optional[Dict[str, str]] = None
    
    def set_api_key(self, openai_key: Optional[str] = None, stability_key: Optional[str] = None):
        """Set API keys dynamically (for user-provided keys)."""
        if openai_key:
            self.openai_api_key = openai_key
        if stability_key:
            self.stability_api_key = stability_key

    def set_user_keys(self, keys: Optional[Dict[str, str]] = None):
        """Set user BYOK keys for TokenRouter calls."""
        self._user_keys = keys
    
    async def generate(
        self,
        prompt: str,
        model: str = "auto",
        size: str = "1024x1024",
        quality: str = "standard",
        style: str = "vivid",
        n: int = 1,
        response_format: str = "url",  # "url" or "b64_json"
    ) -> List[GeneratedImage]:
        """
        Generate images using TokenRouter image models (primary) or DALL-E (fallback).
        
        Args:
            prompt: Text description of the image to generate
            model: "auto" (smart route), specific TokenRouter model, or "dall-e-3"/"dall-e-2"
            size: Image size (1024x1024, 1792x1024, 1024x1792 for DALL-E 3)
            quality: "standard" or "hd" (DALL-E 3 only)
            style: "vivid" or "natural" (DALL-E 3 only)
            n: Number of images (1 for DALL-E 3, 1-10 for DALL-E 2)
            response_format: "url" or "b64_json"
        
        Returns:
            List of GeneratedImage objects
        """
        # Try TokenRouter image models first (unless user explicitly requests DALL-E)
        _is_dalle = model.startswith("dall-e")
        if not _is_dalle:
            try:
                result = await self._generate_via_tokenrouter(prompt, model)
                if result:
                    return result
            except Exception as _tr_err:
                logger.warning(f"TokenRouter image generation failed, falling back to DALL-E: {_tr_err}")
        
        # Fallback: DALL-E direct API
        return await self._generate_via_dalle(prompt, model if _is_dalle else "dall-e-3", size, quality, style, n, response_format)

    async def _generate_via_tokenrouter(
        self,
        prompt: str,
        model: str = "auto",
    ) -> Optional[List[GeneratedImage]]:
        """Generate image via TokenRouter image models (chat completions with inline image output)."""
        # Select model
        if model == "auto" or model not in TOKENROUTER_IMAGE_MODELS:
            selected_model = "openai/gpt-5-image"  # Best quality
        else:
            selected_model = model

        logger.info(f"🎨 [TokenRouter] Generating image with {selected_model}: {prompt[:80]}")

        request = LLMRequest(
            messages=[
                {"role": "system", "content": "You are an image generation assistant. Generate the requested image. Respond ONLY with the image — no text explanation needed unless the user asks."},
                {"role": "user", "content": f"Generate an image: {prompt}"},
            ],
            model=selected_model,
            provider="tokenrouter",
            max_tokens=4096,
        )

        response = await _llm_client.complete(request, user_keys=self._user_keys)

        # Check for images in the response (GPT-5-image, etc. return images in dedicated field)
        if hasattr(response, 'images') and response.images:
            images = []
            for img_data in response.images:
                url = img_data.get("url", "")
                b64 = img_data.get("b64_json", "")
                if url and url.startswith("data:image/"):
                    # Extract base64 from data URL
                    b64_part = url.split(",", 1)[1] if "," in url else ""
                    images.append(GeneratedImage(
                        url=None,
                        base64_data=b64_part,
                        revised_prompt=prompt,
                        model=selected_model,
                        size="1024x1024",
                    ))
                elif url:
                    images.append(GeneratedImage(
                        url=url,
                        revised_prompt=prompt,
                        model=selected_model,
                        size="1024x1024",
                    ))
                elif b64:
                    images.append(GeneratedImage(
                        base64_data=b64,
                        revised_prompt=prompt,
                        model=selected_model,
                        size="1024x1024",
                    ))
            if images:
                logger.info(f"🎨 [TokenRouter] Generated {len(images)} image(s) with {selected_model} (via images field)")
                return images

        if not response.content:
            return None

        # Parse response — TokenRouter image models return images as:
        # 1. Markdown image URLs: ![description](https://...)
        # 2. Raw URLs to generated images
        # 3. Base64 inline data
        images = self._parse_image_response(response.content, selected_model)
        if images:
            logger.info(f"🎨 [TokenRouter] Generated {len(images)} image(s) with {selected_model}")
            return images

        # If the model responded with text only (e.g. describing the image), return as-is
        # with the content stored for the caller to display
        return [GeneratedImage(
            url=None,
            base64_data=None,
            revised_prompt=response.content,
            model=selected_model,
            size="1024x1024",
        )]

    def _parse_image_response(self, content: str, model: str) -> List[GeneratedImage]:
        """Parse image URLs/data from TokenRouter model response."""
        images = []

        # Pattern 1: Markdown images ![alt](url)
        md_imgs = re.findall(r'!\[([^\]]*)\]\(([^)]+)\)', content)
        for alt, url in md_imgs:
            if url.startswith(('http://', 'https://')):
                images.append(GeneratedImage(
                    url=url,
                    revised_prompt=alt or None,
                    model=model,
                    size="1024x1024",
                ))

        # Pattern 2: Standalone URLs to image files
        if not images:
            url_pattern = re.findall(r'(https?://[^\s"<>]+\.(?:png|jpg|jpeg|webp|gif)[^\s"<>]*)', content)
            for url in url_pattern:
                images.append(GeneratedImage(
                    url=url,
                    model=model,
                    size="1024x1024",
                ))

        # Pattern 3: Base64 data URIs
        if not images:
            b64_pattern = re.findall(r'data:image/[^;]+;base64,([A-Za-z0-9+/=]+)', content)
            for b64 in b64_pattern:
                images.append(GeneratedImage(
                    base64_data=b64,
                    model=model,
                    size="1024x1024",
                ))

        return images

    async def _generate_via_dalle(
        self,
        prompt: str,
        model: str = "dall-e-3",
        size: str = "1024x1024",
        quality: str = "standard",
        style: str = "vivid",
        n: int = 1,
        response_format: str = "url",
    ) -> List[GeneratedImage]:
        """Generate image via direct DALL-E API (fallback)."""
        if not self.openai_api_key:
            raise ValueError("No image generation available. TokenRouter failed and no OpenAI API key configured.")
        
        # Validate parameters for DALL-E 3
        if model == "dall-e-3":
            n = 1  # DALL-E 3 only supports 1 image at a time
            if size not in ["1024x1024", "1792x1024", "1024x1792"]:
                size = "1024x1024"
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                payload = {
                    "model": model,
                    "prompt": prompt,
                    "n": n,
                    "size": size,
                    "response_format": response_format,
                }
                
                # Add DALL-E 3 specific options
                if model == "dall-e-3":
                    payload["quality"] = quality
                    payload["style"] = style
                
                response = await client.post(
                    f"{self.base_url}/images/generations",
                    headers={
                        "Authorization": f"Bearer {self.openai_api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                
                images = []
                for item in data.get("data", []):
                    images.append(GeneratedImage(
                        url=item.get("url"),
                        base64_data=item.get("b64_json"),
                        revised_prompt=item.get("revised_prompt"),
                        model=model,
                        size=size,
                    ))
                
                logger.info(f"🎨 [DALL-E] Generated {len(images)} image(s) with {model}")
                return images
                
        except httpx.HTTPStatusError as e:
            error_detail = ""
            try:
                error_data = e.response.json()
                error_detail = error_data.get("error", {}).get("message", str(e))
            except:
                error_detail = str(e)
            
            logger.error(f"DALL-E generation failed: {error_detail}")
            raise ValueError(f"Image generation failed: {error_detail}")
        except Exception as e:
            logger.error(f"DALL-E generation error: {e}")
            raise
    
    async def edit_image(
        self,
        image_base64: str,
        mask_base64: Optional[str],
        prompt: str,
        size: str = "1024x1024",
        n: int = 1,
    ) -> List[GeneratedImage]:
        """
        Edit an existing image using DALL-E 2.
        
        Args:
            image_base64: Base64 encoded PNG image to edit
            mask_base64: Base64 encoded PNG mask (transparent areas will be edited)
            prompt: Description of the edit
            size: Output size
            n: Number of variations
        
        Returns:
            List of GeneratedImage objects
        """
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not configured")
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                files = {
                    "image": ("image.png", base64.b64decode(image_base64), "image/png"),
                    "prompt": (None, prompt),
                    "n": (None, str(n)),
                    "size": (None, size),
                }
                
                if mask_base64:
                    files["mask"] = ("mask.png", base64.b64decode(mask_base64), "image/png")
                
                response = await client.post(
                    f"{self.base_url}/images/edits",
                    headers={
                        "Authorization": f"Bearer {self.openai_api_key}",
                    },
                    files=files,
                )
                response.raise_for_status()
                data = response.json()
                
                images = []
                for item in data.get("data", []):
                    images.append(GeneratedImage(
                        url=item.get("url"),
                        base64_data=item.get("b64_json"),
                        model="dall-e-2",
                        size=size,
                    ))
                
                return images
                
        except Exception as e:
            logger.error(f"Image edit error: {e}")
            raise
    
    async def create_variation(
        self,
        image_base64: str,
        size: str = "1024x1024",
        n: int = 1,
    ) -> List[GeneratedImage]:
        """
        Create variations of an existing image using DALL-E 2.
        
        Args:
            image_base64: Base64 encoded PNG image
            size: Output size
            n: Number of variations
        
        Returns:
            List of GeneratedImage objects
        """
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not configured")
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                files = {
                    "image": ("image.png", base64.b64decode(image_base64), "image/png"),
                    "n": (None, str(n)),
                    "size": (None, size),
                }
                
                response = await client.post(
                    f"{self.base_url}/images/variations",
                    headers={
                        "Authorization": f"Bearer {self.openai_api_key}",
                    },
                    files=files,
                )
                response.raise_for_status()
                data = response.json()
                
                images = []
                for item in data.get("data", []):
                    images.append(GeneratedImage(
                        url=item.get("url"),
                        base64_data=item.get("b64_json"),
                        model="dall-e-2",
                        size=size,
                    ))
                
                return images
                
        except Exception as e:
            logger.error(f"Image variation error: {e}")
            raise
    
    def should_generate_image(self, message: str) -> bool:
        """Determine if a message is requesting image generation."""
        message_lower = message.lower()
        
        # Normalize common abbreviations
        message_lower = message_lower.replace(" u ", " you ")
        message_lower = message_lower.replace("can u ", "can you ")
        message_lower = message_lower.replace("could u ", "could you ")
        message_lower = message_lower.replace("pls ", "please ")
        message_lower = message_lower.replace("plz ", "please ")
        message_lower = message_lower.replace("pic ", "picture ")
        message_lower = message_lower.replace("pics ", "pictures ")
        
        # Direct image generation triggers
        image_triggers = [
            "generate image", "create image", "make image",
            "generate picture", "create picture", "make picture",
            "draw", "paint", "illustrate", "sketch",
            "generate art", "create art", "make art",
            "image of", "picture of", "photo of",
            "visualize", "render",
            "dall-e", "dalle",
            "generate a", "create a",
            # More flexible patterns
            "can you create", "can you make", "can you generate", "can you draw",
            "could you create", "could you make", "could you generate", "could you draw",
            "please create", "please make", "please generate", "please draw",
            "i want a picture", "i want an image", "i need a picture", "i need an image",
            "show me a picture", "show me an image",
            "make me a", "create me a", "draw me a",
            # Handle typos and variations
            "flower", "flover",  # Common request
            "picture flower", "picture flover",
        ]
        
        # Check for triggers
        has_trigger = any(trigger in message_lower for trigger in image_triggers)
        
        # Additional check: if message contains visual descriptors
        visual_descriptors = [
            "colorful", "realistic", "abstract", "cartoon", "anime",
            "photorealistic", "3d", "digital art", "oil painting",
            "watercolor", "pixel art", "minimalist", "surreal",
            "sunset", "sunrise", "landscape", "portrait", "scenery",
            "light", "lighting", "sunlight", "moonlight",
        ]
        has_visual = any(desc in message_lower for desc in visual_descriptors)
        
        # Check for image-related verbs with objects
        image_verbs = ["create", "make", "generate", "draw", "paint", "show"]
        image_nouns = ["picture", "image", "photo", "art", "illustration", "drawing", "painting"]
        has_verb_noun = any(verb in message_lower for verb in image_verbs) and any(noun in message_lower for noun in image_nouns)
        
        return has_trigger or has_verb_noun or (has_visual and len(message) > 20)
    
    def extract_image_prompt(self, message: str) -> str:
        """Extract the image generation prompt from a user message."""
        message_lower = message.lower()
        
        # Remove common prefixes
        prefixes_to_remove = [
            "generate an image of",
            "generate image of",
            "create an image of",
            "create image of",
            "make an image of",
            "make image of",
            "generate a picture of",
            "create a picture of",
            "make a picture of",
            "draw me",
            "draw a",
            "draw",
            "paint me",
            "paint a",
            "paint",
            "can you generate",
            "can you create",
            "can you make",
            "please generate",
            "please create",
            "please make",
            "i want",
            "i need",
            "show me",
        ]
        
        result = message
        for prefix in prefixes_to_remove:
            if message_lower.startswith(prefix):
                result = message[len(prefix):].strip()
                break
        
        return result.strip()


# Global instance
image_generation = ImageGenerationService()
