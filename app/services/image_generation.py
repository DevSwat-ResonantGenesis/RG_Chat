"""Image Generation Service for Resonant Chat.

Provides AI image generation capabilities using:
- OpenAI DALL-E 3 (primary)
- OpenAI DALL-E 2 (fallback)
- Stability AI (future)
"""
import os
import logging
import httpx
import base64
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


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
    """Image generation service using OpenAI DALL-E."""
    
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.stability_api_key = os.getenv("STABILITY_API_KEY")
        self.timeout = 60.0  # Image generation can take time
        self.base_url = "https://api.openai.com/v1"
    
    def set_api_key(self, openai_key: Optional[str] = None, stability_key: Optional[str] = None):
        """Set API keys dynamically (for user-provided keys)."""
        if openai_key:
            self.openai_api_key = openai_key
        if stability_key:
            self.stability_api_key = stability_key
    
    async def generate(
        self,
        prompt: str,
        model: str = "dall-e-3",
        size: str = "1024x1024",
        quality: str = "standard",
        style: str = "vivid",
        n: int = 1,
        response_format: str = "url",  # "url" or "b64_json"
    ) -> List[GeneratedImage]:
        """
        Generate images using DALL-E.
        
        Args:
            prompt: Text description of the image to generate
            model: "dall-e-3" or "dall-e-2"
            size: Image size (1024x1024, 1792x1024, 1024x1792 for DALL-E 3)
            quality: "standard" or "hd" (DALL-E 3 only)
            style: "vivid" or "natural" (DALL-E 3 only)
            n: Number of images (1 for DALL-E 3, 1-10 for DALL-E 2)
            response_format: "url" or "b64_json"
        
        Returns:
            List of GeneratedImage objects
        """
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not configured. Please add your API key in settings.")
        
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
                
                logger.info(f"🎨 Generated {len(images)} image(s) with {model}")
                return images
                
        except httpx.HTTPStatusError as e:
            error_detail = ""
            try:
                error_data = e.response.json()
                error_detail = error_data.get("error", {}).get("message", str(e))
            except:
                error_detail = str(e)
            
            logger.error(f"Image generation failed: {error_detail}")
            raise ValueError(f"Image generation failed: {error_detail}")
        except Exception as e:
            logger.error(f"Image generation error: {e}")
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
