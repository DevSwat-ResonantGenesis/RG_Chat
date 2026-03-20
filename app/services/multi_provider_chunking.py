"""
Multi-Provider Chunking Service
================================

Handles large text inputs by splitting them across multiple AI providers.
Each provider processes a chunk, then results are combined.

This solves the token limit problem for large documents.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ChunkResult:
    """Result from processing a single chunk."""
    chunk_index: int
    provider: str
    content: str
    success: bool
    error: Optional[str] = None


@dataclass
class ChunkingConfig:
    """Configuration for chunking behavior."""
    max_tokens_per_chunk: int = 4000  # Conservative limit
    overlap_tokens: int = 200  # Overlap between chunks for context
    min_chunk_size: int = 100  # Don't create tiny chunks


class MultiProviderChunker:
    """
    Splits large inputs across multiple providers for parallel processing.
    
    Strategy:
    1. Estimate token count (4 chars ≈ 1 token)
    2. If within single provider limit, use one provider
    3. If exceeds limit, split into chunks
    4. Assign chunks to available providers round-robin
    5. Process in parallel
    6. Combine results in order
    """
    
    def __init__(self):
        self.config = ChunkingConfig()
        self._router = None
    
    def set_router(self, router):
        """Set the MultiAIRouter instance."""
        self._router = router
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough: 4 chars ≈ 1 token)."""
        return len(text) // 4
    
    def needs_chunking(self, text: str) -> bool:
        """Check if text needs to be chunked."""
        return self.estimate_tokens(text) > self.config.max_tokens_per_chunk
    
    def split_into_chunks(self, text: str) -> List[str]:
        """
        Split text into chunks with overlap for context continuity.
        
        Tries to split at paragraph or sentence boundaries.
        """
        estimated_tokens = self.estimate_tokens(text)
        
        if estimated_tokens <= self.config.max_tokens_per_chunk:
            return [text]
        
        # Calculate number of chunks needed
        effective_chunk_size = self.config.max_tokens_per_chunk - self.config.overlap_tokens
        num_chunks = (estimated_tokens // effective_chunk_size) + 1
        
        # Convert token estimate to character positions
        chars_per_chunk = (self.config.max_tokens_per_chunk * 4)
        overlap_chars = (self.config.overlap_tokens * 4)
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chars_per_chunk, len(text))
            
            # Try to find a good break point (paragraph or sentence)
            if end < len(text):
                # Look for paragraph break
                para_break = text.rfind('\n\n', start + chars_per_chunk // 2, end)
                if para_break > start:
                    end = para_break + 2
                else:
                    # Look for sentence break
                    for punct in ['. ', '! ', '? ', '.\n', '!\n', '?\n']:
                        sent_break = text.rfind(punct, start + chars_per_chunk // 2, end)
                        if sent_break > start:
                            end = sent_break + len(punct)
                            break
            
            chunk = text[start:end].strip()
            if len(chunk) >= self.config.min_chunk_size:
                chunks.append(chunk)
            
            # Move start with overlap
            start = end - overlap_chars if end < len(text) else end
        
        logger.info(f"📦 Split text into {len(chunks)} chunks")
        return chunks
    
    async def process_with_chunking(
        self,
        text: str,
        task_prompt: str,
        context: List[Dict],
        available_providers: List[str],
        user_api_keys: Optional[Dict[str, str]] = None,
    ) -> Tuple[str, List[ChunkResult]]:
        """
        Process large text by chunking across multiple providers.
        
        Args:
            text: The large text to process
            task_prompt: What to do with the text (e.g., "Summarize this")
            context: Conversation context
            available_providers: List of available provider names
            user_api_keys: User's API keys for BYOK
        
        Returns:
            Tuple of (combined_result, chunk_results)
        """
        if not self._router:
            raise RuntimeError("Router not set. Call set_router() first.")
        
        # Set user API keys if provided
        if user_api_keys:
            self._router.set_user_api_keys(user_api_keys)
        
        # Check if chunking is needed
        if not self.needs_chunking(text):
            logger.info("📄 Text fits in single provider, no chunking needed")
            result = await self._process_single(text, task_prompt, context, available_providers[0])
            return result.content, [result]
        
        # Split into chunks
        chunks = self.split_into_chunks(text)
        logger.info(f"📦 Processing {len(chunks)} chunks across {len(available_providers)} providers")
        
        # Assign chunks to providers round-robin
        chunk_assignments: List[Tuple[int, str, str]] = []
        for i, chunk in enumerate(chunks):
            provider = available_providers[i % len(available_providers)]
            chunk_assignments.append((i, provider, chunk))
        
        # Process chunks in parallel
        tasks = [
            self._process_chunk(idx, provider, chunk, task_prompt, context, len(chunks))
            for idx, provider, chunk in chunk_assignments
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to ChunkResult
        chunk_results: List[ChunkResult] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                chunk_results.append(ChunkResult(
                    chunk_index=i,
                    provider=chunk_assignments[i][1],
                    content="",
                    success=False,
                    error=str(result)
                ))
            else:
                chunk_results.append(result)
        
        # Sort by chunk index and combine
        chunk_results.sort(key=lambda x: x.chunk_index)
        
        # Combine successful results
        combined_parts = []
        for result in chunk_results:
            if result.success and result.content:
                combined_parts.append(result.content)
            elif not result.success:
                logger.warning(f"Chunk {result.chunk_index} failed: {result.error}")
        
        if not combined_parts:
            return "Failed to process the document. All chunks failed.", chunk_results
        
        # Add section markers for clarity
        if len(combined_parts) > 1:
            combined = "\n\n---\n\n".join([
                f"**Part {i+1}:**\n{part}" 
                for i, part in enumerate(combined_parts)
            ])
        else:
            combined = combined_parts[0]
        
        return combined, chunk_results
    
    async def _process_single(
        self,
        text: str,
        task_prompt: str,
        context: List[Dict],
        provider: str,
    ) -> ChunkResult:
        """Process text with a single provider."""
        try:
            full_prompt = f"{task_prompt}\n\n{text}"
            result = await self._router.route_query(
                message=full_prompt,
                context=context,
                preferred_provider=provider,
            )
            return ChunkResult(
                chunk_index=0,
                provider=provider,
                content=result.get("response", ""),
                success=True
            )
        except Exception as e:
            return ChunkResult(
                chunk_index=0,
                provider=provider,
                content="",
                success=False,
                error=str(e)
            )
    
    async def _process_chunk(
        self,
        chunk_index: int,
        provider: str,
        chunk: str,
        task_prompt: str,
        context: List[Dict],
        total_chunks: int,
    ) -> ChunkResult:
        """Process a single chunk with a specific provider."""
        try:
            # Add chunk context to the prompt
            chunk_prompt = (
                f"{task_prompt}\n\n"
                f"[Processing part {chunk_index + 1} of {total_chunks}]\n\n"
                f"{chunk}"
            )
            
            result = await self._router.route_query(
                message=chunk_prompt,
                context=context,
                preferred_provider=provider,
            )
            
            return ChunkResult(
                chunk_index=chunk_index,
                provider=provider,
                content=result.get("response", ""),
                success=True
            )
        except Exception as e:
            logger.error(f"Chunk {chunk_index} failed with {provider}: {e}")
            return ChunkResult(
                chunk_index=chunk_index,
                provider=provider,
                content="",
                success=False,
                error=str(e)
            )
    
    def get_chunking_info(self, text: str) -> Dict[str, Any]:
        """Get information about how text would be chunked."""
        estimated_tokens = self.estimate_tokens(text)
        needs_chunking = self.needs_chunking(text)
        
        if needs_chunking:
            chunks = self.split_into_chunks(text)
            return {
                "estimated_tokens": estimated_tokens,
                "needs_chunking": True,
                "num_chunks": len(chunks),
                "chunk_sizes": [len(c) for c in chunks],
                "max_tokens_per_chunk": self.config.max_tokens_per_chunk,
            }
        else:
            return {
                "estimated_tokens": estimated_tokens,
                "needs_chunking": False,
                "num_chunks": 1,
                "max_tokens_per_chunk": self.config.max_tokens_per_chunk,
            }


# Global instance
multi_provider_chunker = MultiProviderChunker()
