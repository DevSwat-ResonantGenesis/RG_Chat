"""
RAG (Retrieval-Augmented Generation) Engine
Simplified version for new backend - can be enhanced with full old backend features later.

Ported from old backend: ResonantGraphAIV0.1/backend/fastapi_app/services/rag.py
"""
from __future__ import annotations

import hashlib
import math
import logging
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)


class ChunkingStrategy:
    """Base class for text chunking strategies."""
    
    def chunk(self, text: str) -> List[str]:
        raise NotImplementedError


class FixedSizeChunking(ChunkingStrategy):
    """Fixed-size chunking with overlap."""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, text: str) -> List[str]:
        if not text:
            return []
        
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk.strip())
            start += self.chunk_size - self.overlap
        
        return chunks


class SentenceChunking(ChunkingStrategy):
    """Sentence-based chunking."""
    
    def __init__(self, max_chunk_size: int = 500):
        self.max_chunk_size = max_chunk_size
    
    def chunk(self, text: str) -> List[str]:
        if not text:
            return []
        
        sentences = []
        current_sentence = ""
        
        for char in text:
            current_sentence += char
            if char in '.!?':
                sentences.append(current_sentence.strip())
                current_sentence = ""
        
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            if current_size + sentence_size > self.max_chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks


def hash_text(text: str) -> str:
    """Generate SHA-1 hash of text."""
    return hashlib.sha1(text.encode('utf-8')).hexdigest()


def hash_to_coords(hex_hash: str) -> Tuple[float, float, float]:
    """Convert hash to 3D coordinates (from Hash Sphere)."""
    H = int(hex_hash, 16)
    
    mask53 = (1 << 53) - 1
    mask54 = (1 << 54) - 1
    
    x = H >> 107
    y = (H >> 54) & mask53
    z = H & mask54
    
    x_ = (2 * x) / mask53 - 1
    y_ = (2 * y) / mask53 - 1
    z_ = (2 * z) / mask54 - 1
    
    r = math.sqrt(x_ * x_ + y_ * y_ + z_ * z_)
    if r > 1:
        return (x_ / r, y_ / r, z_ / r)
    return (x_, y_, z_)


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(a * a for a in vec2))
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)


def euclidean_distance_3d(coord1: Tuple[float, float, float], coord2: Tuple[float, float, float]) -> float:
    """Calculate Euclidean distance between two 3D coordinates."""
    return math.sqrt(
        (coord1[0] - coord2[0]) ** 2 +
        (coord1[1] - coord2[1]) ** 2 +
        (coord1[2] - coord2[2]) ** 2
    )


class RAGEngine:
    """RAG (Retrieval-Augmented Generation) engine."""
    
    def __init__(self, chunking_strategy: ChunkingStrategy = None):
        self.chunking_strategy = chunking_strategy or FixedSizeChunking()
        self._embedding_cache: Dict[str, List[float]] = {}
    
    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text via ML service."""
        text_hash = hash_text(text)
        if text_hash in self._embedding_cache:
            return self._embedding_cache[text_hash]
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://ml_service:8000/ml/embed",
                    json={"text": text},
                    timeout=10.0,
                )
                if response.status_code == 200:
                    data = response.json()
                    embedding = data.get("vector", [])
                    if embedding:
                        self._embedding_cache[text_hash] = embedding
                    return embedding
        except Exception as e:
            logger.warning(f"Embedding error: {e}")
        
        return []
    
    def get_3d_coords(self, text: str) -> Tuple[float, float, float]:
        """Get 3D coordinates for text."""
        text_hash = hash_text(text)
        return hash_to_coords(text_hash)
    
    def classify_cluster(self, xyz: Tuple[float, float, float]) -> str:
        """Classify cluster based on 3D coordinates."""
        x, y, z = xyz
        if abs(x) > 0.6 and y > 0.3:
            return "alpha"
        elif z > 0.4:
            return "beta"
        else:
            return "gamma"
    
    async def retrieve_memories(
        self,
        user_id: str,
        org_id: str,
        query: str,
        top_k: int = 5,
        agent_hash: Optional[str] = None,
        team_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant memories from memory service."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://memory_service:8000/memory/retrieve",
                    json={
                        "user_id": user_id,
                        "org_id": org_id,
                        "query": query,
                        "limit": top_k,
                        "agent_hash": agent_hash,
                        "team_id": team_id,
                    },
                    timeout=5.0,
                )
                if response.status_code == 200:
                    return response.json()
        except Exception as e:
            logger.warning(f"Memory retrieval error: {e}")
        
        return []
    
    async def store_memory(
        self,
        user_id: str,
        org_id: str,
        content: str,
        chat_id: Optional[str] = None,
        role: Optional[str] = None,
        hash_value: Optional[str] = None,
        xyz: Optional[List[float]] = None,
        agent_hash: Optional[str] = None,
    ) -> bool:
        """Store a memory in memory service."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://memory_service:8000/memory/ingest",
                    json={
                        "user_id": user_id,
                        "org_id": org_id,
                        "content": content,
                        "chat_id": chat_id,
                        "role": role,
                        "hash": hash_value,
                        "xyz": xyz,
                        "agent_hash": agent_hash,
                        "source": "chat",
                    },
                    timeout=5.0,
                )
                return response.status_code == 200
        except Exception as e:
            logger.warning(f"Memory storage error: {e}")
        
        return False
    
    def build_context_from_memories(
        self,
        memories: List[Dict[str, Any]],
        max_tokens: int = 2000,
    ) -> str:
        """Build context string from retrieved memories."""
        if not memories:
            return ""
        
        context_parts = []
        total_chars = 0
        char_limit = max_tokens * 4  # Rough estimate: 4 chars per token
        
        for i, mem in enumerate(memories, 1):
            content = mem.get("content", "") or mem.get("anchor_text", "")
            if not content:
                continue
            
            # Truncate if needed
            if total_chars + len(content) > char_limit:
                remaining = char_limit - total_chars
                if remaining > 100:
                    content = content[:remaining] + "..."
                else:
                    break
            
            context_parts.append(f"[Memory {i}]\n{content}\n")
            total_chars += len(content)
        
        return "\n".join(context_parts)


# Global RAG engine instance
rag_engine = RAGEngine()
