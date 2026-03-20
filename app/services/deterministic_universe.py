"""
Deterministic Universe Service
===============================

Derives deterministic universe components from seeds for reproducible hashing.

Ported from old backend: ResonantGraphAIV0.1/backend/fastapi_app/services/deterministic_universe/
"""
from __future__ import annotations

import hashlib
import hmac
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np

from .resonance_hashing import ResonanceHasher

logger = logging.getLogger(__name__)


class UniverseDeriver:
    """
    Derives deterministic universe components from seeds.
    
    Components:
    - universe_id: Unique identifier
    - hash_offset: Deterministic hash transformation
    - embedding_offset: Deterministic embedding transformation
    - cluster_centers: Fixed cluster geometry
    """
    
    def __init__(self, seed: str):
        self.seed = seed
        self.universe = self._derive_universe(seed)
    
    def _derive_universe(self, seed: str) -> Dict[str, Any]:
        """Derive deterministic universe from seed."""
        master_key = hashlib.sha256(seed.encode()).digest()
        universe_id = hashlib.sha256(master_key).hexdigest()[:16]
        hash_offset = self._generate_hash_offset(master_key)
        embedding_offset = self._generate_embedding_offset(master_key)
        cluster_centers = self._generate_cluster_centers(master_key, num_clusters=10)
        
        return {
            "universe_id": universe_id,
            "hash_offset": hash_offset,
            "embedding_offset": embedding_offset,
            "cluster_centers": cluster_centers,
            "master_key_hash": hashlib.sha256(master_key).hexdigest()[:16]
        }
    
    def _generate_hash_offset(self, master_key: bytes) -> bytes:
        """Generate deterministic hash offset from master key."""
        offset = hmac.new(master_key, b"hash_offset", hashlib.sha256).digest()
        return offset[:32]
    
    def _generate_embedding_offset(self, master_key: bytes, dimension: int = 384) -> np.ndarray:
        """Generate deterministic embedding offset from master key."""
        prng_seed = hmac.new(master_key, b"embedding_offset", hashlib.sha256).digest()
        seed_int = int.from_bytes(prng_seed[:4], 'big')
        rng = np.random.default_rng(seed_int)
        offset = rng.standard_normal(dimension)
        offset = offset / np.linalg.norm(offset) * 0.1
        return offset
    
    def _generate_cluster_centers(self, master_key: bytes, num_clusters: int = 10) -> np.ndarray:
        """Generate deterministic cluster centers from master key."""
        prng_seed = hmac.new(master_key, b"cluster_centers", hashlib.sha256).digest()
        seed_int = int.from_bytes(prng_seed[:4], 'big')
        rng = np.random.default_rng(seed_int)
        centers = rng.random((num_clusters, 3))
        centers = (centers - centers.min()) / (centers.max() - centers.min() + 1e-10)
        return centers
    
    def get_universe_id(self) -> str:
        return self.universe["universe_id"]
    
    def get_hash_offset(self) -> bytes:
        return self.universe["hash_offset"]
    
    def get_embedding_offset(self) -> np.ndarray:
        return self.universe["embedding_offset"]
    
    def get_cluster_centers(self) -> np.ndarray:
        return self.universe["cluster_centers"]
    
    def get_universe_info(self) -> Dict[str, Any]:
        return {
            "universe_id": self.universe["universe_id"],
            "hash_offset_length": len(self.universe["hash_offset"]),
            "embedding_offset_shape": self.universe["embedding_offset"].shape,
            "cluster_centers_shape": self.universe["cluster_centers"].shape,
            "master_key_hash": self.universe["master_key_hash"]
        }


class DeterministicResonanceHasher(ResonanceHasher):
    """
    Deterministic Resonance Hasher
    
    Extends ResonanceHasher with seed-based deterministic transformations.
    Same seed + same text → same hash (100% reproducible)
    """
    
    def __init__(self, anchor_seed: Optional[str] = None):
        super().__init__()
        self.anchor_seed = anchor_seed
        
        if anchor_seed:
            try:
                self.universe = UniverseDeriver(anchor_seed)
                self.is_deterministic = True
                logger.debug(f"✅ Initialized deterministic hasher with universe_id: {self.universe.get_universe_id()}")
            except Exception as e:
                logger.warning(f"Failed to derive universe from seed: {e}")
                self.universe = None
                self.is_deterministic = False
        else:
            self.universe = None
            self.is_deterministic = False
    
    def hash_text(self, text: str, context: Optional[str] = None) -> str:
        """Create deterministic resonance hash from text."""
        base_hash = super().hash_text(text, context)
        
        if self.is_deterministic and self.universe:
            transformed_hash = self._apply_deterministic_transform(base_hash)
            return transformed_hash
        
        return base_hash
    
    def _apply_deterministic_transform(self, hash_str: str) -> str:
        """Apply deterministic transformation to hash."""
        try:
            offset = self.universe.get_hash_offset()
            hash_bytes = bytes.fromhex(hash_str[:64])
            
            if len(offset) < len(hash_bytes):
                offset = (offset * ((len(hash_bytes) // len(offset)) + 1))[:len(hash_bytes)]
            elif len(offset) > len(hash_bytes):
                offset = offset[:len(hash_bytes)]
            
            transformed = bytes(a ^ b for a, b in zip(hash_bytes, offset))
            return transformed.hex()
        
        except Exception as e:
            logger.error(f"Error applying deterministic transform: {e}", exc_info=True)
            return hash_str
    
    def hash_to_universe_id(self, text: str) -> str:
        """Convert text to Universe ID with deterministic transformation."""
        if self.is_deterministic:
            hash_value = self.hash_text(text)
            universe_id = hashlib.sha256(hash_value.encode()).hexdigest()[:16]
            return universe_id
        else:
            return super().hash_to_universe_id(text)


# Global instance (non-deterministic by default)
deterministic_hasher = DeterministicResonanceHasher()
