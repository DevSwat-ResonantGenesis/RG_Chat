"""
PMI-Layer Manager (Simplified)
===============================

Blockchain-like memory integrity layer for creating immutable memory events.

Ported from old backend: ResonantGraphAIV0.1/backend/fastapi_app/services/pmi_layer/pmi_manager.py
"""
from __future__ import annotations

import hashlib
import logging
from typing import Optional, Dict, Any, List
from uuid import UUID
from datetime import datetime

logger = logging.getLogger(__name__)


class PMIManager:
    """
    PMI-Layer Manager (Simplified)
    
    Creates blockchain-like memory events with cryptographic hashing
    for memory integrity and auditability.
    """
    
    # Event types
    EVENT_PROMPT = 0
    EVENT_RESPONSE = 1
    EVENT_SEMANTIC_UPDATE = 2
    EVENT_MEMORY_ANCHOR = 3
    
    def __init__(self):
        self._blocks: List[Dict[str, Any]] = []
        self._state_hash: Optional[str] = None
    
    def _hash(self, data: str) -> str:
        """Generate SHA-256 hash."""
        return hashlib.sha256(data.encode('utf-8')).hexdigest()
    
    def _derive_root_key(
        self,
        user_id: str,
        session_id: str,
        timestamp_ms: int,
        model_fingerprint: str = "default"
    ) -> str:
        """Derive a unique root key for this session."""
        combined = f"{user_id}:{session_id}:{timestamp_ms}:{model_fingerprint}"
        return self._hash(combined)
    
    def _derive_universe_id(
        self,
        root_key: str,
        time_slice: int,
        domain_id: Optional[str] = None
    ) -> str:
        """Derive a universe ID from root key and time slice."""
        combined = f"{root_key}:{time_slice}:{domain_id or 'default'}"
        return self._hash(combined)[:32]
    
    def _calculate_time_slice(self, timestamp_ms: int, slice_duration_ms: int = 3600000) -> int:
        """Calculate time slice (default: 1 hour slices)."""
        return timestamp_ms // slice_duration_ms
    
    def _calculate_state_hash(
        self,
        message_hash: str,
        embedding_hash: Optional[str],
        semantic_hash: Optional[str],
        prev_state_hash: Optional[str]
    ) -> str:
        """Calculate new state hash from components."""
        components = [
            message_hash,
            embedding_hash or "null",
            semantic_hash or "null",
            prev_state_hash or "genesis"
        ]
        combined = ":".join(components)
        return self._hash(combined)
    
    def create_memory_event(
        self,
        user_id: str,
        org_id: str,
        chat_id: Optional[str],
        session_id: str,
        message_text: str,
        event_type: int,
        embedding_bytes: Optional[bytes] = None,
        semantic_vector: Optional[bytes] = None,
        model_name: str = "unknown",
        model_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a memory event and store as MicroBlock.
        
        Args:
            user_id: User ID
            org_id: Organization ID
            chat_id: Optional chat ID
            session_id: Session identifier
            message_text: Message text (will be hashed)
            event_type: Event type (0=prompt, 1=response, 2=semantic_update)
            embedding_bytes: Optional embedding bytes
            semantic_vector: Optional semantic vector
            model_name: Model name
            model_version: Optional model version
        
        Returns:
            Event result dictionary
        """
        try:
            timestamp_ms = int(datetime.utcnow().timestamp() * 1000)
            
            # Calculate hashes
            message_hash = self._hash(message_text)
            embedding_hash = self._hash(embedding_bytes.hex()) if embedding_bytes else None
            semantic_hash = self._hash(semantic_vector.hex()) if semantic_vector else None
            
            # Derive keys
            model_fingerprint = f"{model_name}:{model_version or 'latest'}"
            root_key = self._derive_root_key(
                user_id=user_id,
                session_id=session_id,
                timestamp_ms=timestamp_ms,
                model_fingerprint=model_fingerprint
            )
            
            time_slice = self._calculate_time_slice(timestamp_ms)
            root_universe_id = self._derive_universe_id(
                root_key=root_key,
                time_slice=time_slice
            )
            
            # Create block
            block_data = f"{root_key}:{message_hash}:{timestamp_ms}:{event_type}"
            block_hash = self._hash(block_data)
            
            # Calculate new state hash
            new_state_hash = self._calculate_state_hash(
                message_hash=message_hash,
                embedding_hash=embedding_hash,
                semantic_hash=semantic_hash,
                prev_state_hash=self._state_hash
            )
            
            # Store block
            block = {
                "block_hash": block_hash,
                "root_key": root_key,
                "root_universe_id": root_universe_id,
                "message_hash": message_hash,
                "embedding_hash": embedding_hash,
                "semantic_hash": semantic_hash,
                "event_type": event_type,
                "timestamp_ms": timestamp_ms,
                "timestamp": datetime.utcnow().isoformat(),
                "user_id": user_id,
                "org_id": org_id,
                "chat_id": chat_id,
                "prev_state_hash": self._state_hash,
                "state_hash": new_state_hash
            }
            
            self._blocks.append(block)
            self._state_hash = new_state_hash
            
            logger.info(f"📦 PMI: Created block {block_hash[:16]}... for event type {event_type}")
            
            return {
                "root_key": root_key,
                "root_universe_id": root_universe_id,
                "block_hash": block_hash,
                "state_hash": new_state_hash,
                "event_type": event_type,
                "timestamp": block["timestamp"]
            }
            
        except Exception as e:
            logger.error(f"Error creating memory event: {e}", exc_info=True)
            return {
                "error": str(e),
                "event_type": event_type
            }
    
    def verify_chain(self) -> bool:
        """Verify the integrity of the block chain."""
        if not self._blocks:
            return True
        
        try:
            prev_state = None
            for block in self._blocks:
                # Verify state hash chain
                expected_state = self._calculate_state_hash(
                    message_hash=block["message_hash"],
                    embedding_hash=block.get("embedding_hash"),
                    semantic_hash=block.get("semantic_hash"),
                    prev_state_hash=prev_state
                )
                
                if block["state_hash"] != expected_state:
                    logger.warning(f"Chain verification failed at block {block['block_hash'][:16]}...")
                    return False
                
                prev_state = block["state_hash"]
            
            return True
            
        except Exception as e:
            logger.error(f"Error verifying chain: {e}")
            return False
    
    def get_block_count(self) -> int:
        """Get the number of blocks in the chain."""
        return len(self._blocks)
    
    def get_current_state_hash(self) -> Optional[str]:
        """Get the current state hash."""
        return self._state_hash
    
    def get_blocks_by_chat(self, chat_id: str) -> List[Dict[str, Any]]:
        """Get all blocks for a specific chat."""
        return [b for b in self._blocks if b.get("chat_id") == chat_id]


# Global instance
pmi_manager = PMIManager()
