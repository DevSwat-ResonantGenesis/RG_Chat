"""
DSID-P Integration for Resonant Chat
=====================================

Integrates Decentralized Secure Identity with Provenance (DSID-P) into the chat pipeline.

Features:
- Create DSIDs for each chat message
- Link message lineage (parent → child)
- Enable content verification via hash proofs
- Track conversation provenance
"""

import hashlib
import secrets
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)


# DSID Configuration
DSID_PREFIX = "dsid"
DSID_VERSION = 1


@dataclass
class MessageDSID:
    """DSID record for a chat message"""
    dsid: str
    entity_type: str  # "user_message" or "assistant_message"
    entity_id: str  # Message UUID
    content_hash: str
    parent_dsid: Optional[str] = None
    root_dsid: Optional[str] = None
    lineage_depth: int = 0
    chat_id: Optional[str] = None
    user_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class ConversationLineage:
    """Lineage chain for a conversation"""
    root_dsid: str
    chat_id: str
    messages: List[MessageDSID] = field(default_factory=list)
    merkle_root: Optional[str] = None


class DSIDIntegration:
    """
    DSID-P Integration for Chat Messages
    
    Implements Layer 1-2 of the HSU-Spec:
    - Cryptographic identity for messages
    - Content hashing with SHA-256
    - Lineage tracking (parent → child)
    - Merkle tree proofs
    """
    
    def __init__(self):
        self._message_dsids: Dict[str, MessageDSID] = {}  # entity_id -> DSID
        self._chat_lineages: Dict[str, ConversationLineage] = {}  # chat_id -> lineage
        self._blockchain_service_url = "http://blockchain_service:8000"
    
    def hash_content(self, content: Any) -> str:
        """Generate SHA-256 hash of content"""
        if isinstance(content, dict):
            content_str = str(sorted(content.items()))
        elif isinstance(content, bytes):
            content_str = content.decode('utf-8', errors='ignore')
        else:
            content_str = str(content)
        
        return hashlib.sha256(content_str.encode()).hexdigest()
    
    def generate_dsid(
        self,
        entity_type: str,
        content_hash: str,
    ) -> str:
        """
        Generate a DSID identifier.
        
        Format: dsid:v{version}:{entity_type}:{content_hash[:16]}:{random}
        """
        random_suffix = secrets.token_hex(4)
        return f"{DSID_PREFIX}:v{DSID_VERSION}:{entity_type}:{content_hash[:16]}:{random_suffix}"
    
    def create_message_dsid(
        self,
        message_id: str,
        content: str,
        role: str,  # "user" or "assistant"
        chat_id: str,
        user_id: str,
        parent_message_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MessageDSID:
        """
        Create a DSID for a chat message.
        
        Args:
            message_id: UUID of the message
            content: Message content
            role: "user" or "assistant"
            chat_id: Chat/conversation ID
            user_id: User ID
            parent_message_id: ID of the previous message (for lineage)
            metadata: Additional metadata (provider, model, etc.)
        
        Returns:
            MessageDSID object
        """
        # Hash the content
        content_hash = self.hash_content(content)
        
        # Determine entity type
        entity_type = f"{role}_message"
        
        # Get parent DSID if exists
        parent_dsid = None
        root_dsid = None
        lineage_depth = 0
        
        if parent_message_id and parent_message_id in self._message_dsids:
            parent = self._message_dsids[parent_message_id]
            parent_dsid = parent.dsid
            root_dsid = parent.root_dsid or parent.dsid
            lineage_depth = parent.lineage_depth + 1
        
        # Generate DSID
        dsid_str = self.generate_dsid(entity_type, content_hash)
        
        # If no parent, this is the root
        if not root_dsid:
            root_dsid = dsid_str
        
        # Create MessageDSID
        message_dsid = MessageDSID(
            dsid=dsid_str,
            entity_type=entity_type,
            entity_id=message_id,
            content_hash=content_hash,
            parent_dsid=parent_dsid,
            root_dsid=root_dsid,
            lineage_depth=lineage_depth,
            chat_id=chat_id,
            user_id=user_id,
            metadata=metadata or {},
        )
        
        # Store in cache
        self._message_dsids[message_id] = message_dsid
        
        # Update chat lineage
        if chat_id not in self._chat_lineages:
            self._chat_lineages[chat_id] = ConversationLineage(
                root_dsid=root_dsid,
                chat_id=chat_id,
            )
        self._chat_lineages[chat_id].messages.append(message_dsid)
        
        # Update Merkle root
        self._update_merkle_root(chat_id)
        
        logger.info(f"🔗 Created DSID: {dsid_str[:32]}... for {entity_type}")
        
        return message_dsid
    
    def _update_merkle_root(self, chat_id: str) -> None:
        """Update the Merkle root for a conversation"""
        if chat_id not in self._chat_lineages:
            return
        
        lineage = self._chat_lineages[chat_id]
        hashes = [msg.content_hash for msg in lineage.messages]
        
        if not hashes:
            return
        
        lineage.merkle_root = self.compute_merkle_root(hashes)
    
    def compute_merkle_root(self, hashes: List[str]) -> str:
        """Compute Merkle root from a list of hashes"""
        if not hashes:
            return hashlib.sha256(b"").hexdigest()
        
        if len(hashes) == 1:
            return hashes[0]
        
        working_hashes = hashes.copy()
        
        # Build tree
        while len(working_hashes) > 1:
            # Pad to even number at each level
            if len(working_hashes) % 2 == 1:
                working_hashes.append(working_hashes[-1])
            
            new_level = []
            for i in range(0, len(working_hashes), 2):
                combined = working_hashes[i] + working_hashes[i + 1]
                new_hash = hashlib.sha256(combined.encode()).hexdigest()
                new_level.append(new_hash)
            working_hashes = new_level
        
        return working_hashes[0]
    
    def compute_merkle_proof(
        self,
        message_id: str,
    ) -> List[Dict[str, str]]:
        """
        Compute Merkle proof for a message.
        
        Returns list of sibling hashes needed to verify the message.
        """
        if message_id not in self._message_dsids:
            return []
        
        message_dsid = self._message_dsids[message_id]
        chat_id = message_dsid.chat_id
        
        if not chat_id or chat_id not in self._chat_lineages:
            return []
        
        lineage = self._chat_lineages[chat_id]
        hashes = [msg.content_hash for msg in lineage.messages]
        target_hash = message_dsid.content_hash
        
        if target_hash not in hashes:
            return []
        
        proof = []
        working_hashes = hashes.copy()
        
        # Pad to even number
        if len(working_hashes) % 2 == 1:
            working_hashes.append(working_hashes[-1])
        
        target_idx = working_hashes.index(target_hash)
        
        while len(working_hashes) > 1:
            new_level = []
            for i in range(0, len(working_hashes), 2):
                if i == target_idx or i + 1 == target_idx:
                    sibling_idx = i + 1 if i == target_idx else i
                    proof.append({
                        "hash": working_hashes[sibling_idx],
                        "position": "right" if sibling_idx > target_idx else "left",
                    })
                    target_idx = i // 2
                
                combined = working_hashes[i] + working_hashes[i + 1]
                new_hash = hashlib.sha256(combined.encode()).hexdigest()
                new_level.append(new_hash)
            
            working_hashes = new_level
        
        return proof
    
    def verify_message(
        self,
        message_id: str,
        content: str,
    ) -> Tuple[bool, str]:
        """
        Verify a message's content against its DSID.
        
        Returns:
            Tuple of (is_valid, reason)
        """
        if message_id not in self._message_dsids:
            return False, "DSID not found for message"
        
        message_dsid = self._message_dsids[message_id]
        content_hash = self.hash_content(content)
        
        if content_hash != message_dsid.content_hash:
            return False, "Content hash mismatch - message may have been tampered"
        
        return True, "Verified - content matches DSID"
    
    def verify_merkle_proof(
        self,
        target_hash: str,
        merkle_root: str,
        proof: List[Dict[str, str]],
    ) -> bool:
        """Verify a Merkle proof"""
        current = target_hash
        
        for step in proof:
            sibling = step["hash"]
            if step["position"] == "left":
                combined = sibling + current
            else:
                combined = current + sibling
            current = hashlib.sha256(combined.encode()).hexdigest()
        
        return current == merkle_root
    
    def get_message_lineage(
        self,
        message_id: str,
    ) -> List[MessageDSID]:
        """Get the full lineage chain for a message"""
        if message_id not in self._message_dsids:
            return []
        
        lineage = []
        current_id = message_id
        
        while current_id:
            if current_id not in self._message_dsids:
                break
            
            dsid = self._message_dsids[current_id]
            lineage.append(dsid)
            
            # Find parent by DSID
            parent_dsid = dsid.parent_dsid
            if not parent_dsid:
                break
            
            # Find message with this DSID
            current_id = None
            for msg_id, msg_dsid in self._message_dsids.items():
                if msg_dsid.dsid == parent_dsid:
                    current_id = msg_id
                    break
        
        return lineage
    
    def get_conversation_lineage(
        self,
        chat_id: str,
    ) -> Optional[ConversationLineage]:
        """Get the full lineage for a conversation"""
        return self._chat_lineages.get(chat_id)
    
    def get_dsid_by_message(self, message_id: str) -> Optional[MessageDSID]:
        """Get DSID for a message ID"""
        return self._message_dsids.get(message_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get DSID integration statistics"""
        total_messages = len(self._message_dsids)
        total_chats = len(self._chat_lineages)
        
        user_messages = sum(1 for d in self._message_dsids.values() if d.entity_type == "user_message")
        assistant_messages = sum(1 for d in self._message_dsids.values() if d.entity_type == "assistant_message")
        
        avg_depth = 0
        if total_messages > 0:
            avg_depth = sum(d.lineage_depth for d in self._message_dsids.values()) / total_messages
        
        return {
            "total_dsids": total_messages,
            "total_conversations": total_chats,
            "user_messages": user_messages,
            "assistant_messages": assistant_messages,
            "average_lineage_depth": round(avg_depth, 2),
        }
    
    async def sync_to_blockchain(
        self,
        message_dsid: MessageDSID,
    ) -> Optional[Dict[str, Any]]:
        """
        Sync a message DSID to the blockchain service.
        
        Creates a DSID record and transaction in the blockchain.
        """
        try:
            async with httpx.AsyncClient() as client:
                # Create DSID in blockchain service
                response = await client.post(
                    f"{self._blockchain_service_url}/blockchain/dsid",
                    json={
                        "entity_type": message_dsid.entity_type,
                        "entity_id": message_dsid.entity_id,
                        "content_hash": message_dsid.content_hash,
                        "parent_dsid": message_dsid.parent_dsid,
                        "metadata": {
                            "chat_id": message_dsid.chat_id,
                            "user_id": message_dsid.user_id,
                            **message_dsid.metadata,
                        },
                    },
                    timeout=5.0,
                )
                
                if response.status_code == 200 or response.status_code == 201:
                    result = response.json()
                    logger.info(f"📦 Synced DSID to blockchain: {message_dsid.dsid[:32]}...")
                    return result
                else:
                    logger.warning(f"Blockchain sync failed: {response.status_code}")
                    return None
                    
        except Exception as e:
            logger.warning(f"Blockchain sync error (non-critical): {e}")
            return None


# Global instance
dsid_integration = DSIDIntegration()


# Convenience functions
def create_message_dsid(
    message_id: str,
    content: str,
    role: str,
    chat_id: str,
    user_id: str,
    parent_message_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> MessageDSID:
    """Create a DSID for a chat message"""
    return dsid_integration.create_message_dsid(
        message_id=message_id,
        content=content,
        role=role,
        chat_id=chat_id,
        user_id=user_id,
        parent_message_id=parent_message_id,
        metadata=metadata,
    )


def verify_message(message_id: str, content: str) -> Tuple[bool, str]:
    """Verify a message's content"""
    return dsid_integration.verify_message(message_id, content)


def get_message_lineage(message_id: str) -> List[MessageDSID]:
    """Get lineage for a message"""
    return dsid_integration.get_message_lineage(message_id)


def get_merkle_proof(message_id: str) -> List[Dict[str, str]]:
    """Get Merkle proof for a message"""
    return dsid_integration.compute_merkle_proof(message_id)
