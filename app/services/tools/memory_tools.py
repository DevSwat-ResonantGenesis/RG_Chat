"""
Memory & Hash Sphere Tools
============================

Real executors for all memory-related tools.
Each tool calls the memory_service at MEMORY_SERVICE_URL.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

import httpx

from .base import BaseIntegrationSkill

logger = logging.getLogger(__name__)

MEMORY_SERVICE_URL = os.getenv("MEMORY_SERVICE_URL", "http://memory_service:8000")


class MemoryReadTool(BaseIntegrationSkill):
    skill_id = "memory_read"
    skill_name = "Memory Read"
    api_key_names = []
    intent_keywords = ["read memory", "recall", "what do you remember"]

    async def execute(self, message: str, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    f"{MEMORY_SERVICE_URL}/memory/rag/memories",
                    params={"user_id": user_id, "limit": 20},
                )
                resp.raise_for_status()
                memories = resp.json().get("memories", [])
                if not memories:
                    return {"success": True, "action": "memory_read", "summary": "No memories stored yet.", "count": 0}
                summary = f"**{len(memories)} memories found:**\n\n"
                for i, m in enumerate(memories[:20], 1):
                    content = (m.get("content") or "")[:150]
                    summary += f"{i}. {content}\n"
                return {"success": True, "action": "memory_read", "summary": summary, "count": len(memories)}
        except Exception as e:
            return {"success": False, "action": "memory_read", "error": str(e)[:300]}


class MemoryWriteTool(BaseIntegrationSkill):
    skill_id = "memory_write"
    skill_name = "Memory Write"
    api_key_names = []
    intent_keywords = ["remember this", "save memory", "store this"]

    async def execute(self, message: str, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    f"{MEMORY_SERVICE_URL}/memory/ingest",
                    json={"text": message, "user_id": user_id, "source": "resonant_chat"},
                )
                resp.raise_for_status()
                data = resp.json()
                return {
                    "success": True,
                    "action": "memory_write",
                    "summary": f"**Memory saved.** ID: {data.get('id', 'unknown')}",
                }
        except Exception as e:
            return {"success": False, "action": "memory_write", "error": str(e)[:300]}


class MemoryStatsTool(BaseIntegrationSkill):
    skill_id = "memory_stats"
    skill_name = "Memory Stats"
    api_key_names = []
    intent_keywords = ["memory stats", "memory count", "how many memories"]

    async def execute(self, message: str, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    f"{MEMORY_SERVICE_URL}/memory/stats",
                    params={"user_id": user_id},
                )
                resp.raise_for_status()
                stats = resp.json()
                total = stats.get("total_memories", 0)
                anchors = stats.get("total_anchors", stats.get("total_clusters", 0))
                summary = (
                    f"**Memory Stats:**\n\n"
                    f"- Total memories: {total}\n"
                    f"- Anchors/clusters: {anchors}\n"
                )
                return {"success": True, "action": "memory_stats", "summary": summary, "stats": stats}
        except Exception as e:
            return {"success": False, "action": "memory_stats", "error": str(e)[:300]}


class HashSphereSearchTool(BaseIntegrationSkill):
    skill_id = "hash_sphere_search"
    skill_name = "Hash Sphere Search"
    api_key_names = []
    intent_keywords = ["hash sphere search", "semantic search memory"]

    async def execute(self, message: str, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post(
                    f"{MEMORY_SERVICE_URL}/memory/hash-sphere/extract",
                    json={
                        "query": message,
                        "user_id": user_id,
                        "limit": 10,
                        "use_anchors": True,
                        "use_proximity": True,
                        "use_resonance": True,
                        "use_rag_fallback": True,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                memories = data.get("memories", [])
                if not memories:
                    return {"success": True, "action": "hash_sphere_search", "summary": "No matching memories.", "count": 0}
                summary = f"**Hash Sphere — {len(memories)} results:**\n\n"
                for i, m in enumerate(memories[:10], 1):
                    content = (m.get("content") or "")[:200]
                    score = m.get("hybrid_score", 0)
                    summary += f"{i}. {content}... (score: {score:.2f})\n\n"
                return {"success": True, "action": "hash_sphere_search", "summary": summary, "count": len(memories)}
        except Exception as e:
            return {"success": False, "action": "hash_sphere_search", "error": str(e)[:300]}


class HashSphereAnchorTool(BaseIntegrationSkill):
    skill_id = "hash_sphere_anchor"
    skill_name = "Hash Sphere Anchor"
    api_key_names = []
    intent_keywords = ["create anchor", "anchor memory", "pin memory"]

    async def execute(self, message: str, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    f"{MEMORY_SERVICE_URL}/memory/hash-sphere/anchor",
                    json={"text": message, "user_id": user_id},
                )
                resp.raise_for_status()
                data = resp.json()
                return {
                    "success": True,
                    "action": "hash_sphere_anchor",
                    "summary": f"**Anchor created.** Hash: {data.get('hash', 'N/A')}",
                }
        except Exception as e:
            return {"success": False, "action": "hash_sphere_anchor", "error": str(e)[:300]}


class HashSphereListAnchorsTool(BaseIntegrationSkill):
    skill_id = "hash_sphere_list_anchors"
    skill_name = "List Anchors"
    api_key_names = []
    intent_keywords = ["list anchors", "show anchors", "my anchors"]

    async def execute(self, message: str, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    f"{MEMORY_SERVICE_URL}/memory/hash-sphere/anchors",
                    params={"user_id": user_id},
                )
                resp.raise_for_status()
                anchors = resp.json().get("anchors", [])
                if not anchors:
                    return {"success": True, "action": "hash_sphere_list_anchors", "summary": "No anchors yet.", "count": 0}
                summary = f"**{len(anchors)} anchors:**\n\n"
                for a in anchors[:20]:
                    label = a.get("label", a.get("hash", "?"))[:80]
                    summary += f"- {label}\n"
                return {"success": True, "action": "hash_sphere_list_anchors", "summary": summary, "count": len(anchors)}
        except Exception as e:
            return {"success": False, "action": "hash_sphere_list_anchors", "error": str(e)[:300]}


class HashSphereHashTool(BaseIntegrationSkill):
    skill_id = "hash_sphere_hash"
    skill_name = "Hash Sphere Hash"
    api_key_names = []
    intent_keywords = ["hash text", "compute hash", "resonance hash"]

    async def execute(self, message: str, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{MEMORY_SERVICE_URL}/memory/hash-sphere/hash",
                    json={"text": message, "user_id": user_id},
                )
                resp.raise_for_status()
                data = resp.json()
                return {
                    "success": True,
                    "action": "hash_sphere_hash",
                    "summary": f"**Hash:** `{data.get('hash', 'N/A')}`\n**XYZ:** {data.get('xyz', 'N/A')}",
                    "hash": data.get("hash"),
                    "xyz": data.get("xyz"),
                }
        except Exception as e:
            return {"success": False, "action": "hash_sphere_hash", "error": str(e)[:300]}


class HashSphereResonanceTool(BaseIntegrationSkill):
    skill_id = "hash_sphere_resonance"
    skill_name = "Hash Sphere Resonance"
    api_key_names = []
    intent_keywords = ["resonance score", "compute resonance"]

    async def execute(self, message: str, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{MEMORY_SERVICE_URL}/memory/hash-sphere/resonance",
                    json={"text": message, "user_id": user_id},
                )
                resp.raise_for_status()
                data = resp.json()
                score = data.get("resonance_score", data.get("score", 0))
                return {
                    "success": True,
                    "action": "hash_sphere_resonance",
                    "summary": f"**Resonance score:** {score:.4f}",
                    "resonance_score": score,
                }
        except Exception as e:
            return {"success": False, "action": "hash_sphere_resonance", "error": str(e)[:300]}


# ── Registry ──

MEMORY_TOOLS = {
    "memory_read": MemoryReadTool(),
    "memory_write": MemoryWriteTool(),
    "memory_stats": MemoryStatsTool(),
    "hash_sphere_search": HashSphereSearchTool(),
    "hash_sphere_anchor": HashSphereAnchorTool(),
    "hash_sphere_list_anchors": HashSphereListAnchorsTool(),
    "hash_sphere_hash": HashSphereHashTool(),
    "hash_sphere_resonance": HashSphereResonanceTool(),
}
