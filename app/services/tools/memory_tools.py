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
                confidence = float(data.get("confidence", 0.0))
                answer_from_memory = bool(data.get("answer_from_memory", False))
                evidence_hash = data.get("evidence_hash")

                # Zero-LLM recall: when the confidence gate passed, the memory brain is
                # certain enough to answer directly — lead with the grounded answer.
                if answer_from_memory:
                    top = memories[0]
                    answer = (top.get("content") or "").strip()
                    summary = (
                        f"**Answer (from memory, confidence {confidence:.0%}):**\n\n{answer}\n"
                    )
                    if evidence_hash:
                        summary += f"\n_Evidence: `{evidence_hash}`_"
                    if len(memories) > 1:
                        summary += "\n\n**Supporting memories:**\n"
                        for i, m in enumerate(memories[1:5], 1):
                            summary += f"{i}. {(m.get('content') or '')[:160]}\n"
                    return {
                        "success": True,
                        "action": "hash_sphere_search",
                        "answer_from_memory": True,
                        "answer": answer,
                        "confidence": confidence,
                        "evidence_hash": evidence_hash,
                        "summary": summary,
                        "count": len(memories),
                    }

                # Below the gate — return ranked candidates for the LLM to reason over.
                summary = f"**Hash Sphere — {len(memories)} results (confidence {confidence:.0%}):**\n\n"
                for i, m in enumerate(memories[:10], 1):
                    content = (m.get("content") or "")[:200]
                    score = m.get("hybrid_score", 0)
                    summary += f"{i}. {content}... (score: {score:.2f})\n\n"
                return {
                    "success": True,
                    "action": "hash_sphere_search",
                    "answer_from_memory": False,
                    "confidence": confidence,
                    "summary": summary,
                    "count": len(memories),
                }
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


class MemoryFactsTool(BaseIntegrationSkill):
    skill_id = "memory_facts"
    skill_name = "Memory Facts"
    api_key_names = []
    intent_keywords = ["what do you know about me", "my facts", "my name", "my preferences", "my details"]

    async def execute(self, message: str, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    f"{MEMORY_SERVICE_URL}/memory/facts",
                    params={"user_id": user_id, "limit": 100},
                )
                resp.raise_for_status()
                facts = resp.json().get("facts", [])
                if not facts:
                    return {"success": True, "action": "memory_facts", "summary": "No stored facts yet.", "count": 0}
                lines = [f"- {f.get('fact')}" for f in facts[:40] if f.get("fact")]
                summary = "**What I know about you:**\n\n" + "\n".join(lines)
                return {"success": True, "action": "memory_facts", "summary": summary, "facts": facts, "count": len(facts)}
        except Exception as e:
            return {"success": False, "action": "memory_facts", "error": str(e)[:300]}


class MemoryRagAskTool(BaseIntegrationSkill):
    skill_id = "memory_rag_ask"
    skill_name = "Memory RAG Ask"
    api_key_names = []
    intent_keywords = ["what have i told you", "what do you know about", "based on my memory", "recall about"]

    async def execute(self, message: str, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{MEMORY_SERVICE_URL}/rag/ask",
                    json={"question": message, "user_id": user_id},
                )
                resp.raise_for_status()
                data = resp.json()
                answer = (data.get("answer") or "").strip()
                if not answer:
                    return {"success": True, "action": "memory_rag_ask", "summary": "No relevant memory found for that question.", "count": 0}
                sources = data.get("sources") or []
                summary = f"**Answer (from your memory):**\n\n{answer}"
                return {"success": True, "action": "memory_rag_ask", "summary": summary, "answer": answer, "sources": sources}
        except Exception as e:
            return {"success": False, "action": "memory_rag_ask", "error": str(e)[:300]}


class MemoryUniverseTool(BaseIntegrationSkill):
    skill_id = "memory_universe"
    skill_name = "Memory Universe"
    api_key_names = []
    intent_keywords = ["show my memory universe", "memory layers", "short term memory", "long term memory", "memory clusters"]

    async def execute(self, message: str, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    f"{MEMORY_SERVICE_URL}/rag/universe",
                    params={"user_id": user_id},
                )
                resp.raise_for_status()
                data = resp.json()
                records = data.get("memories", data if isinstance(data, list) else [])
                if not records:
                    return {"success": True, "action": "memory_universe", "summary": "No memory universe data yet.", "count": 0}
                by_layer: Dict[str, int] = {}
                for r in records:
                    layer = (r.get("layer") or "active") if isinstance(r, dict) else "active"
                    by_layer[layer] = by_layer.get(layer, 0) + 1
                layer_lines = "\n".join(f"- {layer}: {count}" for layer, count in by_layer.items())
                summary = f"**Memory universe ({len(records)} entries):**\n\n{layer_lines}"
                return {"success": True, "action": "memory_universe", "summary": summary, "count": len(records), "by_layer": by_layer}
        except Exception as e:
            return {"success": False, "action": "memory_universe", "error": str(e)[:300]}


# ── Registry ──

MEMORY_TOOLS = {
    "memory_read": MemoryReadTool(),
    "memory_write": MemoryWriteTool(),
    "memory_stats": MemoryStatsTool(),
    "memory_facts": MemoryFactsTool(),
    "hash_sphere_search": HashSphereSearchTool(),
    "hash_sphere_anchor": HashSphereAnchorTool(),
    "hash_sphere_list_anchors": HashSphereListAnchorsTool(),
    "hash_sphere_hash": HashSphereHashTool(),
    "hash_sphere_resonance": HashSphereResonanceTool(),
    "memory_rag_ask": MemoryRagAskTool(),
    "memory_universe": MemoryUniverseTool(),
}
