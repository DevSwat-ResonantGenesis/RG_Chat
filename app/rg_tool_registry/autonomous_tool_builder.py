"""
Autonomous Tool Builder
========================
Allows the orchestrator to BUILD, REGISTER, and USE new tools at runtime.

When the agent architect or orchestrator needs a capability that doesn't exist,
this module:
1. Uses LLM to design the tool (name, description, params, handler code)
2. Validates the generated code in a sandbox
3. Registers it into the unified ToolRegistry at runtime
4. Persists the definition to DB so it survives restarts
5. Makes it immediately available to all agents

This is what makes the platform FULLY AUTONOMOUS — it can extend itself.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import logging
import re
import textwrap
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple

from .registry import ToolDef, ToolParam, ToolCategory, ToolAccess, ParamType, ToolRegistry

logger = logging.getLogger(__name__)

# ── Safety: forbidden modules/builtins in generated tool code ──
_FORBIDDEN_IMPORTS = frozenset({
    "os", "sys", "subprocess", "shutil", "pathlib", "importlib",
    "ctypes", "signal", "socket", "multiprocessing", "threading",
    "pickle", "shelve", "marshal", "code", "codeop", "compile",
    "exec", "eval", "__import__", "open",
})

_ALLOWED_IMPORTS = frozenset({
    "json", "re", "math", "datetime", "time", "hashlib", "base64",
    "urllib.parse", "html", "csv", "io", "typing", "dataclasses",
    "collections", "itertools", "functools", "decimal", "uuid",
    "aiohttp", "httpx", "bs4", "lxml",
})

# ── LLM prompt for tool generation ──
_TOOL_BUILDER_SYSTEM_PROMPT = """You are the Resonant Genesis Autonomous Tool Builder.
Your job is to design and implement new tools that the platform doesn't have yet.

You output STRICTLY valid JSON with this schema:
{
  "name": "snake_case_tool_name",
  "description": "What this tool does (1-2 sentences, <200 chars)",
  "category": "one of: search|memory|agents|media|integrations|utilities|developer|scraping|documents|orchestrator|oauth|billing|custom",
  "params": [
    {"name": "param_name", "type": "string|integer|number|boolean|array|object", "description": "...", "required": true/false, "default": null, "enum": null}
  ],
  "handler_code": "async def handle(params: dict, context: dict) -> dict:\\n    ...\\n    return {\\"result\\": ...}",
  "requires_api_key": null,
  "access": ["registered"],
  "test_input": {"param_name": "test_value"},
  "rationale": "Why this tool is needed and how it works"
}

RULES:
1. handler_code MUST be a single async function named `handle(params, context) -> dict`
2. handler_code can use: json, re, math, datetime, time, hashlib, base64, aiohttp, httpx, bs4, typing, collections
3. handler_code CANNOT use: os, sys, subprocess, shutil, open(), exec(), eval(), pickle, socket
4. The function receives `params` (user inputs) and `context` (user_id, team_id, api_keys, headers)
5. Return a dict with at minimum a "result" key
6. For HTTP tools: use aiohttp or httpx for async requests
7. For OAuth tools: get token from context["api_keys"][service_name]
8. Keep code under 100 lines
9. Include proper error handling with try/except
10. NEVER hardcode API keys or secrets — always read from context
"""

_TOOL_BUILDER_USER_TEMPLATE = """The orchestrator needs a tool that doesn't exist yet.

NEED: {need_description}

EXISTING TOOLS (do NOT duplicate): {existing_tool_names}

USER CONTEXT:
- Connected services: {connected_services}
- Workspace tools: {workspace_tools}

Design and implement this tool. Output ONLY the JSON object, no markdown fences."""


@dataclass
class BuiltTool:
    """A dynamically built tool."""
    name: str
    description: str
    category: str
    params: List[Dict[str, Any]]
    handler_code: str
    handler_fn: Optional[Callable] = None
    requires_api_key: Optional[str] = None
    access: Set[str] = field(default_factory=lambda: {"registered"})
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    created_by: str = "autonomous_builder"
    version: int = 1
    code_hash: str = ""
    test_input: Optional[Dict] = None
    rationale: str = ""
    validated: bool = False


class AutonomousToolBuilder:
    """
    The self-expanding brain of the platform.
    
    Can design, validate, register, and persist new tools at runtime.
    """

    def __init__(self, registry: ToolRegistry):
        self._registry = registry
        self._built_tools: Dict[str, BuiltTool] = {}
        self._build_history: List[Dict] = []
        self._lock = asyncio.Lock()
        logger.info("[AutonomousToolBuilder] Initialized — platform can now self-extend")

    # ── Public API ──

    async def build_tool(
        self,
        need_description: str,
        context: Dict[str, Any],
        llm_call_fn: Callable[..., Coroutine],
    ) -> Tuple[bool, str, Optional[ToolDef]]:
        """
        Build a new tool from a natural language description.
        
        Args:
            need_description: What capability is needed
            context: User context (api_keys, user_id, etc.)
            llm_call_fn: Async function to call LLM — llm_call_fn(system, user) -> str
            
        Returns:
            (success, message, tool_def_or_none)
        """
        async with self._lock:
            start = time.time()
            logger.info(f"[ToolBuilder] Building tool for: {need_description[:100]}")

            # 1. Check if we already have something close
            existing = self._find_similar_tool(need_description)
            if existing:
                return True, f"Tool '{existing.name}' already exists and matches this need.", existing

            # 2. Ask LLM to design the tool
            existing_names = ", ".join(self._registry.get_names()[:80])
            connected = ", ".join(context.get("connected_services", []))
            workspace = ", ".join(context.get("workspace_tools", []))

            user_prompt = _TOOL_BUILDER_USER_TEMPLATE.format(
                need_description=need_description,
                existing_tool_names=existing_names or "none",
                connected_services=connected or "none",
                workspace_tools=workspace or "none",
            )

            try:
                raw = await llm_call_fn(_TOOL_BUILDER_SYSTEM_PROMPT, user_prompt)
            except Exception as e:
                return False, f"LLM call failed: {e}", None

            # 3. Parse the JSON response
            try:
                spec = self._parse_llm_response(raw)
            except Exception as e:
                return False, f"Failed to parse tool spec: {e}", None

            # 4. Validate the handler code (syntax + safety)
            ok, err = self._validate_code(spec.get("handler_code", ""))
            if not ok:
                return False, f"Code validation failed: {err}", None

            # 5. Compile the handler into a callable
            try:
                handler_fn = self._compile_handler(spec["handler_code"])
            except Exception as e:
                return False, f"Code compilation failed: {e}", None

            # 6. Build the ToolDef
            tool_def = self._spec_to_tooldef(spec, handler_fn)

            # 7. Register into the live registry
            self._registry.register(tool_def)

            # 8. Store in our built-tools map
            built = BuiltTool(
                name=spec["name"],
                description=spec["description"],
                category=spec.get("category", "custom"),
                params=spec.get("params", []),
                handler_code=spec["handler_code"],
                handler_fn=handler_fn,
                requires_api_key=spec.get("requires_api_key"),
                code_hash=hashlib.sha256(spec["handler_code"].encode()).hexdigest()[:16],
                test_input=spec.get("test_input"),
                rationale=spec.get("rationale", ""),
                validated=True,
            )
            self._built_tools[spec["name"]] = built

            elapsed = time.time() - start
            self._build_history.append({
                "name": spec["name"],
                "need": need_description[:200],
                "elapsed_s": round(elapsed, 2),
                "success": True,
                "timestamp": datetime.utcnow().isoformat(),
            })

            logger.info(f"[ToolBuilder] ✅ Built '{spec['name']}' in {elapsed:.1f}s — now in registry")
            return True, f"Tool '{spec['name']}' built and registered successfully.", tool_def

    async def build_and_use_tool(
        self,
        need_description: str,
        params: Dict[str, Any],
        context: Dict[str, Any],
        llm_call_fn: Callable[..., Coroutine],
    ) -> Tuple[bool, str, Any]:
        """
        Build a tool AND immediately execute it with the given params.
        One-shot: design → validate → register → execute.
        """
        success, msg, tool_def = await self.build_tool(need_description, context, llm_call_fn)
        if not success or not tool_def:
            return False, msg, None

        # Execute immediately
        try:
            result = await self.execute_built_tool(tool_def.name, params, context)
            return True, f"Built and executed '{tool_def.name}'", result
        except Exception as e:
            return False, f"Built '{tool_def.name}' but execution failed: {e}", None

    async def execute_built_tool(
        self,
        tool_name: str,
        params: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Any:
        """Execute a dynamically built tool."""
        built = self._built_tools.get(tool_name)
        if not built or not built.handler_fn:
            raise ValueError(f"No built tool '{tool_name}' with compiled handler")

        return await built.handler_fn(params, context)

    def get_built_tools(self) -> List[Dict[str, Any]]:
        """List all dynamically built tools."""
        return [
            {
                "name": bt.name,
                "description": bt.description,
                "category": bt.category,
                "params": bt.params,
                "created_at": bt.created_at,
                "version": bt.version,
                "code_hash": bt.code_hash,
                "validated": bt.validated,
                "rationale": bt.rationale,
            }
            for bt in self._built_tools.values()
        ]

    def get_build_history(self) -> List[Dict]:
        """Get build history."""
        return list(self._build_history)

    def has_tool(self, name: str) -> bool:
        """Check if a tool exists (built-in or dynamically built)."""
        return self._registry.get(name) is not None

    def needs_tool(self, capability: str) -> Optional[str]:
        """Check if a capability is missing and return what's needed, or None."""
        # Quick keyword match against existing tools
        cap_lower = capability.lower()
        for tool in self._registry.get_all():
            if any(w in tool.description.lower() for w in cap_lower.split()[:3]):
                return None  # Already have something
        return capability  # Need to build

    # ── Persistence (DB) ──

    async def persist_to_db(self, db_client, user_id: str) -> int:
        """Save all built tools to database for restart survival."""
        count = 0
        for name, bt in self._built_tools.items():
            try:
                await db_client.execute(
                    """INSERT INTO dynamic_tools (user_id, name, description, category, params_json,
                       handler_code, requires_api_key, access_json, created_at, version, code_hash, rationale)
                       VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
                       ON CONFLICT (user_id, name) DO UPDATE SET
                       handler_code=EXCLUDED.handler_code, version=EXCLUDED.version+1,
                       code_hash=EXCLUDED.code_hash, description=EXCLUDED.description""",
                    user_id, bt.name, bt.description, bt.category,
                    json.dumps(bt.params), bt.handler_code, bt.requires_api_key,
                    json.dumps(list(bt.access)), bt.created_at, bt.version,
                    bt.code_hash, bt.rationale,
                )
                count += 1
            except Exception as e:
                logger.error(f"[ToolBuilder] Failed to persist '{name}': {e}")
        return count

    async def load_from_db(self, db_client, user_id: str) -> int:
        """Load user's dynamic tools from database and register them."""
        try:
            rows = await db_client.fetch(
                "SELECT * FROM dynamic_tools WHERE user_id = $1", user_id
            )
        except Exception as e:
            logger.warning(f"[ToolBuilder] DB load failed (table may not exist): {e}")
            return 0

        count = 0
        for row in rows:
            try:
                ok, err = self._validate_code(row["handler_code"])
                if not ok:
                    logger.warning(f"[ToolBuilder] Skipping '{row['name']}': {err}")
                    continue

                handler_fn = self._compile_handler(row["handler_code"])
                params = json.loads(row["params_json"])
                spec = {
                    "name": row["name"],
                    "description": row["description"],
                    "category": row["category"],
                    "params": params,
                    "handler_code": row["handler_code"],
                    "requires_api_key": row.get("requires_api_key"),
                }
                tool_def = self._spec_to_tooldef(spec, handler_fn)
                self._registry.register(tool_def)

                self._built_tools[row["name"]] = BuiltTool(
                    name=row["name"],
                    description=row["description"],
                    category=row["category"],
                    params=params,
                    handler_code=row["handler_code"],
                    handler_fn=handler_fn,
                    code_hash=row.get("code_hash", ""),
                    version=row.get("version", 1),
                    validated=True,
                )
                count += 1
            except Exception as e:
                logger.error(f"[ToolBuilder] Failed to load '{row['name']}': {e}")

        if count:
            logger.info(f"[ToolBuilder] Loaded {count} dynamic tools from DB for user {user_id}")
        return count

    # ── Internal helpers ──

    def _find_similar_tool(self, description: str) -> Optional[ToolDef]:
        """Check if a tool already exists that matches this description."""
        desc_lower = description.lower()
        keywords = set(re.findall(r'\b\w{4,}\b', desc_lower))
        if not keywords:
            return None

        best_match = None
        best_score = 0
        for tool in self._registry.get_all():
            tool_words = set(re.findall(r'\b\w{4,}\b', tool.description.lower()))
            overlap = len(keywords & tool_words)
            score = overlap / max(len(keywords), 1)
            if score > best_score and score > 0.5:
                best_score = score
                best_match = tool

        return best_match

    def _parse_llm_response(self, raw: str) -> Dict[str, Any]:
        """Parse JSON from LLM response, stripping markdown fences."""
        text = raw.strip()
        # Strip ```json ... ```
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        spec = json.loads(text)

        # Validate required fields
        for field in ("name", "description", "handler_code"):
            if field not in spec:
                raise ValueError(f"Missing required field: {field}")

        # Sanitize name
        spec["name"] = re.sub(r'[^a-z0-9_]', '_', spec["name"].lower().strip())
        if not spec["name"]:
            raise ValueError("Empty tool name")

        return spec

    def _validate_code(self, code: str) -> Tuple[bool, Optional[str]]:
        """Validate handler code: syntax check + safety scan."""
        if not code or not code.strip():
            return False, "Empty handler code"

        # Syntax check
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

        # Must contain exactly one async function named 'handle'
        funcs = [n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef)]
        if not any(f.name == "handle" for f in funcs):
            return False, "Must contain 'async def handle(params, context)'"

        # Safety: check for forbidden imports and calls
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    mod = alias.name.split('.')[0]
                    if mod in _FORBIDDEN_IMPORTS:
                        return False, f"Forbidden import: {mod}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    mod = node.module.split('.')[0]
                    if mod in _FORBIDDEN_IMPORTS:
                        return False, f"Forbidden import: {mod}"
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ("exec", "eval", "__import__", "compile", "open"):
                        return False, f"Forbidden call: {node.func.id}()"
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in ("system", "popen", "exec", "eval"):
                        return False, f"Forbidden call: .{node.func.attr}()"

        return True, None

    def _compile_handler(self, code: str) -> Callable:
        """Compile handler code into a callable async function."""
        # Create a restricted namespace
        namespace: Dict[str, Any] = {
            "__builtins__": {
                # Safe builtins only
                "dict": dict, "list": list, "tuple": tuple, "set": set,
                "str": str, "int": int, "float": float, "bool": bool,
                "len": len, "range": range, "enumerate": enumerate,
                "zip": zip, "map": map, "filter": filter,
                "sorted": sorted, "reversed": reversed,
                "min": min, "max": max, "sum": sum, "abs": abs, "round": round,
                "isinstance": isinstance, "type": type, "hasattr": hasattr,
                "getattr": getattr, "setattr": setattr,
                "print": print, "repr": repr, "format": format,
                "None": None, "True": True, "False": False,
                "Exception": Exception, "ValueError": ValueError,
                "TypeError": TypeError, "KeyError": KeyError,
                "RuntimeError": RuntimeError, "AttributeError": AttributeError,
            }
        }

        # Pre-import allowed modules
        import json as _json
        import re as _re
        import math as _math
        import datetime as _datetime
        import time as _time
        import hashlib as _hashlib
        import base64 as _base64
        import urllib.parse as _urlparse
        import html as _html

        namespace.update({
            "json": _json, "re": _re, "math": _math,
            "datetime": _datetime, "time": _time,
            "hashlib": _hashlib, "base64": _base64,
            "urllib": type("urllib", (), {"parse": _urlparse}),
            "html": _html,
        })

        # Try to import optional async HTTP libs
        try:
            import aiohttp
            namespace["aiohttp"] = aiohttp
        except ImportError:
            pass
        try:
            import httpx
            namespace["httpx"] = httpx
        except ImportError:
            pass

        exec(code, namespace)

        handler = namespace.get("handle")
        if handler is None or not asyncio.iscoroutinefunction(handler):
            raise ValueError("handler_code must define 'async def handle(params, context)'")

        return handler

    def _spec_to_tooldef(self, spec: Dict[str, Any], handler_fn: Callable) -> ToolDef:
        """Convert a parsed spec dict into a ToolDef."""
        # Map category string to enum
        cat_map = {
            "search": ToolCategory.SEARCH, "memory": ToolCategory.MEMORY,
            "agents": ToolCategory.AGENTS, "media": ToolCategory.MEDIA,
            "integrations": ToolCategory.INTEGRATIONS, "utilities": ToolCategory.UTILITIES,
            "developer": ToolCategory.DEVELOPER, "scraping": ToolCategory.SCRAPING,
            "documents": ToolCategory.DOCUMENTS, "orchestrator": ToolCategory.ORCHESTRATOR,
            "oauth": ToolCategory.OAUTH, "billing": ToolCategory.BILLING,
            "custom": ToolCategory.CUSTOM,
        }
        category = cat_map.get(spec.get("category", "custom"), ToolCategory.CUSTOM)

        # Map param type strings to enum
        ptype_map = {
            "string": ParamType.STRING, "integer": ParamType.INTEGER,
            "number": ParamType.NUMBER, "boolean": ParamType.BOOLEAN,
            "array": ParamType.ARRAY, "object": ParamType.OBJECT,
        }

        params = []
        for p in spec.get("params", []):
            params.append(ToolParam(
                name=p["name"],
                type=ptype_map.get(p.get("type", "string"), ParamType.STRING),
                description=p.get("description", ""),
                required=p.get("required", False),
                default=p.get("default"),
                enum=p.get("enum"),
            ))

        # Map access strings to enum
        access_map = {
            "registered": ToolAccess.REGISTERED, "guest": ToolAccess.GUEST,
            "agent": ToolAccess.AGENT, "ide": ToolAccess.IDE,
        }
        access_set = {access_map.get(a, ToolAccess.REGISTERED) for a in spec.get("access", ["registered"])}

        return ToolDef(
            name=spec["name"],
            description=spec["description"][:200],
            category=category,
            params=params,
            handler=f"_dynamic_{spec['name']}",
            handler_fn=handler_fn,
            access=access_set,
            requires_api_key=spec.get("requires_api_key"),
            priority=30,
            max_result_chars=8000,
        )


# ── Singleton builder (initialized lazily with registry) ──
_builder_instance: Optional[AutonomousToolBuilder] = None


def get_tool_builder(registry: Optional[ToolRegistry] = None) -> AutonomousToolBuilder:
    """Get or create the singleton AutonomousToolBuilder."""
    global _builder_instance
    if _builder_instance is None:
        if registry is None:
            from .builtin_tools import build_registry
            registry = build_registry()
        _builder_instance = AutonomousToolBuilder(registry)
    return _builder_instance
