"""
Code Visualizer Granular Tools
================================

Real executors for code_visualizer_* sub-tools.
Each calls rg_ast_analysis service with specific action.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

import httpx

from .base import BaseIntegrationSkill

logger = logging.getLogger(__name__)

CODE_VIZ_URL = os.getenv("AST_ANALYSIS_SERVICE_URL") or os.getenv("CODE_VISUALIZER_URL", "http://rg_ast_analysis:8000")


class _CodeVizTool(BaseIntegrationSkill):
    """Base for code visualizer sub-tools. Delegates to parent with action hint."""
    api_key_names = []
    _action: str = "scan"

    async def execute(self, message: str, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{CODE_VIZ_URL}/analyze",
                    json={"message": message, "user_id": user_id, "action": self._action},
                    headers={"x-user-id": user_id},
                )
                resp.raise_for_status()
                data = resp.json()
                return {
                    "success": True,
                    "action": self.skill_id,
                    "summary": data.get("summary", data.get("result", str(data)[:2000])),
                    "data": data,
                }
        except Exception as e:
            return {"success": False, "action": self.skill_id, "error": str(e)[:300]}


class CodeVisualizerScanTool(_CodeVizTool):
    skill_id = "code_visualizer_scan"
    skill_name = "Code Scan"
    _action = "scan"
    intent_keywords = ["scan code", "analyze code", "code scan"]

class CodeVisualizerFunctionsTool(_CodeVizTool):
    skill_id = "code_visualizer_functions"
    skill_name = "Code Functions"
    _action = "functions"
    intent_keywords = ["list functions", "show functions", "code functions"]

class CodeVisualizerTraceTool(_CodeVizTool):
    skill_id = "code_visualizer_trace"
    skill_name = "Code Trace"
    _action = "trace"
    intent_keywords = ["trace code", "trace function", "call trace"]

class CodeVisualizerGovernanceTool(_CodeVizTool):
    skill_id = "code_visualizer_governance"
    skill_name = "Code Governance"
    _action = "governance"
    intent_keywords = ["code governance", "code quality", "governance scan"]

class CodeVisualizerGraphTool(_CodeVizTool):
    skill_id = "code_visualizer_graph"
    skill_name = "Code Graph"
    _action = "graph"
    intent_keywords = ["code graph", "dependency graph", "call graph"]

class CodeVisualizerPipelineTool(_CodeVizTool):
    skill_id = "code_visualizer_pipeline"
    skill_name = "Code Pipeline"
    _action = "pipeline"
    intent_keywords = ["code pipeline", "data pipeline", "pipeline analysis"]

class CodeVisualizerFilterTool(_CodeVizTool):
    skill_id = "code_visualizer_filter"
    skill_name = "Code Filter"
    _action = "filter"
    intent_keywords = ["filter code", "filter functions", "code filter"]

class CodeVisualizerByTypeTool(_CodeVizTool):
    skill_id = "code_visualizer_by_type"
    skill_name = "Code By Type"
    _action = "by_type"
    intent_keywords = ["code by type", "group by type", "code types"]


CODE_VISUALIZER_TOOLS = {
    "code_visualizer_scan": CodeVisualizerScanTool(),
    "code_visualizer_functions": CodeVisualizerFunctionsTool(),
    "code_visualizer_trace": CodeVisualizerTraceTool(),
    "code_visualizer_governance": CodeVisualizerGovernanceTool(),
    "code_visualizer_graph": CodeVisualizerGraphTool(),
    "code_visualizer_pipeline": CodeVisualizerPipelineTool(),
    "code_visualizer_filter": CodeVisualizerFilterTool(),
    "code_visualizer_by_type": CodeVisualizerByTypeTool(),
}
