"""
State Physics Granular Tools
==============================

Real executors for sp_* sub-tools.
Each calls rg_users_invarients_sim service.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

import httpx

from .base import BaseIntegrationSkill

logger = logging.getLogger(__name__)

SP_URL = os.getenv("STATE_PHYSICS_URL", "http://rg_users_invarients_sim:8091")


class _SPTool(BaseIntegrationSkill):
    """Base for state physics sub-tools."""
    api_key_names = []
    _method: str = "GET"
    _path: str = "/"

    async def execute(self, message: str, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                if self._method == "GET":
                    resp = await client.get(f"{SP_URL}{self._path}", params={"user_id": user_id})
                else:
                    resp = await client.post(f"{SP_URL}{self._path}", json={"message": message, "user_id": user_id})
                resp.raise_for_status()
                data = resp.json()
                return {"success": True, "action": self.skill_id, "summary": str(data)[:3000], "data": data}
        except Exception as e:
            return {"success": False, "action": self.skill_id, "error": str(e)[:300]}


class SPStateTool(_SPTool):
    skill_id = "sp_state"; skill_name = "SP State"; _path = "/state"; intent_keywords = ["physics state"]

class SPResetTool(_SPTool):
    skill_id = "sp_reset"; skill_name = "SP Reset"; _method = "POST"; _path = "/reset"; intent_keywords = ["reset physics"]

class SPNodesTool(_SPTool):
    skill_id = "sp_nodes"; skill_name = "SP Nodes"; _path = "/nodes"; intent_keywords = ["physics nodes"]

class SPMetricsTool(_SPTool):
    skill_id = "sp_metrics"; skill_name = "SP Metrics"; _path = "/metrics"; intent_keywords = ["physics metrics"]

class SPIdentityTool(_SPTool):
    skill_id = "sp_identity"; skill_name = "SP Identity"; _path = "/identity"; intent_keywords = ["physics identity"]

class SPSimulateTool(_SPTool):
    skill_id = "sp_simulate"; skill_name = "SP Simulate"; _method = "POST"; _path = "/simulate"; intent_keywords = ["simulate physics"]

class SPGalaxyTool(_SPTool):
    skill_id = "sp_galaxy"; skill_name = "SP Galaxy"; _path = "/galaxy"; intent_keywords = ["galaxy view"]

class SPDemoTool(_SPTool):
    skill_id = "sp_demo"; skill_name = "SP Demo"; _method = "POST"; _path = "/demo"; intent_keywords = ["physics demo"]

class SPAsymmetryTool(_SPTool):
    skill_id = "sp_asymmetry"; skill_name = "SP Asymmetry"; _path = "/asymmetry"; intent_keywords = ["asymmetry"]

class SPPhysicsConfigTool(_SPTool):
    skill_id = "sp_physics_config"; skill_name = "SP Physics Config"; _path = "/physics/config"; intent_keywords = ["physics config"]

class SPEntropyConfigTool(_SPTool):
    skill_id = "sp_entropy_config"; skill_name = "SP Entropy Config"; _path = "/entropy/config"; intent_keywords = ["entropy config"]

class SPEntropyToggleTool(_SPTool):
    skill_id = "sp_entropy_toggle"; skill_name = "SP Entropy Toggle"; _method = "POST"; _path = "/entropy/toggle"; intent_keywords = ["toggle entropy"]

class SPEntropyPerturbationTool(_SPTool):
    skill_id = "sp_entropy_perturbation"; skill_name = "SP Perturbation"; _method = "POST"; _path = "/entropy/perturbation"; intent_keywords = ["entropy perturbation"]

class SPAgentSpawnTool(_SPTool):
    skill_id = "sp_agent_spawn"; skill_name = "SP Agent Spawn"; _method = "POST"; _path = "/agents/spawn"; intent_keywords = ["spawn agent physics"]

class SPAgentStepTool(_SPTool):
    skill_id = "sp_agent_step"; skill_name = "SP Agent Step"; _method = "POST"; _path = "/agents/step"; intent_keywords = ["agent step physics"]

class SPAgentKillTool(_SPTool):
    skill_id = "sp_agent_kill"; skill_name = "SP Agent Kill"; _method = "POST"; _path = "/agents/kill"; intent_keywords = ["kill agent physics"]

class SPAgentsSpawnTool(_SPTool):
    skill_id = "sp_agents_spawn"; skill_name = "SP Agents Spawn Batch"; _method = "POST"; _path = "/agents/spawn-batch"; intent_keywords = ["spawn agents batch"]

class SPAgentsKillAllTool(_SPTool):
    skill_id = "sp_agents_kill_all"; skill_name = "SP Kill All Agents"; _method = "POST"; _path = "/agents/kill-all"; intent_keywords = ["kill all agents physics"]

class SPExperimentTool(_SPTool):
    skill_id = "sp_experiment"; skill_name = "SP Experiment"; _method = "POST"; _path = "/experiment"; intent_keywords = ["physics experiment"]

class SPMemoryCostTool(_SPTool):
    skill_id = "sp_memory_cost"; skill_name = "SP Memory Cost"; _path = "/memory/cost"; intent_keywords = ["memory cost physics"]

class SPMetricsRecordTool(_SPTool):
    skill_id = "sp_metrics_record"; skill_name = "SP Metrics Record"; _method = "POST"; _path = "/metrics/record"; intent_keywords = ["record metrics"]


STATE_PHYSICS_TOOLS = {
    "sp_state": SPStateTool(), "sp_reset": SPResetTool(), "sp_nodes": SPNodesTool(),
    "sp_metrics": SPMetricsTool(), "sp_identity": SPIdentityTool(), "sp_simulate": SPSimulateTool(),
    "sp_galaxy": SPGalaxyTool(), "sp_demo": SPDemoTool(), "sp_asymmetry": SPAsymmetryTool(),
    "sp_physics_config": SPPhysicsConfigTool(), "sp_entropy_config": SPEntropyConfigTool(),
    "sp_entropy_toggle": SPEntropyToggleTool(), "sp_entropy_perturbation": SPEntropyPerturbationTool(),
    "sp_agent_spawn": SPAgentSpawnTool(), "sp_agent_step": SPAgentStepTool(),
    "sp_agent_kill": SPAgentKillTool(), "sp_agents_spawn": SPAgentsSpawnTool(),
    "sp_agents_kill_all": SPAgentsKillAllTool(), "sp_experiment": SPExperimentTool(),
    "sp_memory_cost": SPMemoryCostTool(), "sp_metrics_record": SPMetricsRecordTool(),
}
