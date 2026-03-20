"""
Dynamic Team Composition System (DTCS)
=======================================

Phase 4: Automatically compose optimal teams based on task analysis.

Features:
- Analyze task complexity and requirements
- Select optimal agent combination
- Dynamic workflow selection (sequential vs parallel)
- Team size optimization
"""
from __future__ import annotations

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import re

logger = logging.getLogger(__name__)


@dataclass
class TaskAnalysis:
    """Analysis of a task for team composition."""
    complexity: str  # 'simple', 'moderate', 'complex'
    domains: List[str]  # Detected domains (code, security, architecture, etc.)
    requires_review: bool
    requires_testing: bool
    requires_planning: bool
    estimated_agents: int
    recommended_workflow: str  # 'sequential' or 'parallel_merge'
    confidence: float


@dataclass
class DynamicTeam:
    """A dynamically composed team."""
    name: str
    agents: List[str]
    workflow: str
    rationale: str
    task_analysis: TaskAnalysis


class DynamicTeamComposer:
    """
    Dynamically composes optimal teams based on task analysis.
    """
    
    def __init__(self):
        # Domain detection patterns
        self.domain_patterns = {
            "code": [
                "write code", "implement", "create function", "generate", "build",
                "script", "program", "class", "method", "algorithm"
            ],
            "debug": [
                "fix", "debug", "error", "bug", "issue", "broken", "not working",
                "fails", "crash", "exception", "traceback"
            ],
            "security": [
                "security", "vulnerability", "hack", "exploit", "injection",
                "authentication", "authorization", "encrypt", "password", "token"
            ],
            "architecture": [
                "architecture", "design", "structure", "scalable", "microservice",
                "pattern", "system design", "infrastructure"
            ],
            "test": [
                "test", "unit test", "coverage", "mock", "assertion", "spec",
                "integration test", "e2e", "qa"
            ],
            "review": [
                "review", "critique", "feedback", "improve", "refactor",
                "code quality", "best practice", "optimize"
            ],
            "documentation": [
                "document", "readme", "api doc", "jsdoc", "docstring",
                "comments", "explain code"
            ],
            "database": [
                "database", "sql", "query", "schema", "migration", "model",
                "orm", "postgres", "mysql", "mongodb"
            ],
            "api": [
                "api", "endpoint", "rest", "graphql", "route", "http",
                "request", "response", "webhook"
            ],
            "devops": [
                "deploy", "ci/cd", "docker", "kubernetes", "pipeline",
                "terraform", "aws", "cloud", "infrastructure"
            ],
            "planning": [
                "plan", "strategy", "roadmap", "steps", "approach",
                "how to", "project", "milestone"
            ],
            "research": [
                "research", "find", "compare", "difference", "alternatives",
                "options", "investigate", "explore"
            ],
        }
        
        # Complexity indicators
        self.complexity_indicators = {
            "complex": [
                "full", "complete", "comprehensive", "entire", "all",
                "from scratch", "production", "enterprise", "scalable"
            ],
            "moderate": [
                "add", "modify", "update", "change", "improve",
                "extend", "enhance"
            ],
            "simple": [
                "quick", "simple", "basic", "small", "minor",
                "just", "only", "single"
            ]
        }
    
    def analyze_task(self, task: str) -> TaskAnalysis:
        """Analyze a task to determine optimal team composition."""
        task_lower = task.lower()
        
        # Detect domains
        domains = []
        for domain, patterns in self.domain_patterns.items():
            if any(pattern in task_lower for pattern in patterns):
                domains.append(domain)
        
        # If no domains detected, default to reasoning
        if not domains:
            domains = ["reasoning"]
        
        # Detect complexity
        complexity = "moderate"  # default
        for level, indicators in self.complexity_indicators.items():
            if any(indicator in task_lower for indicator in indicators):
                complexity = level
                break
        
        # Determine if review/testing/planning is needed
        requires_review = (
            complexity in ["moderate", "complex"] or
            "review" in domains or
            any(word in task_lower for word in ["quality", "best practice", "improve"])
        )
        
        requires_testing = (
            complexity == "complex" or
            "test" in domains or
            "code" in domains or
            any(word in task_lower for word in ["test", "verify", "validate"])
        )
        
        requires_planning = (
            complexity == "complex" or
            "planning" in domains or
            any(word in task_lower for word in ["plan", "strategy", "approach", "steps"])
        )
        
        # Estimate number of agents needed
        if complexity == "simple":
            estimated_agents = 1
        elif complexity == "moderate":
            estimated_agents = 2 if requires_review else 1
        else:  # complex
            estimated_agents = min(len(domains) + (1 if requires_review else 0), 4)
        
        # Determine workflow
        if len(domains) > 2 and complexity == "complex":
            recommended_workflow = "parallel_merge"
        else:
            recommended_workflow = "sequential"
        
        # Calculate confidence
        confidence = min(1.0, len(domains) * 0.3 + 0.4)
        
        return TaskAnalysis(
            complexity=complexity,
            domains=domains,
            requires_review=requires_review,
            requires_testing=requires_testing,
            requires_planning=requires_planning,
            estimated_agents=estimated_agents,
            recommended_workflow=recommended_workflow,
            confidence=confidence,
        )
    
    def compose_team(self, task: str) -> Optional[DynamicTeam]:
        """Compose an optimal team for a task."""
        analysis = self.analyze_task(task)
        
        # Simple tasks don't need teams
        if analysis.complexity == "simple" and len(analysis.domains) == 1:
            return None
        
        # Build agent list
        agents = []
        
        # Add primary domain agents
        for domain in analysis.domains[:2]:  # Max 2 primary domains
            agents.append(domain)
        
        # Add review if needed
        if analysis.requires_review and "review" not in agents:
            agents.append("review")
        
        # Add testing if needed
        if analysis.requires_testing and "test" not in agents:
            agents.append("test")
        
        # Add planning if needed (at the start)
        if analysis.requires_planning and "planning" not in agents:
            agents.insert(0, "planning")
        
        # Limit team size
        agents = agents[:4]
        
        # Generate team name
        if len(agents) == 2:
            team_name = f"{agents[0].title()}-{agents[1].title()} Duo"
        elif len(agents) == 3:
            team_name = f"{agents[0].title()} Trio"
        else:
            team_name = f"Dynamic {analysis.complexity.title()} Team"
        
        # Generate rationale
        rationale = (
            f"Task complexity: {analysis.complexity}. "
            f"Detected domains: {', '.join(analysis.domains)}. "
            f"Using {analysis.recommended_workflow} workflow for optimal results."
        )
        
        return DynamicTeam(
            name=team_name,
            agents=agents,
            workflow=analysis.recommended_workflow,
            rationale=rationale,
            task_analysis=analysis,
        )
    
    def should_use_dynamic_team(self, task: str) -> Tuple[bool, Optional[DynamicTeam]]:
        """Determine if a dynamic team should be used for a task.

        DISABLED: Dynamic team composition fires too broadly — domain patterns
        match common words like "build", "how to", "fix", "test", "review",
        causing multi-agent overhead for simple questions. A single focused LLM
        call with clean context outperforms a committee of agents on nearly all tasks.
        User can still explicitly select teams via the UI team picker.
        """
        return False, None
    
    def get_agent_for_domain(self, domain: str) -> str:
        """Map domain to agent type."""
        domain_to_agent = {
            "code": "code",
            "debug": "debug",
            "security": "security",
            "architecture": "architecture",
            "test": "test",
            "review": "review",
            "documentation": "documentation",
            "database": "database",
            "api": "api",
            "devops": "devops",
            "planning": "planning",
            "research": "research",
            "reasoning": "reasoning",
        }
        return domain_to_agent.get(domain, "reasoning")
    
    def explain_composition(self, team: DynamicTeam) -> str:
        """Generate a human-readable explanation of team composition."""
        explanation = [
            f"**Team: {team.name}**",
            f"",
            f"**Agents:** {' → '.join(team.agents) if team.workflow == 'sequential' else ' + '.join(team.agents)}",
            f"",
            f"**Workflow:** {team.workflow.replace('_', ' ').title()}",
            f"",
            f"**Rationale:** {team.rationale}",
        ]
        
        if team.task_analysis.requires_review:
            explanation.append("- ✓ Includes code review for quality assurance")
        if team.task_analysis.requires_testing:
            explanation.append("- ✓ Includes testing for reliability")
        if team.task_analysis.requires_planning:
            explanation.append("- ✓ Includes planning for structured approach")
        
        return "\n".join(explanation)


# Global instance
dynamic_team_composer = DynamicTeamComposer()
