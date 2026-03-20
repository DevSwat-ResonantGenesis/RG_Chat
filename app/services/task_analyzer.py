"""
Task Analyzer
==============

Phase 1 of Agent Autonomy Enhancement.

Analyzes incoming tasks to determine complexity, required skills, and priority.
Enables intelligent agent selection based on task characteristics.

Author: Resonant Chat Systems Team
Date: December 26, 2025
"""
from __future__ import annotations

import re
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4


@dataclass
class TaskProfile:
    """Profile of an analyzed task."""
    message: str
    complexity_score: float  # 0.0 - 1.0
    required_skills: List[str]
    estimated_time_ms: float
    priority: TaskPriority
    technical_terms_count: int
    code_blocks_count: int
    question_marks_count: int
    message_length: int


class TaskAnalyzer:
    """Analyzes tasks to extract characteristics for intelligent agent selection."""
    
    # Technical terms that indicate complexity
    TECHNICAL_TERMS = {
        # Programming
        "algorithm", "function", "class", "method", "variable", "array", "object",
        "async", "await", "promise", "callback", "closure", "prototype", "inheritance",
        "polymorphism", "encapsulation", "abstraction", "interface", "abstract",
        
        # Architecture
        "microservice", "monolith", "scalability", "load balancing", "caching",
        "database", "schema", "migration", "orm", "sql", "nosql", "redis",
        "architecture", "design pattern", "mvc", "mvvm", "repository",
        
        # DevOps
        "docker", "kubernetes", "k8s", "ci/cd", "pipeline", "deployment",
        "container", "orchestration", "terraform", "ansible", "jenkins",
        
        # Web
        "api", "rest", "graphql", "endpoint", "http", "https", "websocket",
        "cors", "csrf", "xss", "authentication", "authorization", "jwt",
        "oauth", "session", "cookie", "token",
        
        # Frontend
        "react", "vue", "angular", "component", "props", "state", "redux",
        "hook", "lifecycle", "virtual dom", "jsx", "tsx", "webpack",
        
        # Backend
        "server", "route", "middleware", "controller", "service", "repository",
        "model", "validation", "serialization", "deserialization",
        
        # Testing
        "unit test", "integration test", "e2e", "mock", "stub", "spy",
        "coverage", "assertion", "test case", "test suite",
        
        # Security
        "vulnerability", "exploit", "injection", "sanitization", "encryption",
        "hashing", "ssl", "tls", "certificate", "firewall",
        
        # Performance
        "optimization", "bottleneck", "profiling", "benchmark", "latency",
        "throughput", "memory leak", "garbage collection",
    }
    
    # Skill keywords mapping
    SKILL_KEYWORDS = {
        "code_analysis": ["analyze code", "review code", "code review", "examine code"],
        "debugging": ["debug", "fix bug", "error", "issue", "problem", "broken", "not working"],
        "code_generation": ["write code", "generate code", "create function", "implement"],
        "refactoring": ["refactor", "restructure", "clean up", "reorganize"],
        "testing": ["test", "unit test", "integration test", "coverage"],
        "documentation": ["document", "readme", "docs", "documentation"],
        "security": ["security", "vulnerability", "exploit", "secure"],
        "performance": ["optimize", "performance", "speed up", "faster", "slow"],
        "database": ["database", "sql", "query", "schema", "table"],
        "api_design": ["api", "endpoint", "rest", "graphql"],
        "deployment": ["deploy", "deployment", "ci/cd", "docker", "kubernetes"],
        "ui_design": ["ui", "interface", "design", "layout", "styling"],
        "architecture": ["architecture", "design pattern", "system design", "scalable"],
        "teaching": ["explain", "teach", "tutorial", "learn", "beginner"],
        "research": ["research", "investigate", "find", "compare"],
        "planning": ["plan", "strategy", "roadmap", "steps"],
        "migration": ["migrate", "upgrade", "convert", "port"],
        "accessibility": ["accessibility", "a11y", "wcag", "aria"],
        "internationalization": ["i18n", "localization", "translate"],
        "version_control": ["git", "merge", "branch", "commit"],
        "regex": ["regex", "regular expression", "pattern"],
        "math": ["calculate", "math", "equation", "formula"],
    }
    
    def analyze_task(self, message: str) -> TaskProfile:
        """Analyze a task message and return its profile."""
        message_lower = message.lower()
        
        # Calculate complexity
        complexity_score = self._calculate_complexity(message, message_lower)
        
        # Extract required skills
        required_skills = self._extract_skills(message_lower)
        
        # Estimate time
        estimated_time_ms = self._estimate_time(message, complexity_score)
        
        # Determine priority
        priority = self._determine_priority(message_lower, complexity_score)
        
        # Count technical indicators
        technical_terms_count = self._count_technical_terms(message_lower)
        code_blocks_count = message.count("```")
        question_marks_count = message.count("?")
        
        profile = TaskProfile(
            message=message,
            complexity_score=complexity_score,
            required_skills=required_skills,
            estimated_time_ms=estimated_time_ms,
            priority=priority,
            technical_terms_count=technical_terms_count,
            code_blocks_count=code_blocks_count,
            question_marks_count=question_marks_count,
            message_length=len(message),
        )
        
        logger.info(
            f"Task analyzed: complexity={complexity_score:.2f}, "
            f"skills={required_skills[:3]}, priority={priority.name}"
        )
        
        return profile
    
    def _calculate_complexity(self, message: str, message_lower: str) -> float:
        """Calculate task complexity score (0.0 - 1.0)."""
        factors = {}
        
        # 1. Message length (longer = more complex)
        factors["length"] = min(len(message) / 1000, 1.0) * 0.2
        
        # 2. Technical terms (more = more complex)
        technical_count = self._count_technical_terms(message_lower)
        factors["technical_terms"] = min(technical_count / 10, 1.0) * 0.3
        
        # 3. Code blocks (presence indicates technical task)
        code_blocks = message.count("```")
        factors["code_blocks"] = min(code_blocks / 3, 1.0) * 0.2
        
        # 4. Question marks (multiple questions = more complex)
        questions = message.count("?")
        factors["questions"] = min(questions / 5, 1.0) * 0.1
        
        # 5. Multi-part requests (and, also, additionally)
        multi_part_keywords = ["and", "also", "additionally", "furthermore", "moreover"]
        multi_part_count = sum(1 for kw in multi_part_keywords if kw in message_lower)
        factors["multi_part"] = min(multi_part_count / 3, 1.0) * 0.1
        
        # 6. Complexity keywords
        complexity_keywords = [
            "complex", "complicated", "advanced", "sophisticated", "intricate",
            "comprehensive", "detailed", "thorough", "in-depth"
        ]
        has_complexity_keyword = any(kw in message_lower for kw in complexity_keywords)
        factors["complexity_keywords"] = 0.1 if has_complexity_keyword else 0.0
        
        # Calculate total
        total_score = sum(factors.values())
        
        logger.debug(f"Complexity factors: {factors}, total: {total_score:.2f}")
        
        return min(total_score, 1.0)
    
    def _count_technical_terms(self, message_lower: str) -> int:
        """Count technical terms in message."""
        count = 0
        for term in self.TECHNICAL_TERMS:
            if term in message_lower:
                count += 1
        return count
    
    def _extract_skills(self, message_lower: str) -> List[str]:
        """Extract required skills from message."""
        skills = []
        
        for skill, keywords in self.SKILL_KEYWORDS.items():
            for keyword in keywords:
                if keyword in message_lower:
                    if skill not in skills:
                        skills.append(skill)
                    break
        
        # If no specific skills detected, add general skills
        if not skills:
            skills.append("general_reasoning")
        
        return skills
    
    def _estimate_time(self, message: str, complexity_score: float) -> float:
        """Estimate task completion time in milliseconds."""
        # Base time
        base_time = 1000  # 1 second
        
        # Adjust by complexity
        complexity_multiplier = 1 + (complexity_score * 2)  # 1x to 3x
        
        # Adjust by message length
        length_multiplier = 1 + (len(message) / 500)  # Longer messages take more time
        
        # Adjust by code blocks (code analysis takes longer)
        code_blocks = message.count("```")
        code_multiplier = 1 + (code_blocks * 0.5)
        
        estimated_time = base_time * complexity_multiplier * length_multiplier * code_multiplier
        
        # Cap at 10 seconds
        return min(estimated_time, 10000)
    
    def _determine_priority(self, message_lower: str, complexity_score: float) -> TaskPriority:
        """Determine task priority."""
        # Urgent keywords
        urgent_keywords = ["urgent", "asap", "immediately", "critical", "emergency", "now"]
        if any(kw in message_lower for kw in urgent_keywords):
            return TaskPriority.URGENT
        
        # High priority keywords
        high_keywords = ["important", "priority", "soon", "quickly", "fast"]
        if any(kw in message_lower for kw in high_keywords):
            return TaskPriority.HIGH
        
        # Complex tasks are higher priority
        if complexity_score > 0.7:
            return TaskPriority.HIGH
        elif complexity_score > 0.4:
            return TaskPriority.MEDIUM
        else:
            return TaskPriority.LOW
    
    def get_task_summary(self, profile: TaskProfile) -> str:
        """Get a human-readable summary of the task profile."""
        return (
            f"Task: {profile.message[:50]}...\n"
            f"Complexity: {profile.complexity_score:.2%}\n"
            f"Skills: {', '.join(profile.required_skills[:5])}\n"
            f"Priority: {profile.priority.name}\n"
            f"Est. Time: {profile.estimated_time_ms:.0f}ms\n"
            f"Technical Terms: {profile.technical_terms_count}\n"
            f"Code Blocks: {profile.code_blocks_count}"
        )


# Global instance
task_analyzer = TaskAnalyzer()
