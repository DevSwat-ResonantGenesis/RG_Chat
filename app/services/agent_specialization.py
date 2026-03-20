"""
Agent Specialization Training System (ASTS)
============================================

Phase 4: Enables agents to specialize based on user patterns and codebase context.

Features:
- Learn from user's coding style
- Adapt to project-specific patterns
- Custom terminology learning
- Domain-specific knowledge accumulation
"""
from __future__ import annotations

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict
import re

logger = logging.getLogger(__name__)


@dataclass
class SpecializationProfile:
    """Specialization profile for an agent-user combination."""
    agent_type: str
    user_id: str
    
    # Learned patterns
    coding_style: Dict[str, Any] = field(default_factory=dict)
    preferred_frameworks: List[str] = field(default_factory=list)
    common_patterns: List[str] = field(default_factory=list)
    terminology: Dict[str, str] = field(default_factory=dict)  # custom term -> meaning
    
    # Project context
    project_languages: List[str] = field(default_factory=list)
    project_structure: Dict[str, Any] = field(default_factory=dict)
    
    # Learning stats
    interactions_count: int = 0
    last_updated: str = ""
    confidence_score: float = 0.0


class AgentSpecializationEngine:
    """
    Enables agents to learn and specialize based on user interactions.
    """
    
    def __init__(self):
        self.profiles: Dict[str, Dict[str, SpecializationProfile]] = {}  # user_id -> agent_type -> profile
        
        # Pattern detectors
        self.framework_patterns = {
            "react": ["useState", "useEffect", "jsx", "tsx", "component"],
            "vue": ["v-if", "v-for", "ref(", "computed(", ".vue"],
            "angular": ["@Component", "@Injectable", "ngOnInit", ".ts"],
            "express": ["app.get", "app.post", "req, res", "middleware"],
            "fastapi": ["@app.get", "@app.post", "async def", "Depends"],
            "django": ["models.Model", "views.py", "urls.py", "admin.site"],
            "nextjs": ["getServerSideProps", "getStaticProps", "pages/", "app/"],
            "tailwind": ["className=", "flex", "grid", "bg-", "text-"],
        }
        
        self.style_patterns = {
            "functional": ["const", "=>", "map(", "filter(", "reduce("],
            "class_based": ["class ", "constructor(", "this.", "extends"],
            "async_heavy": ["async ", "await ", "Promise", ".then("],
            "typed": ["interface ", "type ", ": string", ": number", "TypeScript"],
        }
    
    def get_profile(self, user_id: str, agent_type: str) -> SpecializationProfile:
        """Get or create a specialization profile."""
        if user_id not in self.profiles:
            self.profiles[user_id] = {}
        
        if agent_type not in self.profiles[user_id]:
            self.profiles[user_id][agent_type] = SpecializationProfile(
                agent_type=agent_type,
                user_id=user_id,
                last_updated=datetime.now().isoformat(),
            )
        
        return self.profiles[user_id][agent_type]
    
    def learn_from_interaction(
        self,
        user_id: str,
        agent_type: str,
        task: str,
        response: str,
        context: List[Dict[str, Any]],
        feedback_positive: bool = True,
    ):
        """Learn from a user interaction to improve specialization."""
        profile = self.get_profile(user_id, agent_type)
        
        # Only learn from positive interactions
        if not feedback_positive:
            return
        
        # Detect frameworks from task and context
        all_text = task + " " + response + " " + str(context)
        detected_frameworks = self._detect_frameworks(all_text)
        for framework in detected_frameworks:
            if framework not in profile.preferred_frameworks:
                profile.preferred_frameworks.append(framework)
        
        # Detect coding style
        detected_styles = self._detect_coding_style(all_text)
        for style, count in detected_styles.items():
            profile.coding_style[style] = profile.coding_style.get(style, 0) + count
        
        # Detect programming languages
        detected_languages = self._detect_languages(all_text)
        for lang in detected_languages:
            if lang not in profile.project_languages:
                profile.project_languages.append(lang)
        
        # Extract common patterns (code snippets that appear frequently)
        patterns = self._extract_patterns(task)
        for pattern in patterns:
            if pattern not in profile.common_patterns and len(profile.common_patterns) < 50:
                profile.common_patterns.append(pattern)
        
        # Update stats
        profile.interactions_count += 1
        profile.last_updated = datetime.now().isoformat()
        profile.confidence_score = min(1.0, profile.interactions_count / 100)
        
        logger.debug(f"🎓 Learned from interaction for {agent_type} (user: {user_id[:8]}...)")
    
    def _detect_frameworks(self, text: str) -> List[str]:
        """Detect frameworks mentioned in text."""
        detected = []
        text_lower = text.lower()
        
        for framework, patterns in self.framework_patterns.items():
            if any(pattern.lower() in text_lower for pattern in patterns):
                detected.append(framework)
        
        return detected
    
    def _detect_coding_style(self, text: str) -> Dict[str, int]:
        """Detect coding style patterns."""
        detected = {}
        
        for style, patterns in self.style_patterns.items():
            count = sum(1 for pattern in patterns if pattern in text)
            if count > 0:
                detected[style] = count
        
        return detected
    
    def _detect_languages(self, text: str) -> List[str]:
        """Detect programming languages."""
        languages = []
        
        language_indicators = {
            "python": ["def ", "import ", "from ", "elif", "__init__", ".py"],
            "javascript": ["const ", "let ", "function ", "=>", ".js"],
            "typescript": ["interface ", "type ", ": string", ": number", ".ts"],
            "java": ["public class", "private ", "void ", ".java"],
            "go": ["func ", "package ", "import (", ".go"],
            "rust": ["fn ", "let mut", "impl ", ".rs"],
            "sql": ["SELECT ", "FROM ", "WHERE ", "INSERT ", "UPDATE "],
        }
        
        for lang, indicators in language_indicators.items():
            if any(indicator in text for indicator in indicators):
                languages.append(lang)
        
        return languages
    
    def _extract_patterns(self, text: str) -> List[str]:
        """Extract reusable patterns from text."""
        patterns = []
        
        # Extract function/method names
        func_matches = re.findall(r'(?:def|function|const|let)\s+(\w+)', text)
        patterns.extend(func_matches[:5])
        
        # Extract class names
        class_matches = re.findall(r'class\s+(\w+)', text)
        patterns.extend(class_matches[:3])
        
        return patterns
    
    def get_specialization_prompt(self, user_id: str, agent_type: str) -> str:
        """Generate a specialization prompt based on learned patterns."""
        profile = self.get_profile(user_id, agent_type)
        
        if profile.interactions_count < 5:
            return ""  # Not enough data to specialize
        
        prompt_parts = []
        
        # Add framework preferences
        if profile.preferred_frameworks:
            frameworks = ", ".join(profile.preferred_frameworks[:5])
            prompt_parts.append(f"User prefers these frameworks: {frameworks}.")
        
        # Add coding style
        if profile.coding_style:
            dominant_style = max(profile.coding_style.items(), key=lambda x: x[1])[0]
            prompt_parts.append(f"User tends to use {dominant_style} coding style.")
        
        # Add language preferences
        if profile.project_languages:
            languages = ", ".join(profile.project_languages[:3])
            prompt_parts.append(f"Primary languages: {languages}.")
        
        # Add custom terminology
        if profile.terminology:
            terms = ", ".join([f"'{k}' means '{v}'" for k, v in list(profile.terminology.items())[:3]])
            prompt_parts.append(f"Custom terminology: {terms}.")
        
        if not prompt_parts:
            return ""
        
        return "USER SPECIALIZATION: " + " ".join(prompt_parts)
    
    def add_terminology(self, user_id: str, agent_type: str, term: str, meaning: str):
        """Add custom terminology for a user."""
        profile = self.get_profile(user_id, agent_type)
        profile.terminology[term] = meaning
        profile.last_updated = datetime.now().isoformat()
        logger.debug(f"📚 Added terminology '{term}' for {agent_type}")
    
    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get specialization stats for a user."""
        if user_id not in self.profiles:
            return {"total_profiles": 0, "agents": {}}
        
        user_profiles = self.profiles[user_id]
        return {
            "total_profiles": len(user_profiles),
            "agents": {
                agent_type: {
                    "interactions": profile.interactions_count,
                    "confidence": profile.confidence_score,
                    "frameworks": profile.preferred_frameworks,
                    "languages": profile.project_languages,
                }
                for agent_type, profile in user_profiles.items()
            }
        }
    
    def reset_profile(self, user_id: str, agent_type: Optional[str] = None):
        """Reset specialization profile(s) for a user."""
        if user_id not in self.profiles:
            return
        
        if agent_type:
            if agent_type in self.profiles[user_id]:
                del self.profiles[user_id][agent_type]
        else:
            del self.profiles[user_id]
        
        logger.info(f"🔄 Reset specialization for user {user_id[:8]}...")


# Global instance
agent_specialization = AgentSpecializationEngine()
