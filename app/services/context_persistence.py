"""
Context Persistence System (CPS)
=================================

Phase 5.7: Remember project context across sessions.

Features:
- Store project-specific context
- Persist file structures and patterns
- Remember user preferences per project
- Cross-session continuity
"""
from __future__ import annotations

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class ProjectContext:
    """Stored context for a project."""
    project_id: str
    user_id: str
    name: str
    
    # Technical context
    languages: List[str] = field(default_factory=list)
    frameworks: List[str] = field(default_factory=list)
    file_patterns: Dict[str, str] = field(default_factory=dict)  # pattern -> description
    key_files: List[str] = field(default_factory=list)
    
    # Learned patterns
    coding_conventions: Dict[str, str] = field(default_factory=dict)
    common_imports: List[str] = field(default_factory=list)
    naming_patterns: Dict[str, str] = field(default_factory=dict)
    
    # Session data
    recent_files: List[str] = field(default_factory=list)
    recent_topics: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: str = ""
    updated_at: str = ""
    session_count: int = 0


@dataclass
class SessionContext:
    """Context for a single session."""
    session_id: str
    project_id: str
    user_id: str
    started_at: str
    
    # Session-specific data
    open_files: List[str] = field(default_factory=list)
    current_task: str = ""
    conversation_summary: str = ""
    key_decisions: List[str] = field(default_factory=list)


class ContextPersistenceEngine:
    """
    Manages persistent context across sessions.
    """
    
    def __init__(self, context_ttl_days: int = 90):
        self.projects: Dict[str, ProjectContext] = {}  # project_id -> context
        self.sessions: Dict[str, SessionContext] = {}  # session_id -> context
        self.user_projects: Dict[str, List[str]] = {}  # user_id -> project_ids
        self.context_ttl_days = context_ttl_days
    
    def _generate_project_id(self, user_id: str, project_name: str) -> str:
        """Generate unique project ID."""
        content = f"{user_id}:{project_name}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]
    
    def _generate_session_id(self, user_id: str, project_id: str) -> str:
        """Generate unique session ID."""
        content = f"{user_id}:{project_id}:{datetime.now().isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def create_or_get_project(
        self,
        user_id: str,
        project_name: str,
    ) -> ProjectContext:
        """Create or retrieve project context."""
        project_id = self._generate_project_id(user_id, project_name)
        
        if project_id in self.projects:
            project = self.projects[project_id]
            project.updated_at = datetime.now().isoformat()
            project.session_count += 1
            return project
        
        project = ProjectContext(
            project_id=project_id,
            user_id=user_id,
            name=project_name,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            session_count=1,
        )
        
        self.projects[project_id] = project
        
        if user_id not in self.user_projects:
            self.user_projects[user_id] = []
        self.user_projects[user_id].append(project_id)
        
        logger.info(f"📁 Created project context: {project_name} ({project_id})")
        return project
    
    def start_session(
        self,
        user_id: str,
        project_id: str,
        open_files: List[str] = None,
    ) -> SessionContext:
        """Start a new session for a project."""
        session_id = self._generate_session_id(user_id, project_id)
        
        session = SessionContext(
            session_id=session_id,
            project_id=project_id,
            user_id=user_id,
            started_at=datetime.now().isoformat(),
            open_files=open_files or [],
        )
        
        self.sessions[session_id] = session
        
        # Update project with session info
        if project_id in self.projects:
            project = self.projects[project_id]
            if open_files:
                # Add to recent files (keep last 20)
                for f in open_files:
                    if f not in project.recent_files:
                        project.recent_files.insert(0, f)
                project.recent_files = project.recent_files[:20]
        
        logger.info(f"🚀 Started session: {session_id}")
        return session
    
    def update_project_context(
        self,
        project_id: str,
        languages: List[str] = None,
        frameworks: List[str] = None,
        file_patterns: Dict[str, str] = None,
        key_files: List[str] = None,
        coding_conventions: Dict[str, str] = None,
        common_imports: List[str] = None,
        naming_patterns: Dict[str, str] = None,
    ):
        """Update project context with learned information."""
        if project_id not in self.projects:
            return
        
        project = self.projects[project_id]
        
        if languages:
            for lang in languages:
                if lang not in project.languages:
                    project.languages.append(lang)
        
        if frameworks:
            for fw in frameworks:
                if fw not in project.frameworks:
                    project.frameworks.append(fw)
        
        if file_patterns:
            project.file_patterns.update(file_patterns)
        
        if key_files:
            for f in key_files:
                if f not in project.key_files:
                    project.key_files.append(f)
        
        if coding_conventions:
            project.coding_conventions.update(coding_conventions)
        
        if common_imports:
            for imp in common_imports:
                if imp not in project.common_imports:
                    project.common_imports.append(imp)
        
        if naming_patterns:
            project.naming_patterns.update(naming_patterns)
        
        project.updated_at = datetime.now().isoformat()
    
    def add_topic(self, project_id: str, topic: str):
        """Add a topic to recent topics."""
        if project_id not in self.projects:
            return
        
        project = self.projects[project_id]
        if topic not in project.recent_topics:
            project.recent_topics.insert(0, topic)
        project.recent_topics = project.recent_topics[:10]
    
    def update_session(
        self,
        session_id: str,
        current_task: str = None,
        conversation_summary: str = None,
        key_decision: str = None,
    ):
        """Update session context."""
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        
        if current_task:
            session.current_task = current_task
        
        if conversation_summary:
            session.conversation_summary = conversation_summary
        
        if key_decision:
            session.key_decisions.append(key_decision)
    
    def get_project_context(self, project_id: str) -> Optional[ProjectContext]:
        """Get project context."""
        return self.projects.get(project_id)
    
    def get_session_context(self, session_id: str) -> Optional[SessionContext]:
        """Get session context."""
        return self.sessions.get(session_id)
    
    def get_user_projects(self, user_id: str) -> List[ProjectContext]:
        """Get all projects for a user."""
        project_ids = self.user_projects.get(user_id, [])
        return [
            self.projects[pid]
            for pid in project_ids
            if pid in self.projects
        ]
    
    def build_context_prompt(self, project_id: str) -> str:
        """Build a context prompt from stored project information."""
        project = self.projects.get(project_id)
        if not project:
            return ""
        
        parts = ["PROJECT CONTEXT:"]
        
        if project.languages:
            parts.append(f"Languages: {', '.join(project.languages)}")
        
        if project.frameworks:
            parts.append(f"Frameworks: {', '.join(project.frameworks)}")
        
        if project.coding_conventions:
            conventions = "; ".join([f"{k}: {v}" for k, v in list(project.coding_conventions.items())[:5]])
            parts.append(f"Conventions: {conventions}")
        
        if project.naming_patterns:
            patterns = "; ".join([f"{k}: {v}" for k, v in list(project.naming_patterns.items())[:3]])
            parts.append(f"Naming: {patterns}")
        
        if project.recent_topics:
            parts.append(f"Recent topics: {', '.join(project.recent_topics[:5])}")
        
        if project.key_files:
            parts.append(f"Key files: {', '.join(project.key_files[:5])}")
        
        return "\n".join(parts)
    
    def learn_from_code(self, project_id: str, code: str, file_path: str = ""):
        """Learn patterns from code."""
        if project_id not in self.projects:
            return
        
        project = self.projects[project_id]
        
        # Detect language
        if file_path:
            ext = file_path.split(".")[-1].lower()
            lang_map = {
                "py": "Python", "js": "JavaScript", "ts": "TypeScript",
                "tsx": "TypeScript/React", "jsx": "JavaScript/React",
                "java": "Java", "go": "Go", "rs": "Rust",
                "css": "CSS", "scss": "SCSS", "html": "HTML",
            }
            if ext in lang_map and lang_map[ext] not in project.languages:
                project.languages.append(lang_map[ext])
        
        # Detect frameworks from imports
        code_lower = code.lower()
        framework_indicators = {
            "react": ["import react", "from 'react'", "from \"react\""],
            "vue": ["import vue", "from 'vue'"],
            "angular": ["@angular/", "@component"],
            "express": ["require('express')", "from 'express'"],
            "fastapi": ["from fastapi", "import fastapi"],
            "django": ["from django", "import django"],
            "nextjs": ["from 'next'", "next/"],
            "tailwind": ["tailwind", "className="],
        }
        
        for framework, indicators in framework_indicators.items():
            if any(ind in code_lower for ind in indicators):
                if framework not in project.frameworks:
                    project.frameworks.append(framework)
        
        # Extract common imports
        import re
        import_patterns = [
            r"import\s+(\w+)",
            r"from\s+['\"]([^'\"]+)['\"]",
            r"require\(['\"]([^'\"]+)['\"]\)",
        ]
        
        for pattern in import_patterns:
            matches = re.findall(pattern, code)
            for match in matches[:5]:
                if match not in project.common_imports and len(project.common_imports) < 50:
                    project.common_imports.append(match)
        
        project.updated_at = datetime.now().isoformat()
    
    def cleanup_old_contexts(self):
        """Remove old contexts beyond TTL."""
        cutoff = datetime.now() - timedelta(days=self.context_ttl_days)
        
        to_remove = []
        for project_id, project in self.projects.items():
            try:
                updated = datetime.fromisoformat(project.updated_at)
                if updated < cutoff:
                    to_remove.append(project_id)
            except:
                pass
        
        for project_id in to_remove:
            del self.projects[project_id]
            for user_id, project_ids in self.user_projects.items():
                if project_id in project_ids:
                    project_ids.remove(project_id)
        
        if to_remove:
            logger.info(f"🧹 Cleaned up {len(to_remove)} old project contexts")
    
    def export_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Export project context as dict."""
        project = self.projects.get(project_id)
        if not project:
            return None
        return asdict(project)


# Global instance
context_persistence = ContextPersistenceEngine()
