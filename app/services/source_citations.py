"""
Source Citations System (SCS)
==============================

Phase 5.13: Agents cite documentation and sources in responses.

Features:
- Extract and format citations
- Link to documentation
- Reference code sources
- Citation validation
"""
from __future__ import annotations

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """A single citation."""
    id: str
    source_type: str  # 'documentation', 'code', 'url', 'reference'
    title: str
    url: Optional[str] = None
    snippet: Optional[str] = None
    confidence: float = 1.0


@dataclass
class CitedResponse:
    """Response with citations."""
    content: str
    citations: List[Citation]
    citation_count: int
    has_verified_sources: bool


class SourceCitationEngine:
    """
    Manages source citations in agent responses.
    """
    
    def __init__(self):
        # Known documentation sources
        self.doc_sources = {
            "react": "https://react.dev/reference",
            "vue": "https://vuejs.org/api/",
            "angular": "https://angular.io/api",
            "python": "https://docs.python.org/3/",
            "javascript": "https://developer.mozilla.org/en-US/docs/Web/JavaScript",
            "typescript": "https://www.typescriptlang.org/docs/",
            "nodejs": "https://nodejs.org/api/",
            "fastapi": "https://fastapi.tiangolo.com/",
            "django": "https://docs.djangoproject.com/",
            "flask": "https://flask.palletsprojects.com/",
            "nextjs": "https://nextjs.org/docs",
            "tailwind": "https://tailwindcss.com/docs",
            "postgres": "https://www.postgresql.org/docs/",
            "mongodb": "https://www.mongodb.com/docs/",
            "redis": "https://redis.io/docs/",
            "docker": "https://docs.docker.com/",
            "kubernetes": "https://kubernetes.io/docs/",
            "aws": "https://docs.aws.amazon.com/",
            "git": "https://git-scm.com/docs",
        }
        
        # Citation patterns to detect
        self.citation_patterns = [
            r'according to (?:the )?(\w+) (?:documentation|docs)',
            r'as (?:documented|described) in (\w+)',
            r'see (?:the )?(\w+) (?:documentation|docs|reference)',
            r'from (?:the )?(\w+) (?:documentation|docs)',
            r'(?:the )?(\w+) (?:documentation|docs) (?:states|says|mentions)',
        ]
        
        # URL pattern
        self.url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    
    def extract_citations(self, text: str) -> List[Citation]:
        """Extract citations from text."""
        citations = []
        citation_id = 0
        
        # Extract URLs
        urls = re.findall(self.url_pattern, text)
        for url in urls:
            citation_id += 1
            citations.append(Citation(
                id=f"cite_{citation_id}",
                source_type="url",
                title=self._get_url_title(url),
                url=url,
                confidence=0.9,
            ))
        
        # Extract documentation references
        text_lower = text.lower()
        for pattern in self.citation_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if match in self.doc_sources:
                    citation_id += 1
                    citations.append(Citation(
                        id=f"cite_{citation_id}",
                        source_type="documentation",
                        title=f"{match.title()} Documentation",
                        url=self.doc_sources[match],
                        confidence=0.8,
                    ))
        
        # Extract code references (file paths)
        file_patterns = [
            r'`([a-zA-Z0-9_/\-\.]+\.[a-zA-Z]+)`',
            r'in (?:file )?[\'"]?([a-zA-Z0-9_/\-\.]+\.[a-zA-Z]+)[\'"]?',
        ]
        for pattern in file_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if '.' in match and len(match) < 100:
                    citation_id += 1
                    citations.append(Citation(
                        id=f"cite_{citation_id}",
                        source_type="code",
                        title=match,
                        confidence=0.7,
                    ))
        
        # Deduplicate
        seen = set()
        unique_citations = []
        for c in citations:
            key = (c.source_type, c.title, c.url)
            if key not in seen:
                seen.add(key)
                unique_citations.append(c)
        
        return unique_citations
    
    def _get_url_title(self, url: str) -> str:
        """Extract a title from URL."""
        # Remove protocol
        title = re.sub(r'^https?://', '', url)
        # Get domain
        domain = title.split('/')[0]
        # Clean up
        domain = domain.replace('www.', '')
        return domain
    
    def add_citations_to_response(
        self,
        response: str,
        task: str = "",
        agent_type: str = "",
    ) -> CitedResponse:
        """Add citation markers to a response."""
        citations = self.extract_citations(response)
        
        # Add inline citation markers if not already present
        modified_response = response
        
        # Add documentation links for known technologies
        for tech, url in self.doc_sources.items():
            # Only add if tech is mentioned but not already linked
            if tech in response.lower() and url not in response:
                # Check if we should add a citation
                pattern = rf'\b{tech}\b'
                if re.search(pattern, response, re.IGNORECASE):
                    # Add to citations if not already there
                    if not any(c.url == url for c in citations):
                        citations.append(Citation(
                            id=f"cite_{len(citations)+1}",
                            source_type="documentation",
                            title=f"{tech.title()} Documentation",
                            url=url,
                            confidence=0.6,
                        ))
        
        # Add citation section if citations exist
        if citations:
            citation_section = "\n\n---\n**Sources:**\n"
            for i, c in enumerate(citations[:5]):  # Limit to 5 citations
                if c.url:
                    citation_section += f"- [{c.title}]({c.url})\n"
                else:
                    citation_section += f"- {c.title}\n"
            
            # Only add if not already has sources section
            if "**Sources:**" not in response and "## Sources" not in response:
                modified_response = response + citation_section
        
        return CitedResponse(
            content=modified_response,
            citations=citations,
            citation_count=len(citations),
            has_verified_sources=any(c.url for c in citations),
        )
    
    def generate_citation_prompt(self, agent_type: str) -> str:
        """Generate a prompt addition to encourage citations."""
        return (
            "When referencing documentation, APIs, or external sources, "
            "include the source name. For example: 'According to the React documentation...' "
            "or 'As described in the Python docs...'. "
            "Include URLs when referencing specific documentation pages."
        )
    
    def validate_citations(self, citations: List[Citation]) -> List[Citation]:
        """Validate citations (check if URLs are from known sources)."""
        validated = []
        for c in citations:
            if c.url:
                # Check if URL is from a known documentation source
                for tech, base_url in self.doc_sources.items():
                    if base_url in c.url or tech in c.url.lower():
                        c.confidence = min(c.confidence + 0.1, 1.0)
                        break
            validated.append(c)
        return validated
    
    def format_citations_markdown(self, citations: List[Citation]) -> str:
        """Format citations as markdown."""
        if not citations:
            return ""
        
        lines = ["", "---", "### References", ""]
        
        for i, c in enumerate(citations, 1):
            if c.url:
                lines.append(f"{i}. [{c.title}]({c.url})")
            else:
                lines.append(f"{i}. {c.title}")
            
            if c.snippet:
                lines.append(f"   > {c.snippet[:100]}...")
        
        return "\n".join(lines)
    
    def get_relevant_docs(self, task: str) -> List[Dict[str, str]]:
        """Get relevant documentation links for a task."""
        task_lower = task.lower()
        relevant = []
        
        for tech, url in self.doc_sources.items():
            if tech in task_lower:
                relevant.append({
                    "technology": tech,
                    "url": url,
                    "title": f"{tech.title()} Documentation",
                })
        
        return relevant


# Global instance
source_citations = SourceCitationEngine()
