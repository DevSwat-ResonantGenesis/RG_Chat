"""
Enhanced Hallucination Detection System (EHDS)
===============================================

Phase 5.16 + 6.0: Advanced hallucination detection and flagging.

Features:
- Detect fabricated libraries/APIs
- Flag non-existent functions
- Identify made-up statistics
- Confidence-based warnings
- RAG-based verification against stored knowledge
- Semantic claim verification
- System-prompt grounding (default, free)
- LLM-as-judge verification (optional, uses second LLM call)
- User knowledge base cross-referencing (uploaded facts/data)
"""
from __future__ import annotations

import re
import logging
import hashlib
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class HallucinationConfig:
    """Per-user hallucination detection configuration."""
    system_prompt_grounding: bool = True  # Default ON - free, checks response vs system prompt
    llm_as_judge: bool = False  # Optional - expensive, uses second LLM call
    knowledge_base_check: bool = False  # Optional - checks against user-uploaded facts


@dataclass
class KnowledgeBaseEntry:
    """A user-uploaded knowledge base entry."""
    id: str
    user_id: str
    title: str
    content: str
    entry_type: str  # 'fact', 'document', 'data', 'book_excerpt'
    created_at: str = ""


@dataclass
class HallucinationFlag:
    """A detected potential hallucination."""
    type: str  # 'fake_library', 'fake_api', 'fake_statistic', 'unsupported_claim'
    content: str
    confidence: float  # 0-1, how confident we are this is a hallucination
    suggestion: str


@dataclass
class HallucinationReport:
    """Full hallucination analysis report."""
    flags: List[HallucinationFlag]
    risk_score: float  # 0-1
    risk_level: str  # 'low', 'medium', 'high'
    summary: str
    should_warn_user: bool
    rag_verification: Optional[Dict[str, Any]] = None  # RAG verification results
    claim_verification: Optional[Dict[str, Any]] = None  # Semantic claim verification


@dataclass
class ClaimExtraction:
    """An extracted factual claim from text."""
    claim: str
    claim_type: str  # 'factual', 'technical', 'statistical', 'reference'
    confidence: float
    source_text: str


class HallucinationDetector:
    """
    Detects potential hallucinations in AI responses.
    """
    
    def __init__(self):
        # Known real libraries/packages (subset for validation)
        self.known_packages = {
            # Python
            "numpy", "pandas", "requests", "flask", "django", "fastapi",
            "sqlalchemy", "pytest", "asyncio", "typing", "dataclasses",
            "pydantic", "httpx", "aiohttp", "celery", "redis", "boto3",
            # JavaScript/Node
            "react", "vue", "angular", "express", "next", "axios",
            "lodash", "moment", "dayjs", "typescript", "webpack", "vite",
            "tailwindcss", "prisma", "mongoose", "sequelize",
            # General
            "docker", "kubernetes", "terraform", "ansible",
        }
        
        # Suspicious patterns that often indicate hallucination
        self.suspicious_patterns = [
            # Fake library patterns
            (r'import\s+(\w+_helper)', 'fake_library', 'Generic helper library'),
            (r'from\s+(\w+_utils)\s+import', 'fake_library', 'Generic utils library'),
            (r'require\([\'"](\w+-magic)[\'"]\)', 'fake_library', 'Magic library'),
            
            # Fake API patterns
            (r'api\.(\w+)\.ai/', 'fake_api', 'Non-standard AI API'),
            (r'https://api\.fake', 'fake_api', 'Fake API URL'),
            
            # Fake statistics
            (r'(\d{2,3})% of (?:developers|companies|users)', 'fake_statistic', 'Unverified statistic'),
            (r'studies show that (\d+%)', 'fake_statistic', 'Uncited study'),
            (r'according to research,? (\d+)', 'fake_statistic', 'Uncited research'),
        ]
        
        # Overconfident claim patterns
        self.overconfident_patterns = [
            r'this is the only way',
            r'you must always',
            r'never use',
            r'always use',
            r'the best practice is always',
            r'everyone knows',
            r'it\'s obvious that',
        ]
        
        # Fabrication indicators
        self.fabrication_indicators = [
            r'the official \w+ documentation states',
            r'according to the \w+ team',
            r'as announced by',
            r'the latest version \d+\.\d+\.\d+ includes',
        ]
        
        # System leak patterns - internal debug output being shown to users
        self.system_leak_patterns = [
            (r'Result:\s*\{[\'"]?\w+[\'"]?:', 'system_leak', 'Internal debug output leaked'),
            (r'method[\'"]?:\s*[\'"]?\w+_eval', 'system_leak', 'Internal method name leaked'),
            (r'evaluated locally', 'system_leak', 'Internal processing message leaked'),
            (r'autonomous_local', 'system_leak', 'Internal provider name leaked'),
            (r'\{[\'"]statement[\'"]:', 'system_leak', 'Raw JSON object in response'),
            (r'Logical operation evaluated', 'system_leak', 'Internal decision framework output'),
            (r'rule_based|llm_required|cached_decision', 'system_leak', 'Internal method type leaked'),
            (r'confidence:\s*\d+\.\d+', 'system_leak', 'Internal confidence score leaked'),
        ]
        
        # Claim extraction patterns for RAG verification
        self.claim_patterns = {
            'factual': [
                r'(?:is|are|was|were)\s+(?:the|a|an)?\s*(\w+(?:\s+\w+){1,5})',
                r'(?:has|have|had)\s+(\w+(?:\s+\w+){1,5})',
            ],
            'technical': [
                r'(?:function|method|class|module)\s+`?(\w+)`?\s+(?:does|returns|takes)',
                r'`(\w+(?:\.\w+)*)`\s+(?:is|provides|handles)',
                r'(?:import|require)\s+(?:from\s+)?[\'"]?(\w+)[\'"]?',
            ],
            'statistical': [
                r'(\d+(?:\.\d+)?%)\s+(?:of|increase|decrease)',
                r'(\d+(?:,\d{3})*)\s+(?:users|requests|transactions)',
                r'(?:average|mean|median)\s+(?:of\s+)?(\d+(?:\.\d+)?)',
            ],
            'reference': [
                r'according to\s+(.+?)(?:\.|,)',
                r'as stated in\s+(.+?)(?:\.|,)',
                r'the documentation (?:says|states)\s+(.+?)(?:\.|,)',
            ],
        }
        
        # Stopwords for claim extraction
        self._stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
        }
    
    def analyze(self, response: str, task: str = "") -> HallucinationReport:
        """Analyze a response for potential hallucinations."""
        flags = []
        
        # Check for suspicious patterns
        for pattern, flag_type, description in self.suspicious_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for match in matches:
                flags.append(HallucinationFlag(
                    type=flag_type,
                    content=match if isinstance(match, str) else match[0],
                    confidence=0.6,
                    suggestion=f"Verify: {description}",
                ))
        
        # Check for unknown packages in imports
        import_flags = self._check_imports(response)
        flags.extend(import_flags)
        
        # Check for overconfident claims
        for pattern in self.overconfident_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                flags.append(HallucinationFlag(
                    type="unsupported_claim",
                    content=pattern,
                    confidence=0.4,
                    suggestion="Absolute claims should be verified",
                ))
        
        # Check for fabrication indicators
        for pattern in self.fabrication_indicators:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for match in matches:
                flags.append(HallucinationFlag(
                    type="unsupported_claim",
                    content=match,
                    confidence=0.5,
                    suggestion="Citation needed for this claim",
                ))
        
        # Check for system leak patterns (internal debug output)
        for pattern, flag_type, description in self.system_leak_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                flags.append(HallucinationFlag(
                    type=flag_type,
                    content=description,
                    confidence=0.95,  # Very high confidence - this is definitely wrong
                    suggestion="Response contains internal system output instead of proper answer",
                ))
        
        # Check for fake version numbers
        version_flags = self._check_versions(response)
        flags.extend(version_flags)
        
        # Calculate risk score
        if not flags:
            risk_score = 0.0
        else:
            risk_score = min(1.0, sum(f.confidence for f in flags) / 3)
        
        # Determine risk level
        if risk_score >= 0.7:
            risk_level = "high"
        elif risk_score >= 0.4:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        # Generate summary
        summary = self._generate_summary(flags, risk_level)
        
        return HallucinationReport(
            flags=flags,
            risk_score=risk_score,
            risk_level=risk_level,
            summary=summary,
            should_warn_user=risk_score >= 0.5,
        )
    
    def _check_imports(self, response: str) -> List[HallucinationFlag]:
        """Check for potentially fake imports."""
        flags = []
        
        # Python imports
        python_imports = re.findall(r'(?:import|from)\s+(\w+)', response)
        
        # JavaScript requires/imports
        js_imports = re.findall(r'(?:require|import)\s*\(?[\'"]([^\'\"]+)[\'"]', response)
        
        all_imports = python_imports + js_imports
        
        for imp in all_imports:
            imp_lower = imp.lower()
            # Skip standard library and known packages
            if imp_lower in self.known_packages:
                continue
            if imp_lower.startswith(('_', '.')):
                continue
            if len(imp_lower) < 3:
                continue
            
            # Flag suspicious package names
            suspicious_suffixes = ['_helper', '_utils', '_magic', '_easy', '_simple', '_pro']
            if any(imp_lower.endswith(s) for s in suspicious_suffixes):
                flags.append(HallucinationFlag(
                    type="fake_library",
                    content=imp,
                    confidence=0.7,
                    suggestion=f"Verify package '{imp}' exists",
                ))
        
        return flags
    
    def _check_versions(self, response: str) -> List[HallucinationFlag]:
        """Check for potentially fake version numbers."""
        flags = []
        
        # Look for version patterns
        version_patterns = [
            r'(\w+)\s+(?:version\s+)?v?(\d+\.\d+\.\d+)',
            r'(\w+)@(\d+\.\d+\.\d+)',
        ]
        
        for pattern in version_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for package, version in matches:
                # Flag suspiciously high version numbers
                try:
                    major = int(version.split('.')[0])
                    if major > 50:  # Most packages don't have major version > 50
                        flags.append(HallucinationFlag(
                            type="fake_version",
                            content=f"{package} {version}",
                            confidence=0.6,
                            suggestion=f"Verify version {version} exists for {package}",
                        ))
                except:
                    pass
        
        return flags
    
    def _generate_summary(self, flags: List[HallucinationFlag], risk_level: str) -> str:
        """Generate a summary of the hallucination analysis."""
        if not flags:
            return "No potential hallucinations detected."
        
        type_counts = {}
        for f in flags:
            type_counts[f.type] = type_counts.get(f.type, 0) + 1
        
        parts = [f"Risk level: {risk_level.upper()}"]
        
        if "fake_library" in type_counts:
            parts.append(f"{type_counts['fake_library']} potentially fake libraries")
        if "fake_api" in type_counts:
            parts.append(f"{type_counts['fake_api']} potentially fake APIs")
        if "fake_statistic" in type_counts:
            parts.append(f"{type_counts['fake_statistic']} unverified statistics")
        if "unsupported_claim" in type_counts:
            parts.append(f"{type_counts['unsupported_claim']} unsupported claims")
        if "system_leak" in type_counts:
            parts.append(f"{type_counts['system_leak']} internal system outputs leaked")
        
        return ". ".join(parts) + "."
    
    def get_warning_message(self, report: HallucinationReport) -> Optional[str]:
        """Get a user-facing warning message if needed."""
        if not report.should_warn_user:
            return None
        
        warning = "⚠️ **Verification Recommended**\n\n"
        warning += "This response may contain information that should be verified:\n"
        
        for flag in report.flags[:3]:  # Show top 3 flags
            warning += f"- {flag.suggestion}\n"
        
        warning += "\nPlease verify critical information before using."
        return warning
    
    def add_verification_notes(self, response: str, report: HallucinationReport) -> str:
        """Add verification notes to a response if needed."""
        if not report.should_warn_user:
            return response
        
        warning = self.get_warning_message(report)
        if warning:
            return response + "\n\n---\n" + warning
        
        return response
    
    # ========================================================================
    # RAG-BASED VERIFICATION METHODS
    # ========================================================================
    
    def extract_claims(self, response: str) -> List[ClaimExtraction]:
        """
        Extract verifiable claims from a response.
        
        Returns a list of claims that can be verified against stored knowledge.
        """
        claims = []
        
        for claim_type, patterns in self.claim_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, response, re.IGNORECASE)
                for match in matches:
                    claim_text = match.group(1) if match.groups() else match.group(0)
                    if claim_text and len(claim_text) > 5:
                        # Clean up the claim
                        claim_text = claim_text.strip()
                        if claim_text.lower() not in self._stopwords:
                            claims.append(ClaimExtraction(
                                claim=claim_text,
                                claim_type=claim_type,
                                confidence=0.7,
                                source_text=match.group(0)[:100]
                            ))
        
        # Deduplicate claims
        seen = set()
        unique_claims = []
        for claim in claims:
            claim_key = claim.claim.lower()
            if claim_key not in seen:
                seen.add(claim_key)
                unique_claims.append(claim)
        
        return unique_claims[:20]  # Limit to 20 claims
    
    def verify_against_rag(
        self, 
        response: str, 
        rag_sources: List[Dict[str, Any]],
        anchors: List[str] = None
    ) -> Dict[str, Any]:
        """
        Verify response claims against RAG sources and anchors.
        
        Returns verification results with support scores.
        """
        if not rag_sources and not anchors:
            return {
                "verified": False,
                "reason": "no_sources",
                "support_score": 0.0,
                "claims_checked": 0,
                "claims_supported": 0,
            }
        
        # Extract claims from response
        claims = self.extract_claims(response)
        if not claims:
            return {
                "verified": True,
                "reason": "no_verifiable_claims",
                "support_score": 1.0,
                "claims_checked": 0,
                "claims_supported": 0,
            }
        
        # Build source text corpus
        source_corpus = ""
        if rag_sources:
            for source in rag_sources:
                source_corpus += " " + source.get("content", "")
        if anchors:
            source_corpus += " " + " ".join(anchors)
        
        source_corpus_lower = source_corpus.lower()
        
        # Check each claim against sources
        supported_claims = 0
        claim_results = []
        
        for claim in claims:
            claim_lower = claim.claim.lower()
            
            # Check for direct mention
            is_supported = False
            support_type = "none"
            
            # Direct string match
            if claim_lower in source_corpus_lower:
                is_supported = True
                support_type = "direct_match"
            else:
                # Check for keyword overlap
                claim_words = set(claim_lower.split()) - self._stopwords
                source_words = set(source_corpus_lower.split())
                overlap = len(claim_words & source_words)
                if overlap >= len(claim_words) * 0.6:  # 60% overlap
                    is_supported = True
                    support_type = "keyword_overlap"
            
            if is_supported:
                supported_claims += 1
            
            claim_results.append({
                "claim": claim.claim[:50],
                "type": claim.claim_type,
                "supported": is_supported,
                "support_type": support_type,
            })
        
        # Calculate support score
        support_score = supported_claims / len(claims) if claims else 1.0
        
        return {
            "verified": support_score >= 0.5,
            "reason": "rag_verification",
            "support_score": round(support_score, 4),
            "claims_checked": len(claims),
            "claims_supported": supported_claims,
            "claim_details": claim_results[:10],  # Limit details
        }
    
    def analyze_with_rag(
        self, 
        response: str, 
        task: str = "",
        rag_sources: List[Dict[str, Any]] = None,
        anchors: List[str] = None
    ) -> HallucinationReport:
        """
        Analyze a response with RAG-based verification.
        
        Enhanced version of analyze() that includes RAG verification.
        """
        # Run base analysis
        report = self.analyze(response, task)
        
        # Add RAG verification if sources available
        if rag_sources or anchors:
            rag_verification = self.verify_against_rag(response, rag_sources or [], anchors or [])
            report.rag_verification = rag_verification
            
            # Adjust risk score based on RAG verification
            if rag_verification["verified"]:
                # Reduce risk if claims are supported
                support_bonus = rag_verification["support_score"] * 0.2
                report.risk_score = max(0.0, report.risk_score - support_bonus)
            else:
                # Increase risk if claims are not supported
                if rag_verification["claims_checked"] > 0:
                    unsupported_penalty = (1 - rag_verification["support_score"]) * 0.15
                    report.risk_score = min(1.0, report.risk_score + unsupported_penalty)
            
            # Recalculate risk level
            if report.risk_score >= 0.7:
                report.risk_level = "high"
            elif report.risk_score >= 0.4:
                report.risk_level = "medium"
            else:
                report.risk_level = "low"
            
            report.should_warn_user = report.risk_score >= 0.5
        
        return report
    
    def compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two texts using keyword overlap.
        
        Simple but effective for hallucination detection.
        """
        if not text1 or not text2:
            return 0.0
        
        # Tokenize and filter
        words1 = set(w.lower() for w in re.findall(r'\b\w{3,}\b', text1)) - self._stopwords
        words2 = set(w.lower() for w in re.findall(r'\b\w{3,}\b', text2)) - self._stopwords
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def ground_against_system_prompt(
        self, 
        response: str, 
        system_prompt: str,
        user_message: str = "",
    ) -> Dict[str, Any]:
        """
        Check if the response contradicts the system prompt instructions.
        
        This is the DEFAULT hallucination check (free, no LLM call).
        Detects cases like the LLM claiming a wrong identity, ignoring instructions, etc.
        """
        if not system_prompt:
            return {"grounded": True, "score": 0.0, "violations": [], "method": "system_prompt_grounding"}
        
        violations = []
        response_lower = response.lower()
        prompt_lower = system_prompt.lower()
        
        # 1. Identity grounding: Check if system prompt defines an identity
        identity_patterns = [
            r'you are (?:called |named )?(\w[\w\s]{1,30})',
            r'your name is (\w[\w\s]{1,30})',
            r'act as (\w[\w\s]{1,30})',
            r'you\'re (\w[\w\s]{1,30})',
        ]
        defined_identities = []
        for pattern in identity_patterns:
            matches = re.findall(pattern, prompt_lower)
            defined_identities.extend(m.strip() for m in matches)
        
        # Check if response claims a different identity
        if defined_identities:
            response_identity_patterns = [
                r'(?:i am|i\'m|my name is|called) (\w[\w\s]{1,30})',
            ]
            for pattern in response_identity_patterns:
                resp_matches = re.findall(pattern, response_lower)
                for resp_identity in resp_matches:
                    resp_id = resp_identity.strip()
                    if resp_id and not any(
                        resp_id in defined or defined in resp_id 
                        for defined in defined_identities
                    ):
                        violations.append({
                            "type": "identity_mismatch",
                            "detail": f"Response claims identity '{resp_id}' but system prompt defines: {defined_identities}",
                            "confidence": 0.85,
                        })
        
        # 2. Instruction violation: Check for "never" / "always" / "do not" directives
        directive_patterns = [
            (r'never\s+(.{5,60}?)(?:\.|,|;|\n)', 'never'),
            (r'do not\s+(.{5,60}?)(?:\.|,|;|\n)', 'do_not'),
            (r'don\'t\s+(.{5,60}?)(?:\.|,|;|\n)', 'do_not'),
            (r'always\s+(.{5,60}?)(?:\.|,|;|\n)', 'always'),
            (r'must\s+(.{5,60}?)(?:\.|,|;|\n)', 'must'),
        ]
        
        for pattern, directive_type in directive_patterns:
            directives = re.findall(pattern, prompt_lower)
            for directive in directives:
                directive_clean = directive.strip()
                if len(directive_clean) < 5:
                    continue
                # Extract key words from the directive
                key_words = set(re.findall(r'\b\w{4,}\b', directive_clean)) - self._stopwords
                if not key_words:
                    continue
                # Check if the response violates this directive
                response_words = set(re.findall(r'\b\w{4,}\b', response_lower))
                overlap = key_words & response_words
                if directive_type in ('never', 'do_not') and len(overlap) >= len(key_words) * 0.5:
                    violations.append({
                        "type": "directive_violation",
                        "detail": f"Response may violate '{directive_type}' directive: '{directive_clean[:60]}'",
                        "confidence": 0.5,
                    })
        
        # 3. Role deviation: Check if LLM breaks character
        role_break_patterns = [
            r'as an ai(?:\s+language)?\s+model',
            r'i(?:\'m| am) (?:just )?(?:an? )?(?:ai|artificial intelligence|language model|llm)',
            r'i don\'t have (?:feelings|emotions|opinions|consciousness)',
            r'i was (?:created|made|built|trained) by',
        ]
        # Only flag if system prompt doesn't mention these patterns
        if 'ai' not in prompt_lower[:200]:
            for pattern in role_break_patterns:
                if re.search(pattern, response_lower):
                    violations.append({
                        "type": "role_deviation",
                        "detail": f"Response breaks character by referencing AI nature",
                        "confidence": 0.6,
                    })
                    break  # Only flag once
        
        # Calculate grounding score (higher = more violations = worse)
        if not violations:
            grounding_score = 0.0
        else:
            grounding_score = min(1.0, sum(v["confidence"] for v in violations) / 2.0)
        
        return {
            "grounded": len(violations) == 0,
            "score": round(grounding_score, 4),
            "violations": violations[:5],
            "method": "system_prompt_grounding",
        }
    
    def check_against_knowledge_base(
        self, 
        response: str, 
        kb_corpus: str,
    ) -> Dict[str, Any]:
        """
        Cross-reference response against user-uploaded knowledge base.
        
        Checks if the response contradicts or is unsupported by user-provided facts.
        """
        if not kb_corpus:
            return {"checked": False, "score": 0.0, "method": "knowledge_base", "reason": "no_entries"}
        
        # Extract claims from response
        claims = self.extract_claims(response)
        if not claims:
            return {"checked": True, "score": 0.0, "method": "knowledge_base", 
                    "reason": "no_verifiable_claims", "claims_checked": 0, "contradictions": []}
        
        kb_lower = kb_corpus.lower()
        contradictions = []
        supported = 0
        
        for claim in claims[:10]:
            claim_lower = claim.claim.lower()
            claim_words = set(re.findall(r'\b\w{4,}\b', claim_lower)) - self._stopwords
            if not claim_words:
                continue
            
            kb_words = set(re.findall(r'\b\w{4,}\b', kb_lower))
            overlap = claim_words & kb_words
            
            # High overlap = claim topic exists in KB
            if len(overlap) >= len(claim_words) * 0.5:
                # Check for contradictory terms
                negation_in_kb = any(
                    f"not {w}" in kb_lower or f"no {w}" in kb_lower
                    for w in overlap
                )
                if negation_in_kb:
                    contradictions.append({
                        "claim": claim.claim[:80],
                        "type": "kb_contradiction",
                        "confidence": 0.7,
                    })
                else:
                    supported += 1
        
        total_checked = min(len(claims), 10)
        contradiction_score = len(contradictions) / max(total_checked, 1) if total_checked > 0 else 0.0
        
        return {
            "checked": True,
            "score": round(contradiction_score, 4),
            "method": "knowledge_base",
            "claims_checked": total_checked,
            "claims_supported": supported,
            "contradictions": contradictions[:5],
        }
    
    async def llm_judge_verify(
        self,
        response: str,
        user_message: str = "",
        system_prompt: str = "",
        router=None,
    ) -> Dict[str, Any]:
        """
        Use a second LLM call to judge the response for hallucinations.
        
        EXPENSIVE: Uses a second LLM call. Only run when user opts in.
        """
        if not router:
            return {"judged": False, "score": 0.0, "method": "llm_as_judge", "reason": "no_router"}
        
        judge_prompt = (
            "You are a hallucination judge. Analyze the following AI response for:\n"
            "1. Factual accuracy - are claims verifiable or made up?\n"
            "2. Identity accuracy - does the AI correctly identify itself?\n"
            "3. Consistency - does the response contradict itself?\n"
            "4. Groundedness - are claims supported by the context?\n\n"
            f"User question: {user_message[:500]}\n\n"
            f"AI response to judge: {response[:2000]}\n\n"
            "Respond with ONLY a JSON object (no markdown):\n"
            '{"hallucination_score": 0.0-1.0, "issues": ["issue1", "issue2"], "verdict": "clean|minor|major"}\n'
            "Score 0.0 = no hallucination, 1.0 = severe hallucination."
        )
        
        try:
            judge_response = await router.route_query(
                    message=judge_prompt,
                    context=[],
                    preferred_provider="groq",  # Use fast/cheap provider for judging
                )
            
            judge_text = judge_response.get("response", "")
            
            # Parse judge response
            import json
            # Try to extract JSON from response
            json_match = re.search(r'\{[^}]+\}', judge_text)
            if json_match:
                try:
                    judge_data = json.loads(json_match.group(0))
                    return {
                        "judged": True,
                        "score": min(1.0, max(0.0, float(judge_data.get("hallucination_score", 0.0)))),
                        "issues": judge_data.get("issues", [])[:5],
                        "verdict": judge_data.get("verdict", "unknown"),
                        "method": "llm_as_judge",
                    }
                except (json.JSONDecodeError, ValueError):
                    pass
            
            # Fallback: check for keywords
            score = 0.0
            if any(w in judge_text.lower() for w in ["hallucination", "fabricat", "made up", "incorrect", "false"]):
                score = 0.5
            if any(w in judge_text.lower() for w in ["severe", "major", "significant"]):
                score = 0.8
            
            return {
                "judged": True,
                "score": score,
                "issues": [judge_text[:200]],
                "verdict": "major" if score > 0.6 else "minor" if score > 0.3 else "clean",
                "method": "llm_as_judge",
            }
        
        except Exception as e:
            logger.error(f"LLM judge verification failed: {e}")
            return {"judged": False, "score": 0.0, "method": "llm_as_judge", "reason": str(e)}
    
    async def analyze_full(
        self,
        response: str,
        task: str = "",
        config: Optional[HallucinationConfig] = None,
        system_prompt: str = "",
        user_message: str = "",
        kb_corpus: str = "",
        rag_sources: List[Dict[str, Any]] = None,
        anchors: List[str] = None,
        router=None,
    ) -> HallucinationReport:
        """
        Full hallucination analysis using all enabled detection methods.
        
        Combines: base regex analysis, RAG verification, system-prompt grounding,
        knowledge base cross-referencing, and LLM-as-judge.
        """
        if config is None:
            config = HallucinationConfig()
        
        # 1. Base analysis (always runs - regex patterns)
        report = self.analyze_with_rag(response, task, rag_sources, anchors)
        
        grounding_result = None
        kb_result = None
        judge_result = None
        
        # 2. System-prompt grounding (default ON, free)
        if config.system_prompt_grounding and system_prompt:
            grounding_result = self.ground_against_system_prompt(response, system_prompt, user_message)
            if grounding_result["score"] > 0:
                # Add grounding violations as flags
                for v in grounding_result.get("violations", []):
                    report.flags.append(HallucinationFlag(
                        type=v["type"],
                        content=v["detail"][:100],
                        confidence=v["confidence"],
                        suggestion=f"System prompt grounding violation: {v['type']}",
                    ))
                # Blend grounding score into risk
                report.risk_score = min(1.0, report.risk_score + grounding_result["score"] * 0.4)
        
        # 3. Knowledge base cross-referencing (optional)
        if config.knowledge_base_check and kb_corpus:
            kb_result = self.check_against_knowledge_base(response, kb_corpus)
            if kb_result.get("score", 0) > 0:
                for c in kb_result.get("contradictions", []):
                    report.flags.append(HallucinationFlag(
                        type="kb_contradiction",
                        content=c["claim"][:100],
                        confidence=c["confidence"],
                        suggestion="Contradicts user knowledge base",
                    ))
                report.risk_score = min(1.0, report.risk_score + kb_result["score"] * 0.3)
        
        # 4. LLM-as-judge (optional, expensive)
        if config.llm_as_judge and router:
            judge_result = await self.llm_judge_verify(response, user_message, system_prompt, router)
            if judge_result.get("score", 0) > 0:
                for issue in judge_result.get("issues", []):
                    report.flags.append(HallucinationFlag(
                        type="llm_judge",
                        content=str(issue)[:100],
                        confidence=judge_result["score"],
                        suggestion="Flagged by LLM judge verification",
                    ))
                # LLM judge has high weight
                report.risk_score = min(1.0, report.risk_score + judge_result["score"] * 0.5)
        
        # Recalculate risk level
        if report.risk_score >= 0.7:
            report.risk_level = "high"
        elif report.risk_score >= 0.3:
            report.risk_level = "medium"
        else:
            report.risk_level = "low"
        
        report.should_warn_user = report.risk_score >= 0.4
        
        # Attach detailed verification results
        report.claim_verification = {
            "system_prompt_grounding": grounding_result,
            "knowledge_base": kb_result,
            "llm_as_judge": judge_result,
            "methods_used": [
                m for m, used in [
                    ("base_regex", True),
                    ("rag_verification", bool(rag_sources or anchors)),
                    ("system_prompt_grounding", config.system_prompt_grounding and bool(system_prompt)),
                    ("knowledge_base", config.knowledge_base_check and bool(kb_corpus)),
                    ("llm_as_judge", config.llm_as_judge and router is not None),
                ] if used
            ],
        }
        
        # Regenerate summary
        report.summary = self._generate_summary(report.flags, report.risk_level)
        
        return report
    
    def check_consistency(self, response: str, previous_responses: List[str]) -> Dict[str, Any]:
        """
        Check if response is consistent with previous responses in conversation.
        
        Detects contradictions that may indicate hallucination.
        """
        if not previous_responses:
            return {"consistent": True, "reason": "no_previous", "score": 1.0}
        
        # Check for direct contradictions
        contradiction_pairs = [
            (r'\bis\s+(\w+)', r'\bis\s+not\s+\1'),
            (r'\bcan\s+(\w+)', r'\bcannot\s+\1'),
            (r'\bwill\s+(\w+)', r'\bwill\s+not\s+\1'),
            (r'\bshould\s+(\w+)', r'\bshould\s+not\s+\1'),
        ]
        
        response_lower = response.lower()
        contradictions = []
        
        for prev in previous_responses[-5:]:  # Check last 5 responses
            prev_lower = prev.lower()
            
            for pos_pattern, neg_pattern in contradiction_pairs:
                # Check if response contradicts previous
                pos_in_prev = re.search(pos_pattern, prev_lower)
                neg_in_resp = re.search(neg_pattern, response_lower)
                
                if pos_in_prev and neg_in_resp:
                    contradictions.append({
                        "previous": pos_in_prev.group(0),
                        "current": neg_in_resp.group(0),
                    })
                
                # Check reverse
                neg_in_prev = re.search(neg_pattern, prev_lower)
                pos_in_resp = re.search(pos_pattern, response_lower)
                
                if neg_in_prev and pos_in_resp:
                    contradictions.append({
                        "previous": neg_in_prev.group(0),
                        "current": pos_in_resp.group(0),
                    })
        
        consistency_score = 1.0 - (len(contradictions) * 0.2)
        consistency_score = max(0.0, min(1.0, consistency_score))
        
        return {
            "consistent": len(contradictions) == 0,
            "reason": "contradiction_check",
            "score": round(consistency_score, 4),
            "contradictions": contradictions[:5],
        }


# Global instance
hallucination_detector = HallucinationDetector()


class UserKnowledgeBase:
    """
    Manages per-user knowledge bases for cross-referencing.
    Users can upload facts, data, documents, book excerpts.
    """
    
    def __init__(self):
        # In-memory storage: {user_id: [KnowledgeBaseEntry, ...]}
        self._entries: Dict[str, List[KnowledgeBaseEntry]] = {}
    
    def add_entry(self, user_id: str, entry_id: str, title: str, content: str, entry_type: str = "fact") -> KnowledgeBaseEntry:
        """Add a knowledge base entry for a user."""
        if user_id not in self._entries:
            self._entries[user_id] = []
        entry = KnowledgeBaseEntry(
            id=entry_id,
            user_id=user_id,
            title=title,
            content=content,
            entry_type=entry_type,
        )
        self._entries[user_id].append(entry)
        logger.info(f"Added KB entry '{title}' ({entry_type}) for user {user_id}")
        return entry
    
    def get_entries(self, user_id: str) -> List[KnowledgeBaseEntry]:
        """Get all knowledge base entries for a user."""
        return self._entries.get(user_id, [])
    
    def delete_entry(self, user_id: str, entry_id: str) -> bool:
        """Delete a knowledge base entry."""
        if user_id not in self._entries:
            return False
        before = len(self._entries[user_id])
        self._entries[user_id] = [e for e in self._entries[user_id] if e.id != entry_id]
        deleted = len(self._entries[user_id]) < before
        if deleted:
            logger.info(f"Deleted KB entry {entry_id} for user {user_id}")
        return deleted
    
    def get_corpus(self, user_id: str) -> str:
        """Get combined text corpus from all user entries."""
        entries = self.get_entries(user_id)
        if not entries:
            return ""
        return "\n\n".join(f"[{e.title}]: {e.content}" for e in entries)


# Global knowledge base instance
user_knowledge_base = UserKnowledgeBase()


class HallucinationSettingsManager:
    """
    Manages per-user hallucination detection settings.
    """
    
    def __init__(self):
        self._configs: Dict[str, HallucinationConfig] = {}
    
    def get_config(self, user_id: str) -> HallucinationConfig:
        """Get hallucination config for a user (defaults if not set)."""
        if user_id not in self._configs:
            self._configs[user_id] = HallucinationConfig()
        return self._configs[user_id]
    
    def update_config(self, user_id: str, **kwargs) -> HallucinationConfig:
        """Update hallucination config for a user."""
        config = self.get_config(user_id)
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        logger.info(f"Updated hallucination config for user {user_id}: {kwargs}")
        return config


# Global settings manager
hallucination_settings = HallucinationSettingsManager()
