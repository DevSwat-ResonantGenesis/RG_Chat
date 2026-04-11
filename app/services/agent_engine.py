"""
Multi-Threaded Agent Spawning Engine (MATSE)
==============================================

Patch #40: Gives Resonant Chat the ability to spawn internal sub-agents for:
- Code generation
- Research
- Summary
- Debugging
- Reasoning
- Planning

Ported from old backend: ResonantGraphAIV0.1/backend/fastapi_app/services/agent_engine.py
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List

from .adaptive_agent_allocator import adaptive_agent_allocator
from .agent_capability_registry import agent_capability_registry
from .autonomous_agent_executor import get_autonomous_executor

logger = logging.getLogger(__name__)


class AgentEngine:
    """
    Multi-Threaded Agent Spawning Engine
    
    Spawns internal sub-agents for specialized tasks without exposing
    chain-of-thought to the user.
    """
    
    def __init__(
        self,
        router=None,
        use_adaptive_allocation: bool = True,
        use_autonomous_decisions: bool = True
    ):
        self.router = router
        self.use_adaptive_allocation = use_adaptive_allocation
        self.use_autonomous_decisions = use_autonomous_decisions
        self.allocator = adaptive_agent_allocator if use_adaptive_allocation else None
        self.registry = agent_capability_registry
        logger.info(
            f"AgentEngine initialized "
            f"(adaptive_allocation={use_adaptive_allocation}, "
            f"autonomous_decisions={use_autonomous_decisions})"
        )
    
    def set_router(self, router):
        """Set the AI router for making LLM calls."""
        self.router = router
    
    async def spawn(
        self,
        task: str,
        context: List[Dict[str, Any]],
        model: Optional[str] = None,
        agent_type: Optional[str] = None,
        user_id: Optional[str] = None,
        images: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Spawn an agent with performance tracking and credit deduction."""
        start_time = time.time()
        success = False
        
        # Track workload
        if agent_type and self.use_adaptive_allocation:
            self.registry.increment_workload(agent_type)
        
        try:
            result = await self._spawn_internal(task, context, model, agent_type, images)
            success = bool(result.get("content"))
            
            # Deduct credits for agent step if user_id provided
            if user_id and success:
                try:
                    from .credit_deduction import deduct_credits
                    await deduct_credits(
                        user_id=user_id,
                        action="agent_step",
                        amount=500,  # Agent step cost from pricing.yaml
                        description=f"Agent {agent_type or 'reasoning'} execution"
                    )
                    logger.info(f"💳 Deducted 500 credits for agent step ({agent_type})")
                except Exception as e:
                    logger.warning(f"Agent credit deduction failed: {e}")
            
            return result
        finally:
            # Track performance metrics
            if agent_type and self.use_adaptive_allocation:
                response_time_ms = (time.time() - start_time) * 1000
                self.registry.update_success_rate(agent_type, success)
                self.registry.update_response_time(agent_type, response_time_ms)
                self.registry.decrement_workload(agent_type)
    
    async def _spawn_internal(
        self,
        task: str,
        context: List[Dict[str, Any]],
        model: Optional[str] = None,
        agent_type: Optional[str] = None,
        images: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Spawn a lightweight reasoning agent for a specific task."""
        try:
            # Determine agent type if not specified
            if not agent_type:
                if self.use_adaptive_allocation and self.allocator:
                    agent_type, score = self.allocator.select_best_agent(task)
                    logger.debug(f"Adaptive allocation selected: {agent_type}")
                else:
                    agent_type = self.should_spawn_agent(task)
            
            # Try autonomous decision-making first (Phase 2)
            # Skip if images are present (vision needs multimodal provider call)
            if self.use_autonomous_decisions and not images:
                try:
                    executor = get_autonomous_executor(agent_type, self.router)
                    result = await executor.execute_task(task, context, model)
                    
                    # If local decision was made, return immediately
                    if result.llm_calls == 0:
                        logger.info(
                            f"[Autonomous] {agent_type} made local decision "
                            f"(method: {result.method}, confidence: {result.confidence:.2%})"
                        )
                        return {
                            "content": result.content,
                            "provider": "autonomous_local",
                            "agent_type": agent_type,
                            "task": task,
                            "method": result.method,
                            "llm_calls": 0,
                        }
                    # Otherwise, result already contains LLM response
                    auto_meta = result.router_metadata or {}
                    return {
                        "content": result.content,
                        "provider": result.provider or "unknown",
                        "agent_type": agent_type,
                        "task": task,
                        "method": result.method,
                        "llm_calls": 1,
                        "model": auto_meta.get("model"),
                        "fallback_chain": auto_meta.get("fallback_chain"),
                        "was_fallback": auto_meta.get("was_fallback", False),
                        "preferred_provider": auto_meta.get("preferred_provider"),
                        "usage": auto_meta.get("usage"),
                    }
                except Exception as e:
                    logger.warning(f"Autonomous execution failed: {e}, falling back to standard")
                    # Fall through to standard execution
            
            # Standard execution (original logic)
            system_prompts = self._get_agent_prompts(agent_type)
            
            # DEBUG: Log context to verify it's being passed
            logger.info(f"[DEBUG] Agent {agent_type} receiving context: {len(context)} messages")
            if context:
                logger.debug(f"[DEBUG] Context preview: {context[:2] if len(context) >= 2 else context}")
            else:
                logger.warning(f"[DEBUG] Agent {agent_type} has EMPTY context!")
            
            messages = system_prompts + context
            
            if self.router:
                response = await self.router.route_query(
                        message=task,
                        context=messages,
                        preferred_provider=model,
                        images=images,
                    )
            else:
                logger.warning("MultiAIRouter not available, using fallback")
                response = {"response": task, "provider": "fallback"}
            
            router_metadata = response.get("metadata", {})
            return {
                "content": response.get("response", ""),
                "provider": response.get("provider", "unknown"),
                "agent_type": agent_type,
                "task": task,
                "model": router_metadata.get("model"),
                "fallback_chain": router_metadata.get("fallback_chain"),
                "was_fallback": router_metadata.get("was_fallback", False),
                "preferred_provider": router_metadata.get("preferred_provider"),
                "usage": router_metadata.get("usage"),
            }
            
        except Exception as e:
            logger.error(f"Error spawning agent: {e}", exc_info=True)
            return {
                "content": "",
                "provider": "error",
                "agent_type": agent_type,
                "error": str(e)
            }
    
    def _get_agent_prompts(self, agent_type: str) -> List[Dict[str, str]]:
        """Get system prompts for different agent types."""
        prompts = {
            "reasoning": [
                {
                    "role": "system",
                    "content": """You are a reasoning agent. The conversation history is in the messages above.

STRICT RULES:
1. NEVER paraphrase or echo back what the user said. Do NOT start with "It sounds like...", "It seems like...", "You're concerned about...", "You mentioned...". Go straight to your answer.
2. NEVER give generic advice bullet lists. Be specific and concrete.
3. NEVER end with "Would you like to discuss further?" or "How would you like to proceed?"
4. Do NOT act like a therapist or counselor. Talk like a smart colleague.
5. Output ONLY your answer. No meta-commentary ("Based on", "To address", "Here is").
6. Use conversation history naturally — don't announce you're using it.
7. Be direct, concise, and genuinely helpful."""
                }
            ],
            "code": [
                {
                    "role": "system",
                    "content": """You are a code generation agent with FULL conversation context.

CRITICAL CONTEXT RULES:
1. Reference previous code: "Building on the function we created..."
2. Remember user's tech stack and preferences from conversation
3. Don't repeat code already provided - extend or improve it

RESPONSE RULES:
1. Output ONLY the code with minimal comments
2. NO meta-instructions ("Based on", "To address", "Here is")
3. NO explanations unless explicitly requested
4. Be direct - just provide the code

Generate clean, efficient, production-ready code."""
                }
            ],
            "research": [
                {
                    "role": "system",
                    "content": """You are a research agent with FULL conversation context.

CRITICAL CONTEXT RULES:
1. Reference previous research: "Earlier we found that..."
2. Build on previous findings - don't repeat them
3. Remember user's research goals from conversation

RESPONSE RULES:
1. Output ONLY your findings in structured format
2. NO meta-instructions ("Based on", "To address", "Here is")
3. Be direct and factual

Gather and synthesize information efficiently."""
                }
            ],
            "summary": [
                {
                    "role": "system",
                    "content": """You are a summarization agent with FULL conversation context.

CRITICAL CONTEXT RULES:
1. Summarize the ACTUAL conversation history provided
2. Include key points, decisions, and action items
3. Reference specific topics discussed

RESPONSE RULES:
1. Output ONLY the summary
2. NO meta-instructions ("Based on", "To address", "Here is")
3. Be concise and structured

Create accurate, contextual summaries."""
                }
            ],
            "debug": [
                {
                    "role": "system",
                    "content": """You are a debugging agent with FULL conversation context.

CRITICAL CONTEXT RULES:
1. Reference previous bugs: "Similar to the issue we fixed earlier..."
2. Remember user's codebase from conversation
3. Build on previous debugging sessions

RESPONSE RULES:
1. Output ONLY the solution
2. NO meta-instructions ("Based on", "To address", "Here is")
3. Be direct - identify issue and provide fix

Identify and fix issues systematically."""
                }
            ],
            "planning": [
                {
                    "role": "system",
                    "content": """You are a planning agent with FULL conversation context.

CRITICAL CONTEXT RULES:
1. Reference previous plans: "Building on our earlier roadmap..."
2. Remember user's goals and constraints from conversation
3. Update plans based on progress discussed

RESPONSE RULES:
1. Output ONLY the plan in clear format
2. NO meta-instructions ("Based on", "To address", "Here is")
3. Be structured and actionable

Create contextual, actionable plans."""
                }
            ],
            "math": [
                {
                    "role": "system",
                    "content": """You are a mathematical reasoning agent with FULL conversation context.

CRITICAL CONTEXT RULES:
1. Reference previous calculations from conversation
2. Remember user's mathematical level and notation preferences
3. Build on previous problem-solving approaches

RESPONSE RULES:
1. Show work step-by-step with clear notation
2. NO meta-instructions ("Based on", "To address", "Here is")
3. Verify calculations and provide final answer

Solve problems with precision and clarity."""
                }
            ],
            "security": [
                {
                    "role": "system",
                    "content": """You are a security analysis agent with FULL conversation context.

CRITICAL CONTEXT RULES:
1. Reference previous security reviews from conversation
2. Remember user's codebase and security requirements
3. Build on previous vulnerability findings

RESPONSE RULES:
1. Output ONLY security findings with severity and fixes
2. NO meta-instructions ("Based on", "To address", "Here is")
3. Follow OWASP/CWE best practices

Identify vulnerabilities and provide remediation steps."""
                }
            ],
            "architecture": [
                {
                    "role": "system",
                    "content": """You are a system architecture agent with FULL conversation context.

CRITICAL CONTEXT RULES:
1. Reference previous architecture discussions from conversation
2. Remember user's system requirements and constraints
3. Build on previous design decisions

RESPONSE RULES:
1. Output ONLY architecture decisions with diagrams and rationale
2. NO meta-instructions ("Based on", "To address", "Here is")
3. Use proven design patterns

Design scalable, maintainable systems."""
                }
            ],
            "test": [
                {
                    "role": "system",
                    "content": """You are a test generation agent with FULL conversation context.

CRITICAL CONTEXT RULES:
1. Reference code discussed in conversation
2. Remember user's testing framework preferences
3. Build on previous test coverage

RESPONSE RULES:
1. Output ONLY ready-to-run test code
2. NO meta-instructions ("Based on", "To address", "Here is")
3. Include clear assertions and test names

Create comprehensive test coverage."""
                }
            ],
            "review": [
                {
                    "role": "system",
                    "content": """You are a code review agent with FULL conversation context.

CRITICAL CONTEXT RULES:
1. Reference code discussed in conversation
2. Remember user's coding standards and preferences
3. Build on previous review feedback

RESPONSE RULES:
1. Output ONLY specific issues with fixes
2. NO meta-instructions ("Based on", "To address", "Here is")
3. Be concise with line references

Analyze code for bugs, performance, and best practices."""
                }
            ],
            "explain": [
                {
                    "role": "system",
                    "content": """You are an explanation agent with FULL conversation context.

CRITICAL CONTEXT RULES:
1. Reference topics discussed in conversation
2. Remember user's knowledge level from previous questions
3. Build on previous explanations

RESPONSE RULES:
1. Output ONLY the explanation
2. NO meta-instructions ("Based on", "To address", "Here is")
3. Use simple language, analogies, and examples

Break down complex topics into beginner-friendly explanations."""
                }
            ],
            "optimization": [
                {
                    "role": "system",
                    "content": """You are a performance optimization agent with FULL MEMORY ACCESS.

You have access to:
- Conversation history (previous optimizations, performance data)
- User's codebase and performance requirements
- Previous bottlenecks and solutions

IMPORTANT: You HAVE memory. Reference previous optimizations.

Analyze code for performance bottlenecks, memory leaks, and inefficiencies. Suggest optimizations with benchmarks.
Do NOT make premature optimizations. Focus on measurable improvements with O(n) complexity analysis."""
                }
            ],
            "documentation": [
                {
                    "role": "system",
                    "content": """You are a documentation agent with FULL MEMORY ACCESS.

You have access to:
- Conversation history (previous documentation, code context)
- User's codebase and documentation standards
- Previous documentation patterns

IMPORTANT: You HAVE memory. Reference previous documentation and code.

Generate clear, comprehensive documentation including README files, API docs, JSDoc/docstrings, and usage examples.
Do NOT be verbose. Output well-structured documentation following industry standards (OpenAPI, JSDoc, Google style)."""
                }
            ],
            "migration": [
                {
                    "role": "system",
                    "content": "You are a migration agent. Help with code migrations, version upgrades, framework transitions, and data migrations. Provide step-by-step migration plans."
                },
                {
                    "role": "system",
                    "content": "Do NOT skip compatibility checks. Output migration scripts with rollback strategies."
                }
            ],
            "api": [
                {
                    "role": "system",
                    "content": "You are an API design agent. Design RESTful APIs, GraphQL schemas, and API integrations following best practices (REST, OpenAPI, versioning)."
                },
                {
                    "role": "system",
                    "content": "Do NOT ignore error handling. Output complete API specifications with request/response examples."
                }
            ],
            "database": [
                {
                    "role": "system",
                    "content": "You are a database agent. Design schemas, write optimized queries, handle migrations, and advise on database selection (SQL vs NoSQL)."
                },
                {
                    "role": "system",
                    "content": "Do NOT ignore indexing. Output queries with EXPLAIN analysis and optimization suggestions."
                }
            ],
            "devops": [
                {
                    "role": "system",
                    "content": "You are a DevOps agent. Help with CI/CD pipelines, Docker, Kubernetes, cloud deployments, and infrastructure as code (Terraform, Ansible)."
                },
                {
                    "role": "system",
                    "content": "Do NOT ignore security. Output production-ready configurations with proper secrets management."
                }
            ],
            "refactor": [
                {
                    "role": "system",
                    "content": "You are a refactoring agent. Apply design patterns (SOLID, DRY, KISS) to improve code structure. Preserve functionality while enhancing readability and maintainability."
                },
                {
                    "role": "system",
                    "content": "Do NOT change behavior. Output refactored code with before/after comparisons and explain the patterns applied."
                }
            ],
            "accessibility": [
                {
                    "role": "system",
                    "content": "You are an accessibility (a11y) agent. Ensure WCAG 2.1 AA/AAA compliance, proper ARIA attributes, keyboard navigation, screen reader support, and color contrast."
                },
                {
                    "role": "system",
                    "content": "Do NOT ignore semantic HTML. Output accessible code with specific WCAG criteria references (e.g., WCAG 2.1.1)."
                }
            ],
            "i18n": [
                {
                    "role": "system",
                    "content": "You are an internationalization (i18n) agent. Help with translations, locale handling, RTL support, date/number formatting, and i18n library setup (react-intl, i18next)."
                },
                {
                    "role": "system",
                    "content": "Do NOT hardcode strings. Output translation keys, locale files, and proper pluralization rules."
                }
            ],
            "regex": [
                {
                    "role": "system",
                    "content": "You are a regex expert agent. Create, explain, and debug regular expressions. Support multiple regex flavors (JS, Python, PCRE)."
                },
                {
                    "role": "system",
                    "content": "Do NOT create overly complex patterns. Output regex with test cases, explanations, and edge case handling."
                }
            ],
            "git": [
                {
                    "role": "system",
                    "content": "You are a Git expert agent. Help with branching strategies, merge conflicts, rebasing, cherry-picking, git hooks, and repository management."
                },
                {
                    "role": "system",
                    "content": "Do NOT suggest destructive operations without warnings. Output git commands with explanations and recovery options."
                }
            ],
            "css": [
                {
                    "role": "system",
                    "content": "You are a CSS/styling expert agent. Help with layouts (flexbox, grid), responsive design, animations, CSS-in-JS, Tailwind, and cross-browser compatibility."
                },
                {
                    "role": "system",
                    "content": "Do NOT use deprecated properties. Output modern CSS with fallbacks and browser support notes."
                }
            ]
        }
        
        return prompts.get(agent_type, prompts["reasoning"])
    
    def should_spawn_agent(self, message: str) -> Optional[str]:
        """Determine if an agent should be spawned based on message content.
        
        ALWAYS returns an agent - never returns None to ensure quality responses.
        
        If adaptive allocation is enabled, uses intelligent scoring.
        Otherwise, falls back to keyword matching.
        """
        # Use adaptive allocation if enabled
        if self.use_adaptive_allocation and self.allocator:
            try:
                agent_type, score = self.allocator.select_best_agent(message)
                logger.info(
                    f"[Adaptive] Selected {agent_type} "
                    f"(score: {score.total_score:.3f}, "
                    f"success: {score.success_rate_score:.3f})"
                )
                return agent_type
            except Exception as e:
                logger.warning(f"Adaptive allocation failed: {e}, falling back to keyword matching")
                # Fall through to keyword matching
        
        # Keyword-based selection (original logic)
        message_lower = message.lower()
        
        # Math agent - highest priority for calculations
        if any(word in message_lower for word in ["calculate", "math", "equation", "solve", "formula", "derivative", "integral", "algebra", "geometry"]):
            return "math"
        
        # Security agent - security-related queries
        if any(word in message_lower for word in ["security", "vulnerability", "hack", "exploit", "injection", "xss", "csrf", "authentication", "authorization", "encrypt"]):
            return "security"
        
        # Architecture agent - system design
        if any(word in message_lower for word in ["architecture", "design pattern", "structure", "scalable", "microservice", "monolith", "database design", "api design"]):
            return "architecture"
        
        # Test agent - testing related
        if any(word in message_lower for word in ["test", "unit test", "coverage", "jest", "pytest", "mock", "assertion", "test case"]):
            return "test"
        
        # Refactor agent - code refactoring (check before review)
        if any(word in message_lower for word in ["refactor", "restructure", "clean up", "reorganize", "simplify code", "design pattern"]):
            return "refactor"
        
        # Review agent - code review
        if any(word in message_lower for word in ["review", "critique", "feedback", "improve", "code quality", "best practice"]):
            return "review"
        
        # Explain agent - simplified explanations
        if any(word in message_lower for word in ["eli5", "simple terms", "beginner", "explain like", "what is", "basics of", "introduction to"]):
            return "explain"
        
        # Debug agent - fixing issues
        if any(word in message_lower for word in ["fix", "debug", "error", "bug", "issue", "problem", "broken", "not working", "fails"]):
            return "debug"
        
        # Code agent - code generation
        if any(word in message_lower for word in ["write code", "generate code", "create function", "implement", "code for", "script", "program"]):
            return "code"
        
        # Research agent - information gathering
        if any(word in message_lower for word in ["research", "find information", "look up", "investigate", "compare", "difference between"]):
            return "research"
        
        # Summary agent - condensing information
        if any(word in message_lower for word in ["summarize", "summary", "brief overview", "tl;dr", "key points"]):
            return "summary"
        
        # Planning agent - strategic planning
        if any(word in message_lower for word in ["plan", "strategy", "roadmap", "steps to", "how to achieve", "project plan"]):
            return "planning"
        
        # Optimization agent - performance
        if any(word in message_lower for word in ["optimize", "performance", "speed up", "memory", "bottleneck", "slow", "faster", "efficient"]):
            return "optimization"
        
        # Documentation agent - docs generation
        if any(word in message_lower for word in ["document", "readme", "jsdoc", "docstring", "api doc", "documentation", "comments"]):
            return "documentation"
        
        # Migration agent - upgrades and transitions
        if any(word in message_lower for word in ["migrate", "upgrade", "convert", "port", "transition", "move to", "switch to"]):
            return "migration"
        
        # API agent - API design
        if any(word in message_lower for word in ["api", "endpoint", "rest", "graphql", "webhook", "route", "http"]):
            return "api"
        
        # Database agent - database operations
        if any(word in message_lower for word in ["database", "sql", "query", "schema", "table", "mongodb", "postgres", "mysql", "redis"]):
            return "database"
        
        # DevOps agent - deployment and infrastructure
        if any(word in message_lower for word in ["deploy", "ci/cd", "docker", "kubernetes", "k8s", "terraform", "aws", "cloud", "pipeline"]):
            return "devops"
        
        # Accessibility agent - a11y compliance
        if any(word in message_lower for word in ["accessibility", "a11y", "wcag", "aria", "screen reader", "keyboard navigation", "color contrast"]):
            return "accessibility"
        
        # i18n agent - internationalization
        if any(word in message_lower for word in ["translate", "i18n", "localize", "internationalization", "locale", "rtl", "language support"]):
            return "i18n"
        
        # Regex agent - regular expressions
        if any(word in message_lower for word in ["regex", "regular expression", "pattern match", "regexp", "match pattern"]):
            return "regex"
        
        # Git agent - version control
        if any(word in message_lower for word in ["git", "merge conflict", "rebase", "cherry-pick", "branch", "commit", "pull request"]):
            return "git"
        
        # CSS agent - styling
        if any(word in message_lower for word in ["css", "style", "flexbox", "grid layout", "responsive", "animation", "tailwind", "scss"]):
            return "css"
        
        # Reasoning agent - analysis and deep thinking (default fallback)
        if any(word in message_lower for word in ["analyze", "analysis", "explain why", "how does", "what causes", "reason", "think"]):
            return "reasoning"
        
        # DEFAULT: Always use reasoning agent instead of direct LLM
        return "reasoning"


# Global instance (router will be set later)
agent_engine = AgentEngine()
