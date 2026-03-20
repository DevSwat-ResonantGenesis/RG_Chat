"""
Agent Voting System (AVS)
==========================

Phase 5.3: Multiple agents vote on the best solution for complex tasks.

Features:
- Multiple agents generate solutions
- Agents vote on each other's solutions
- Weighted voting based on agent expertise
- Consensus-based final answer
"""
from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class VotingCandidate:
    """A candidate solution from an agent."""
    agent_type: str
    content: str
    votes: int = 0
    vote_reasons: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class VotingResult:
    """Result of a voting session."""
    winner: VotingCandidate
    all_candidates: List[VotingCandidate]
    total_votes: int
    consensus_score: float  # 0-1, how much agreement
    voting_summary: str
    execution_time_ms: float


class AgentVotingEngine:
    """
    Orchestrates multi-agent voting on solutions.
    """
    
    def __init__(self, agent_engine=None):
        self.agent_engine = agent_engine
        
        # Agent expertise weights for voting
        self.expertise_weights = {
            "reasoning": 1.2,
            "review": 1.3,
            "architecture": 1.2,
            "security": 1.1,
            "test": 1.0,
            "code": 1.0,
            "debug": 1.0,
            "explain": 0.9,
            "research": 1.1,
        }
    
    def set_agent_engine(self, agent_engine):
        """Set the agent engine for spawning agents."""
        self.agent_engine = agent_engine
    
    async def run_voting(
        self,
        task: str,
        context: List[Dict[str, Any]],
        candidate_agents: List[str] = None,
        voter_agents: List[str] = None,
        preferred_provider: Optional[str] = None,
    ) -> VotingResult:
        """Run a voting session on a task."""
        start_time = datetime.now()
        
        if not self.agent_engine:
            raise RuntimeError("Agent engine not set")
        
        # Default candidate agents
        if not candidate_agents:
            candidate_agents = ["reasoning", "code", "architecture"]
        
        # Default voter agents (different from candidates for objectivity)
        if not voter_agents:
            voter_agents = ["review", "test", "security"]
        
        logger.info(f"🗳️ Starting voting: {len(candidate_agents)} candidates, {len(voter_agents)} voters")
        
        # Step 1: Generate candidate solutions
        candidates = await self._generate_candidates(
            task, context, candidate_agents, preferred_provider
        )
        
        if not candidates:
            raise RuntimeError("No candidates generated")
        
        # Step 2: Have voters evaluate and vote
        await self._collect_votes(
            task, context, candidates, voter_agents, preferred_provider
        )
        
        # Step 3: Determine winner
        candidates.sort(key=lambda c: c.votes, reverse=True)
        winner = candidates[0]
        
        # Calculate consensus score
        total_votes = sum(c.votes for c in candidates)
        consensus_score = winner.votes / total_votes if total_votes > 0 else 0
        
        # Generate voting summary
        voting_summary = self._generate_summary(candidates, winner)
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        logger.info(f"✅ Voting complete: {winner.agent_type} won with {winner.votes} votes")
        
        return VotingResult(
            winner=winner,
            all_candidates=candidates,
            total_votes=total_votes,
            consensus_score=consensus_score,
            voting_summary=voting_summary,
            execution_time_ms=execution_time,
        )
    
    async def _generate_candidates(
        self,
        task: str,
        context: List[Dict[str, Any]],
        candidate_agents: List[str],
        preferred_provider: Optional[str] = None,
    ) -> List[VotingCandidate]:
        """Generate candidate solutions from multiple agents."""
        candidates = []
        
        async def generate_one(agent_type: str) -> Optional[VotingCandidate]:
            try:
                result = await self.agent_engine.spawn(
                    task=task,
                    context=context,
                    agent_type=agent_type,
                    model=preferred_provider,
                )
                content = result.get("content", "")
                if content:
                    return VotingCandidate(
                        agent_type=agent_type,
                        content=content,
                        confidence=result.get("confidence", 0.7),
                    )
            except Exception as e:
                logger.error(f"Candidate {agent_type} failed: {e}")
            return None
        
        # Generate all candidates in parallel
        tasks = [generate_one(agent) for agent in candidate_agents]
        results = await asyncio.gather(*tasks)
        
        for result in results:
            if result:
                candidates.append(result)
        
        return candidates
    
    async def _collect_votes(
        self,
        task: str,
        context: List[Dict[str, Any]],
        candidates: List[VotingCandidate],
        voter_agents: List[str],
        preferred_provider: Optional[str] = None,
    ):
        """Have voter agents evaluate and vote on candidates."""
        
        # Build voting prompt
        candidates_text = "\n\n".join([
            f"=== SOLUTION {i+1} (from {c.agent_type}) ===\n{c.content[:1500]}"
            for i, c in enumerate(candidates)
        ])
        
        voting_prompt = f"""You are evaluating multiple solutions to this task:

TASK: {task}

{candidates_text}

Vote for the BEST solution. Consider:
1. Correctness - Is it technically accurate?
2. Completeness - Does it fully address the task?
3. Quality - Is it well-structured and clear?
4. Practicality - Is it implementable?

Respond with ONLY:
VOTE: [solution number 1-{len(candidates)}]
REASON: [one sentence explanation]"""

        async def vote_one(voter_agent: str) -> Tuple[int, str]:
            try:
                result = await self.agent_engine.spawn(
                    task=voting_prompt,
                    context=[],
                    agent_type=voter_agent,
                    model=preferred_provider,
                )
                response = result.get("content", "")
                
                # Parse vote
                vote_num = self._parse_vote(response, len(candidates))
                reason = self._parse_reason(response)
                
                return vote_num, reason
            except Exception as e:
                logger.error(f"Voter {voter_agent} failed: {e}")
                return 0, ""
        
        # Collect all votes in parallel
        tasks = [vote_one(voter) for voter in voter_agents]
        results = await asyncio.gather(*tasks)
        
        # Apply votes with expertise weights
        for i, (voter_agent, (vote_num, reason)) in enumerate(zip(voter_agents, results)):
            if 1 <= vote_num <= len(candidates):
                weight = self.expertise_weights.get(voter_agent, 1.0)
                candidates[vote_num - 1].votes += int(weight * 10)  # Scale to integers
                if reason:
                    candidates[vote_num - 1].vote_reasons.append(f"{voter_agent}: {reason}")
    
    def _parse_vote(self, response: str, max_candidates: int) -> int:
        """Parse vote number from response."""
        import re
        
        # Look for "VOTE: X" pattern
        match = re.search(r'VOTE:\s*(\d+)', response, re.IGNORECASE)
        if match:
            vote = int(match.group(1))
            if 1 <= vote <= max_candidates:
                return vote
        
        # Fallback: look for any number
        numbers = re.findall(r'\b([1-9])\b', response)
        for num in numbers:
            vote = int(num)
            if 1 <= vote <= max_candidates:
                return vote
        
        return 0
    
    def _parse_reason(self, response: str) -> str:
        """Parse reason from response."""
        import re
        
        match = re.search(r'REASON:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if match:
            return match.group(1).strip()[:200]
        
        return ""
    
    def _generate_summary(
        self,
        candidates: List[VotingCandidate],
        winner: VotingCandidate,
    ) -> str:
        """Generate a summary of the voting results."""
        lines = [
            f"**Voting Results**",
            f"",
            f"Winner: {winner.agent_type} ({winner.votes} votes)",
            f"",
            "All candidates:",
        ]
        
        for i, c in enumerate(candidates):
            status = "🏆" if c == winner else "  "
            lines.append(f"{status} {i+1}. {c.agent_type}: {c.votes} votes")
            if c.vote_reasons:
                for reason in c.vote_reasons[:2]:
                    lines.append(f"      - {reason}")
        
        return "\n".join(lines)


# Global instance
agent_voting = AgentVotingEngine()
