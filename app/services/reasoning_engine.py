"""
Reasoning Engine
================

Phase 4 of Agent Autonomy Enhancement - True Reasoning Capabilities.

Implements hard decision logic, premise validation, self-invalidation,
temporal adaptation, and epistemic boundaries.

This transforms the system from a "well-spoken assistant" to a "decision engine".

Author: Resonant Chat Systems Team
Date: December 27, 2025
"""
from __future__ import annotations

import logging
from typing import Dict, Any, Optional, List, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class SelectionRule(Enum):
    """Selection rules for conflicting objectives."""
    LEXICOGRAPHIC = "lexicographic"  # Prioritize by order
    DOMINANCE = "dominance"  # Choose if dominates on most objectives
    VETO = "veto"  # Reject if fails critical threshold
    SATISFICING = "satisficing"  # First to meet all thresholds
    MAXIMIN = "maximin"  # Maximize worst-case outcome


@dataclass
class Objective:
    """A single objective in a decision problem."""
    name: str
    description: str
    current_value: float
    target_value: float
    weight: float = 1.0
    is_critical: bool = False


@dataclass
class Solution:
    """A solution with its impact on objectives."""
    name: str
    description: str
    objective_impacts: Dict[str, float]  # objective_name -> change
    
    def improves(self, objective_name: str) -> bool:
        """Check if solution improves objective."""
        return self.objective_impacts.get(objective_name, 0) > 0
    
    def worsens(self, objective_name: str) -> bool:
        """Check if solution worsens objective."""
        return self.objective_impacts.get(objective_name, 0) < 0


@dataclass
class TradeOffDecision:
    """A decision with explicit trade-offs."""
    chosen_solution: Solution
    selection_rule: SelectionRule
    improved_objectives: List[str]
    worsened_objectives: List[str]
    sacrificed_objective: str
    justification: str
    trade_off_structure: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chosen_solution": self.chosen_solution.name,
            "selection_rule": self.selection_rule.value,
            "improved_objectives": self.improved_objectives,
            "worsened_objectives": self.worsened_objectives,
            "sacrificed_objective": self.sacrificed_objective,
            "justification": self.justification,
            "trade_off_structure": self.trade_off_structure,
        }


@dataclass
class Premise:
    """A premise used in reasoning."""
    id: str
    statement: str
    is_validated: bool = False
    is_true: Optional[bool] = None
    source: str = "input"
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DependencyChain:
    """Tracks dependencies between premises and conclusions."""
    conclusion: str
    premises: List[str]  # premise IDs
    timestamp: datetime = field(default_factory=datetime.now)
    is_valid: bool = True


@dataclass
class Assumption:
    """An assumption made at a point in time."""
    id: str
    statement: str
    timestamp: datetime
    is_valid: bool = True
    invalidated_at: Optional[datetime] = None
    invalidation_reason: Optional[str] = None


@dataclass
class TemporalDecision:
    """A decision that can be revised over time."""
    decision_id: str
    decision: str
    timestamp: datetime
    assumptions: List[Assumption]
    is_current: bool = True
    revision_of: Optional[str] = None
    revision_reason: Optional[str] = None


@dataclass
class EpistemicBoundary:
    """Defines what can and cannot be known."""
    context: str
    can_be_known: List[str]
    cannot_be_known: List[str]
    requires_additional_info: List[str]
    confidence_level: float


class ReasoningEngine:
    """
    Core reasoning engine with hard logic.
    
    Capabilities:
    1. Hard decision logic (Test 1)
    2. Premise validation (Test 3)
    3. Self-invalidation (Test 6)
    4. Temporal adaptation (Test 2)
    5. Epistemic boundaries (Test 5)
    """
    
    def __init__(self):
        self.premises: Dict[str, Premise] = {}
        self.dependency_chains: List[DependencyChain] = []
        self.assumptions: Dict[str, Assumption] = {}
        self.decisions: Dict[str, TemporalDecision] = {}
        
        logger.info("ReasoningEngine initialized")
    
    # ============================================================
    # TEST 1: CONFLICTING OBJECTIVES - HARD DECISION LOGIC
    # ============================================================
    
    def decide_with_trade_offs(
        self,
        objectives: List[Objective],
        solutions: List[Solution],
        selection_rule: SelectionRule,
        context: Optional[Dict[str, Any]] = None
    ) -> TradeOffDecision:
        """
        Make a decision with explicit trade-offs.
        
        NO balanced approach. NO utility collapse.
        Explicit statement of what is sacrificed and why.
        """
        if not solutions:
            raise ValueError("No solutions provided")
        
        if not objectives:
            raise ValueError("No objectives provided")
        
        # Apply selection rule
        if selection_rule == SelectionRule.LEXICOGRAPHIC:
            chosen = self._lexicographic_selection(objectives, solutions)
        elif selection_rule == SelectionRule.DOMINANCE:
            chosen = self._dominance_selection(objectives, solutions)
        elif selection_rule == SelectionRule.VETO:
            chosen = self._veto_selection(objectives, solutions)
        elif selection_rule == SelectionRule.SATISFICING:
            chosen = self._satisficing_selection(objectives, solutions)
        elif selection_rule == SelectionRule.MAXIMIN:
            chosen = self._maximin_selection(objectives, solutions)
        else:
            raise ValueError(f"Unknown selection rule: {selection_rule}")
        
        # Analyze impacts
        improved = []
        worsened = []
        
        for obj in objectives:
            if chosen.improves(obj.name):
                improved.append(obj.name)
            elif chosen.worsens(obj.name):
                worsened.append(obj.name)
        
        # Identify sacrificed objective (most worsened)
        sacrificed = None
        max_worsening = 0
        for obj_name in worsened:
            worsening = abs(chosen.objective_impacts[obj_name])
            if worsening > max_worsening:
                max_worsening = worsening
                sacrificed = obj_name
        
        if not sacrificed and worsened:
            sacrificed = worsened[0]
        
        # Build trade-off structure
        trade_off_structure = self._build_trade_off_structure(
            objectives, chosen, improved, worsened, sacrificed
        )
        
        # Build justification
        justification = self._build_justification(
            selection_rule, improved, worsened, sacrificed, chosen
        )
        
        decision = TradeOffDecision(
            chosen_solution=chosen,
            selection_rule=selection_rule,
            improved_objectives=improved,
            worsened_objectives=worsened,
            sacrificed_objective=sacrificed or "none",
            justification=justification,
            trade_off_structure=trade_off_structure,
        )
        
        logger.info(
            f"Decision made: {chosen.name} "
            f"(rule: {selection_rule.value}, sacrificed: {sacrificed})"
        )
        
        return decision
    
    def _lexicographic_selection(
        self,
        objectives: List[Objective],
        solutions: List[Solution]
    ) -> Solution:
        """Select by objective priority order."""
        for obj in objectives:
            best_solution = None
            best_impact = float('-inf')
            
            for solution in solutions:
                impact = solution.objective_impacts.get(obj.name, 0)
                if impact > best_impact:
                    best_impact = impact
                    best_solution = solution
            
            if best_solution and best_impact > 0:
                return best_solution
        
        return solutions[0]  # Fallback
    
    def _dominance_selection(
        self,
        objectives: List[Objective],
        solutions: List[Solution]
    ) -> Solution:
        """Select solution that dominates on most objectives."""
        best_solution = None
        max_improvements = 0
        
        for solution in solutions:
            improvements = sum(
                1 for obj in objectives
                if solution.improves(obj.name)
            )
            
            if improvements > max_improvements:
                max_improvements = improvements
                best_solution = solution
        
        return best_solution or solutions[0]
    
    def _veto_selection(
        self,
        objectives: List[Objective],
        solutions: List[Solution]
    ) -> Solution:
        """Select first solution that doesn't violate critical thresholds."""
        for solution in solutions:
            violates_critical = False
            
            for obj in objectives:
                if obj.is_critical:
                    impact = solution.objective_impacts.get(obj.name, 0)
                    new_value = obj.current_value + impact
                    
                    if new_value < obj.target_value * 0.8:  # 80% threshold
                        violates_critical = True
                        break
            
            if not violates_critical:
                return solution
        
        return solutions[0]  # Fallback
    
    def _satisficing_selection(
        self,
        objectives: List[Objective],
        solutions: List[Solution]
    ) -> Solution:
        """Select first solution that meets all thresholds."""
        for solution in solutions:
            meets_all = True
            
            for obj in objectives:
                impact = solution.objective_impacts.get(obj.name, 0)
                new_value = obj.current_value + impact
                
                if new_value < obj.target_value:
                    meets_all = False
                    break
            
            if meets_all:
                return solution
        
        return solutions[0]  # Fallback
    
    def _maximin_selection(
        self,
        objectives: List[Objective],
        solutions: List[Solution]
    ) -> Solution:
        """Select solution that maximizes worst-case outcome."""
        best_solution = None
        best_worst_case = float('-inf')
        
        for solution in solutions:
            worst_case = float('inf')
            
            for obj in objectives:
                impact = solution.objective_impacts.get(obj.name, 0)
                new_value = obj.current_value + impact
                worst_case = min(worst_case, new_value)
            
            if worst_case > best_worst_case:
                best_worst_case = worst_case
                best_solution = solution
        
        return best_solution or solutions[0]
    
    def _build_trade_off_structure(
        self,
        objectives: List[Objective],
        solution: Solution,
        improved: List[str],
        worsened: List[str],
        sacrificed: Optional[str]
    ) -> str:
        """Build explicit trade-off structure."""
        structure = f"Solution '{solution.name}':\n"
        
        if improved:
            structure += f"  Improves: {', '.join(improved)}\n"
        
        if worsened:
            structure += f"  Worsens: {', '.join(worsened)}\n"
        
        if sacrificed:
            structure += f"  Sacrificed: {sacrificed}\n"
        
        # Show magnitudes
        structure += "\nImpact magnitudes:\n"
        for obj_name, impact in solution.objective_impacts.items():
            direction = "↑" if impact > 0 else "↓"
            structure += f"  {obj_name}: {direction} {abs(impact):.2f}\n"
        
        return structure
    
    def _build_justification(
        self,
        rule: SelectionRule,
        improved: List[str],
        worsened: List[str],
        sacrificed: Optional[str],
        solution: Solution
    ) -> str:
        """Build justification for decision."""
        justification = f"Using {rule.value} selection rule:\n"
        
        if rule == SelectionRule.LEXICOGRAPHIC:
            justification += f"Prioritized objectives in order. "
            justification += f"'{solution.name}' best improves highest-priority objectives.\n"
        
        elif rule == SelectionRule.DOMINANCE:
            justification += f"'{solution.name}' dominates on {len(improved)} objectives.\n"
        
        elif rule == SelectionRule.VETO:
            justification += f"'{solution.name}' does not violate critical thresholds.\n"
        
        elif rule == SelectionRule.SATISFICING:
            justification += f"'{solution.name}' first to meet all thresholds.\n"
        
        elif rule == SelectionRule.MAXIMIN:
            justification += f"'{solution.name}' maximizes worst-case outcome.\n"
        
        if sacrificed:
            justification += f"\nExplicit sacrifice: '{sacrificed}' is worsened to improve {', '.join(improved)}.\n"
            justification += f"This trade-off is accepted because the gains in {improved[0]} "
            justification += f"outweigh the loss in {sacrificed} under the {rule.value} rule."
        
        return justification
    
    # ============================================================
    # TEST 3: PREMISE VALIDATION
    # ============================================================
    
    def add_premise(self, statement: str, source: str = "input") -> str:
        """Add a premise for validation."""
        premise_id = f"premise_{len(self.premises)}"
        premise = Premise(
            id=premise_id,
            statement=statement,
            source=source,
        )
        self.premises[premise_id] = premise
        
        logger.debug(f"Added premise: {premise_id}")
        return premise_id
    
    def validate_premise(self, premise_id: str, is_true: bool) -> bool:
        """Validate a premise."""
        premise = self.premises.get(premise_id)
        if not premise:
            logger.warning(f"Premise not found: {premise_id}")
            return False
        
        premise.is_validated = True
        premise.is_true = is_true
        
        # Invalidate dependent chains if premise is false
        if not is_true:
            self._invalidate_dependent_chains(premise_id)
        
        logger.info(f"Premise {premise_id} validated: {is_true}")
        return True
    
    def add_dependency_chain(
        self,
        conclusion: str,
        premise_ids: List[str]
    ) -> DependencyChain:
        """Track dependency between premises and conclusion."""
        chain = DependencyChain(
            conclusion=conclusion,
            premises=premise_ids,
        )
        self.dependency_chains.append(chain)
        
        logger.debug(f"Added dependency chain: {conclusion} depends on {premise_ids}")
        return chain
    
    def _invalidate_dependent_chains(self, premise_id: str):
        """Invalidate all chains depending on a false premise."""
        for chain in self.dependency_chains:
            if premise_id in chain.premises:
                chain.is_valid = False
                logger.warning(
                    f"Invalidated chain: '{chain.conclusion}' "
                    f"(depends on false premise {premise_id})"
                )
    
    def retract_inference(self, conclusion: str, reason: str) -> List[str]:
        """Retract an inference and mark dependency chain."""
        retracted_chains = []
        
        for chain in self.dependency_chains:
            if chain.conclusion == conclusion:
                chain.is_valid = False
                retracted_chains.append(chain.conclusion)
                
                logger.info(
                    f"Retracted: '{conclusion}' "
                    f"(reason: {reason}, premises: {chain.premises})"
                )
        
        return retracted_chains
    
    # ============================================================
    # TEST 6: SELF-INVALIDATION
    # ============================================================
    
    def generate_falsifiers(
        self,
        claim: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Generate conditions that would invalidate a claim."""
        falsifiers = []
        
        # Generic falsifiers based on claim structure
        if "all" in claim.lower():
            falsifiers.append("Finding a single counterexample")
        
        if "never" in claim.lower():
            falsifiers.append("Observing the event occurring once")
        
        if "always" in claim.lower():
            falsifiers.append("Identifying a case where it doesn't hold")
        
        # Context-specific falsifiers
        if context:
            if "assumption" in context:
                falsifiers.append(f"Invalidation of assumption: {context['assumption']}")
            
            if "data_source" in context:
                falsifiers.append(f"Corruption of data source: {context['data_source']}")
        
        # Logical falsifiers
        falsifiers.extend([
            "Discovery of contradictory evidence",
            "Identification of flawed reasoning in derivation",
            "Change in underlying constraints or axioms",
        ])
        
        logger.debug(f"Generated {len(falsifiers)} falsifiers for claim")
        return falsifiers
    
    def check_invalidation_conditions(
        self,
        claim: str,
        new_evidence: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """Check if new evidence invalidates a claim."""
        # Check for direct contradiction
        if "contradicts" in new_evidence:
            return True, f"Direct contradiction: {new_evidence['contradicts']}"
        
        # Check for assumption invalidation
        if "invalidated_assumption" in new_evidence:
            return True, f"Assumption invalidated: {new_evidence['invalidated_assumption']}"
        
        # Check for constraint change
        if "new_constraint" in new_evidence:
            return True, f"New constraint: {new_evidence['new_constraint']}"
        
        return False, None
    
    # ============================================================
    # TEST 2: TEMPORAL ADAPTATION
    # ============================================================
    
    def make_temporal_decision(
        self,
        decision: str,
        assumptions: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Make a decision with tracked assumptions."""
        decision_id = f"decision_{len(self.decisions)}"
        
        assumption_objs = [
            Assumption(
                id=f"assumption_{i}",
                statement=stmt,
                timestamp=datetime.now(),
            )
            for i, stmt in enumerate(assumptions)
        ]
        
        temporal_decision = TemporalDecision(
            decision_id=decision_id,
            decision=decision,
            timestamp=datetime.now(),
            assumptions=assumption_objs,
        )
        
        self.decisions[decision_id] = temporal_decision
        
        # Track assumptions
        for assumption in assumption_objs:
            self.assumptions[assumption.id] = assumption
        
        logger.info(f"Made temporal decision: {decision_id}")
        return decision_id
    
    def invalidate_assumption(
        self,
        assumption_id: str,
        reason: str
    ) -> List[str]:
        """Invalidate an assumption and affected decisions."""
        assumption = self.assumptions.get(assumption_id)
        if not assumption:
            logger.warning(f"Assumption not found: {assumption_id}")
            return []
        
        assumption.is_valid = False
        assumption.invalidated_at = datetime.now()
        assumption.invalidation_reason = reason
        
        # Find affected decisions
        affected_decisions = []
        for decision in self.decisions.values():
            for assum in decision.assumptions:
                if assum.id == assumption_id:
                    decision.is_current = False
                    affected_decisions.append(decision.decision_id)
        
        logger.warning(
            f"Invalidated assumption: {assumption_id} "
            f"(affects {len(affected_decisions)} decisions)"
        )
        
        return affected_decisions
    
    def revise_decision(
        self,
        original_decision_id: str,
        new_decision: str,
        revision_reason: str,
        new_assumptions: List[str]
    ) -> str:
        """Revise a decision when assumptions change."""
        original = self.decisions.get(original_decision_id)
        if not original:
            raise ValueError(f"Original decision not found: {original_decision_id}")
        
        # Mark original as not current
        original.is_current = False
        
        # Create revised decision
        new_decision_id = f"decision_{len(self.decisions)}"
        
        assumption_objs = [
            Assumption(
                id=f"assumption_{len(self.assumptions) + i}",
                statement=stmt,
                timestamp=datetime.now(),
            )
            for i, stmt in enumerate(new_assumptions)
        ]
        
        revised_decision = TemporalDecision(
            decision_id=new_decision_id,
            decision=new_decision,
            timestamp=datetime.now(),
            assumptions=assumption_objs,
            revision_of=original_decision_id,
            revision_reason=revision_reason,
        )
        
        self.decisions[new_decision_id] = revised_decision
        
        # Track new assumptions
        for assumption in assumption_objs:
            self.assumptions[assumption.id] = assumption
        
        logger.info(
            f"Revised decision: {original_decision_id} → {new_decision_id} "
            f"(reason: {revision_reason})"
        )
        
        return new_decision_id
    
    # ============================================================
    # TEST 5: EPISTEMIC BOUNDARIES
    # ============================================================
    
    def define_epistemic_boundary(
        self,
        context: str,
        available_info: List[str]
    ) -> EpistemicBoundary:
        """Define what can and cannot be known from given information."""
        can_be_known = []
        cannot_be_known = []
        requires_additional = []
        
        # Analyze what's directly available
        for info in available_info:
            can_be_known.append(f"Direct fact: {info}")
        
        # Define clear boundaries
        cannot_be_known.extend([
            "Future events not causally determined by available info",
            "Hidden variables not measured or observable",
            "Counterfactuals (what would have happened)",
            "Internal states of external agents",
            "Information outside the scope of provided data",
        ])
        
        # What requires more info
        requires_additional.extend([
            "Causal relationships (requires experimental data)",
            "Generalizations (requires larger sample)",
            "Predictions (requires temporal data)",
        ])
        
        boundary = EpistemicBoundary(
            context=context,
            can_be_known=can_be_known,
            cannot_be_known=cannot_be_known,
            requires_additional_info=requires_additional,
            confidence_level=0.9,
        )
        
        logger.debug(f"Defined epistemic boundary for: {context}")
        return boundary
    
    def check_inference_validity(
        self,
        inference: str,
        boundary: EpistemicBoundary
    ) -> Tuple[bool, str]:
        """Check if an inference is valid given epistemic boundary."""
        # Check if inference requires unknowable information
        for unknowable in boundary.cannot_be_known:
            if any(keyword in inference.lower() for keyword in unknowable.lower().split()):
                return False, f"Inference requires unknowable: {unknowable}"
        
        # Check if inference requires additional information
        for additional in boundary.requires_additional_info:
            if any(keyword in inference.lower() for keyword in additional.lower().split()):
                return False, f"Inference requires additional info: {additional}"
        
        return True, "Inference is within epistemic boundary"


# Global reasoning engine instance
_reasoning_engine: Optional[ReasoningEngine] = None


def get_reasoning_engine() -> ReasoningEngine:
    """Get global reasoning engine instance."""
    global _reasoning_engine
    if _reasoning_engine is None:
        _reasoning_engine = ReasoningEngine()
    return _reasoning_engine
