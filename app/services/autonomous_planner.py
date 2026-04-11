"""
Autonomous Planner - Creates and manages task plans for complex requests.

STATUS: GRADUATED
CREATED: 2025-12-21
GRADUATED: 2025-12-21
GOVERNANCE: Provides autonomous planning for breaking down complex requests.

INVARIANTS:
  - create_plan() always returns a TaskPlan (never None)
  - plan progress is always in range [0.0, 100.0]
  - task dependencies must exist in the same plan
  - max tasks per plan: 50
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging
import uuid

logger = logging.getLogger(__name__)

# Governance: This module is GRADUATED
_IS_STUB = False
_MAX_TASKS_PER_PLAN = 50


class TaskStatus(Enum):
    """Status of a task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class TaskPriority(Enum):
    """Priority of a task."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Task:
    """A single task in a plan."""
    task_id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    dependencies: List[str] = field(default_factory=list)
    result: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskPlan:
    """A complete task plan."""
    plan_id: str
    goal: str
    tasks: List[Task] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def progress(self) -> float:
        """Calculate plan progress as percentage."""
        if not self.tasks:
            return 0.0
        completed = sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED)
        return (completed / len(self.tasks)) * 100
    
    @property
    def next_task(self) -> Optional[Task]:
        """Get the next task to execute."""
        for task in self.tasks:
            if task.status == TaskStatus.PENDING:
                # Check if dependencies are met
                deps_met = all(
                    any(t.task_id == dep and t.status == TaskStatus.COMPLETED 
                        for t in self.tasks)
                    for dep in task.dependencies
                )
                if deps_met:
                    return task
        return None


class AutonomousPlanner:
    """
    Creates and manages task plans for complex requests.
    
    Breaks down complex user requests into actionable steps,
    manages dependencies, and tracks progress.
    """
    
    def __init__(self):
        self.active_plans: Dict[str, TaskPlan] = {}
        self.completed_plans: List[TaskPlan] = []
        
    def create_plan(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None
    ) -> TaskPlan:
        """
        Create a task plan for a goal.
        
        Args:
            goal: The goal to achieve
            context: Optional context information
            
        Returns:
            A TaskPlan with generated tasks
        """
        plan_id = str(uuid.uuid4())[:8]
        tasks = self._generate_tasks(goal, context or {})
        
        plan = TaskPlan(
            plan_id=plan_id,
            goal=goal,
            tasks=tasks,
            metadata=context or {}
        )
        
        self.active_plans[plan_id] = plan
        logger.info(f"Created plan {plan_id} with {len(tasks)} tasks for goal: {goal[:50]}...")
        
        return plan
    
    def _generate_tasks(self, goal: str, context: Dict[str, Any]) -> List[Task]:
        """Generate tasks for a goal."""
        tasks = []
        goal_lower = goal.lower()
        
        # Analyze goal and generate appropriate tasks
        if any(kw in goal_lower for kw in ['code', 'implement', 'build', 'create']):
            tasks = self._generate_coding_tasks(goal, context)
        elif any(kw in goal_lower for kw in ['analyze', 'review', 'check']):
            tasks = self._generate_analysis_tasks(goal, context)
        elif any(kw in goal_lower for kw in ['explain', 'describe', 'what is']):
            tasks = self._generate_explanation_tasks(goal, context)
        else:
            tasks = self._generate_generic_tasks(goal, context)
        
        return tasks
    
    def _generate_coding_tasks(self, goal: str, context: Dict[str, Any]) -> List[Task]:
        """Generate tasks for coding goals."""
        return [
            Task(
                task_id="understand",
                description="Understand the requirements and constraints",
                priority=TaskPriority.HIGH
            ),
            Task(
                task_id="design",
                description="Design the solution architecture",
                dependencies=["understand"],
                priority=TaskPriority.HIGH
            ),
            Task(
                task_id="implement",
                description="Implement the core functionality",
                dependencies=["design"],
                priority=TaskPriority.HIGH
            ),
            Task(
                task_id="test",
                description="Test the implementation",
                dependencies=["implement"],
                priority=TaskPriority.MEDIUM
            ),
            Task(
                task_id="refine",
                description="Refine and optimize the code",
                dependencies=["test"],
                priority=TaskPriority.LOW
            )
        ]
    
    def _generate_analysis_tasks(self, goal: str, context: Dict[str, Any]) -> List[Task]:
        """Generate tasks for analysis goals."""
        return [
            Task(
                task_id="gather",
                description="Gather relevant information",
                priority=TaskPriority.HIGH
            ),
            Task(
                task_id="analyze",
                description="Analyze the gathered information",
                dependencies=["gather"],
                priority=TaskPriority.HIGH
            ),
            Task(
                task_id="synthesize",
                description="Synthesize findings into insights",
                dependencies=["analyze"],
                priority=TaskPriority.MEDIUM
            ),
            Task(
                task_id="report",
                description="Generate analysis report",
                dependencies=["synthesize"],
                priority=TaskPriority.MEDIUM
            )
        ]
    
    def _generate_explanation_tasks(self, goal: str, context: Dict[str, Any]) -> List[Task]:
        """Generate tasks for explanation goals."""
        return [
            Task(
                task_id="identify",
                description="Identify key concepts to explain",
                priority=TaskPriority.HIGH
            ),
            Task(
                task_id="structure",
                description="Structure the explanation",
                dependencies=["identify"],
                priority=TaskPriority.MEDIUM
            ),
            Task(
                task_id="explain",
                description="Provide clear explanation",
                dependencies=["structure"],
                priority=TaskPriority.HIGH
            ),
            Task(
                task_id="examples",
                description="Add examples if helpful",
                dependencies=["explain"],
                priority=TaskPriority.LOW
            )
        ]
    
    def _generate_generic_tasks(self, goal: str, context: Dict[str, Any]) -> List[Task]:
        """Generate generic tasks."""
        return [
            Task(
                task_id="understand",
                description="Understand the request",
                priority=TaskPriority.HIGH
            ),
            Task(
                task_id="process",
                description="Process and respond",
                dependencies=["understand"],
                priority=TaskPriority.HIGH
            ),
            Task(
                task_id="verify",
                description="Verify the response",
                dependencies=["process"],
                priority=TaskPriority.MEDIUM
            )
        ]
    
    def get_plan(self, plan_id: str) -> Optional[TaskPlan]:
        """Get a plan by ID."""
        return self.active_plans.get(plan_id)
    
    def update_task(
        self,
        plan_id: str,
        task_id: str,
        status: TaskStatus,
        result: Optional[str] = None,
        error: Optional[str] = None
    ) -> bool:
        """Update a task's status."""
        plan = self.active_plans.get(plan_id)
        if not plan:
            return False
        
        for task in plan.tasks:
            if task.task_id == task_id:
                task.status = status
                task.result = result
                task.error = error
                if status == TaskStatus.COMPLETED:
                    task.completed_at = datetime.utcnow()
                
                # Check if plan is complete
                if all(t.status == TaskStatus.COMPLETED for t in plan.tasks):
                    plan.status = TaskStatus.COMPLETED
                    plan.completed_at = datetime.utcnow()
                    self.completed_plans.append(plan)
                    del self.active_plans[plan_id]
                
                return True
        
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get planner statistics."""
        return {
            "active_plans": len(self.active_plans),
            "completed_plans": len(self.completed_plans),
            "total_active_tasks": sum(len(p.tasks) for p in self.active_plans.values())
        }


# Global instance
autonomous_planner = AutonomousPlanner()


def create_task_plan(
    goal: str,
    context: Optional[Dict[str, Any]] = None
) -> TaskPlan:
    """
    Convenience function to create a task plan.
    
    Args:
        goal: The goal to achieve
        context: Optional context information
        
    Returns:
        A TaskPlan with generated tasks
    """
    return autonomous_planner.create_plan(goal, context)
