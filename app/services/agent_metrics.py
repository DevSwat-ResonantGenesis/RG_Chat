"""
Agent Performance Metrics System (APMS)
========================================

Phase 4: Track and analyze agent performance for continuous improvement.

Features:
- Per-agent performance tracking
- Response time metrics
- Quality scoring
- Usage analytics
- Error rate tracking
"""
from __future__ import annotations

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


@dataclass
class AgentMetricEntry:
    """A single metric entry for an agent execution."""
    agent_type: str
    execution_time_ms: float
    token_count: int
    success: bool
    error_message: Optional[str]
    quality_score: Optional[float]  # 0-1, from user feedback or auto-evaluation
    timestamp: str
    user_id: str
    task_length: int
    response_length: int


@dataclass
class AgentPerformanceStats:
    """Aggregated performance statistics for an agent."""
    agent_type: str
    total_executions: int
    successful_executions: int
    failed_executions: int
    success_rate: float
    avg_execution_time_ms: float
    min_execution_time_ms: float
    max_execution_time_ms: float
    p95_execution_time_ms: float
    avg_quality_score: float
    avg_token_count: float
    avg_response_length: float
    last_execution: str
    executions_last_hour: int
    executions_last_day: int


class AgentMetricsCollector:
    """
    Collects and analyzes agent performance metrics.
    
    In production, this would be backed by a time-series database like InfluxDB or Prometheus.
    """
    
    def __init__(self, max_entries_per_agent: int = 1000):
        self.metrics: Dict[str, List[AgentMetricEntry]] = defaultdict(list)
        self.max_entries_per_agent = max_entries_per_agent
    
    def record(
        self,
        agent_type: str,
        execution_time_ms: float,
        token_count: int,
        success: bool,
        user_id: str,
        task_length: int,
        response_length: int,
        error_message: Optional[str] = None,
        quality_score: Optional[float] = None,
    ):
        """Record a metric entry for an agent execution."""
        entry = AgentMetricEntry(
            agent_type=agent_type,
            execution_time_ms=execution_time_ms,
            token_count=token_count,
            success=success,
            error_message=error_message,
            quality_score=quality_score,
            timestamp=datetime.now().isoformat(),
            user_id=user_id,
            task_length=task_length,
            response_length=response_length,
        )
        
        self.metrics[agent_type].append(entry)
        
        # Prune old entries
        if len(self.metrics[agent_type]) > self.max_entries_per_agent:
            self.metrics[agent_type] = self.metrics[agent_type][-self.max_entries_per_agent:]
        
        logger.debug(f"📊 Recorded metric for {agent_type}: {execution_time_ms:.0f}ms, success={success}")
    
    def get_agent_stats(self, agent_type: str) -> Optional[AgentPerformanceStats]:
        """Get performance statistics for a specific agent."""
        if agent_type not in self.metrics or not self.metrics[agent_type]:
            return None
        
        entries = self.metrics[agent_type]
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)
        
        successful = [e for e in entries if e.success]
        failed = [e for e in entries if not e.success]
        
        execution_times = [e.execution_time_ms for e in entries]
        quality_scores = [e.quality_score for e in entries if e.quality_score is not None]
        token_counts = [e.token_count for e in entries]
        response_lengths = [e.response_length for e in entries]
        
        # Calculate percentile
        sorted_times = sorted(execution_times)
        p95_index = int(len(sorted_times) * 0.95)
        p95_time = sorted_times[p95_index] if sorted_times else 0
        
        # Count recent executions
        executions_last_hour = sum(
            1 for e in entries
            if datetime.fromisoformat(e.timestamp) > hour_ago
        )
        executions_last_day = sum(
            1 for e in entries
            if datetime.fromisoformat(e.timestamp) > day_ago
        )
        
        return AgentPerformanceStats(
            agent_type=agent_type,
            total_executions=len(entries),
            successful_executions=len(successful),
            failed_executions=len(failed),
            success_rate=len(successful) / len(entries) if entries else 0,
            avg_execution_time_ms=statistics.mean(execution_times) if execution_times else 0,
            min_execution_time_ms=min(execution_times) if execution_times else 0,
            max_execution_time_ms=max(execution_times) if execution_times else 0,
            p95_execution_time_ms=p95_time,
            avg_quality_score=statistics.mean(quality_scores) if quality_scores else 0,
            avg_token_count=statistics.mean(token_counts) if token_counts else 0,
            avg_response_length=statistics.mean(response_lengths) if response_lengths else 0,
            last_execution=entries[-1].timestamp if entries else "",
            executions_last_hour=executions_last_hour,
            executions_last_day=executions_last_day,
        )
    
    def get_all_stats(self) -> Dict[str, AgentPerformanceStats]:
        """Get performance statistics for all agents."""
        return {
            agent_type: self.get_agent_stats(agent_type)
            for agent_type in self.metrics.keys()
            if self.get_agent_stats(agent_type) is not None
        }
    
    def get_top_agents(self, metric: str = "success_rate", limit: int = 5) -> List[Dict[str, Any]]:
        """Get top performing agents by a specific metric."""
        all_stats = self.get_all_stats()
        
        if not all_stats:
            return []
        
        # Sort by the specified metric
        sorted_agents = sorted(
            all_stats.items(),
            key=lambda x: getattr(x[1], metric, 0),
            reverse=True
        )
        
        return [
            {
                "agent_type": agent_type,
                "value": getattr(stats, metric, 0),
                "total_executions": stats.total_executions,
            }
            for agent_type, stats in sorted_agents[:limit]
        ]
    
    def get_slowest_agents(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get slowest agents by average execution time."""
        all_stats = self.get_all_stats()
        
        if not all_stats:
            return []
        
        sorted_agents = sorted(
            all_stats.items(),
            key=lambda x: x[1].avg_execution_time_ms,
            reverse=True
        )
        
        return [
            {
                "agent_type": agent_type,
                "avg_time_ms": stats.avg_execution_time_ms,
                "p95_time_ms": stats.p95_execution_time_ms,
            }
            for agent_type, stats in sorted_agents[:limit]
        ]
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors across all agents."""
        error_counts = defaultdict(int)
        error_messages = defaultdict(list)
        
        for agent_type, entries in self.metrics.items():
            for entry in entries:
                if not entry.success and entry.error_message:
                    error_counts[agent_type] += 1
                    if entry.error_message not in error_messages[agent_type]:
                        error_messages[agent_type].append(entry.error_message[:100])
        
        return {
            "total_errors": sum(error_counts.values()),
            "errors_by_agent": dict(error_counts),
            "recent_error_messages": {
                agent: messages[-5:]  # Last 5 unique errors
                for agent, messages in error_messages.items()
            }
        }
    
    def get_usage_trends(self, hours: int = 24) -> Dict[str, List[int]]:
        """Get hourly usage trends for the last N hours."""
        now = datetime.now()
        trends = defaultdict(lambda: [0] * hours)
        
        for agent_type, entries in self.metrics.items():
            for entry in entries:
                try:
                    entry_time = datetime.fromisoformat(entry.timestamp)
                    hours_ago = int((now - entry_time).total_seconds() / 3600)
                    if 0 <= hours_ago < hours:
                        trends[agent_type][hours - 1 - hours_ago] += 1
                except:
                    pass
        
        return dict(trends)
    
    def add_quality_feedback(
        self,
        agent_type: str,
        timestamp: str,
        quality_score: float,
    ) -> bool:
        """Add quality feedback to a specific execution."""
        if agent_type not in self.metrics:
            return False
        
        for entry in reversed(self.metrics[agent_type]):
            if entry.timestamp == timestamp:
                entry.quality_score = max(0, min(1, quality_score))
                logger.debug(f"📝 Updated quality score for {agent_type}: {quality_score}")
                return True
        
        return False
    
    def export_metrics(self, agent_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Export metrics for analysis."""
        if agent_type:
            entries = self.metrics.get(agent_type, [])
            return [
                {
                    "agent_type": e.agent_type,
                    "execution_time_ms": e.execution_time_ms,
                    "token_count": e.token_count,
                    "success": e.success,
                    "quality_score": e.quality_score,
                    "timestamp": e.timestamp,
                    "task_length": e.task_length,
                    "response_length": e.response_length,
                }
                for e in entries
            ]
        
        # Export all
        result = []
        for entries in self.metrics.values():
            for e in entries:
                result.append({
                    "agent_type": e.agent_type,
                    "execution_time_ms": e.execution_time_ms,
                    "token_count": e.token_count,
                    "success": e.success,
                    "quality_score": e.quality_score,
                    "timestamp": e.timestamp,
                    "task_length": e.task_length,
                    "response_length": e.response_length,
                })
        
        return result


# Global instance
agent_metrics = AgentMetricsCollector()
