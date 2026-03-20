"""
Background Task Queue Infrastructure
=====================================

Simplified task queue for background processing.

Ported from old backend: ResonantGraphAIV0.1/backend/fastapi_app/infrastructure/queue.py
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Callable, Optional, List
from datetime import datetime
from collections import deque
import threading

logger = logging.getLogger(__name__)


class TaskQueue:
    """
    Simple in-process task queue for background processing.
    
    Phase 0: In-memory queue with thread-based workers.
    Future: Can be replaced with Redis + Celery/RQ.
    """
    
    def __init__(self, max_workers: int = 4, max_queue_size: int = 1000):
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self._queue: deque = deque(maxlen=max_queue_size)
        self._workers: List[threading.Thread] = []
        self._running = False
        self._lock = threading.Lock()
        self._stats = {
            "enqueued": 0,
            "processed": 0,
            "failed": 0
        }
    
    def enqueue(
        self,
        task_type: str,
        data: Dict[str, Any],
        func: Optional[Callable] = None
    ) -> bool:
        """
        Enqueue a task for background processing.
        
        Args:
            task_type: Type of task
            data: Task data
            func: Optional handler function
        
        Returns:
            True if enqueued, False if queue full
        """
        with self._lock:
            if len(self._queue) >= self.max_queue_size:
                logger.warning(f"Task queue full, dropping task: {task_type}")
                return False
            
            task = {
                "type": task_type,
                "data": data,
                "func": func,
                "enqueued_at": datetime.utcnow().isoformat()
            }
            self._queue.append(task)
            self._stats["enqueued"] += 1
            logger.debug(f"Enqueued task: {task_type}")
            return True
    
    def _process_task(self, task: Dict[str, Any]) -> bool:
        """Process a single task."""
        try:
            task_type = task.get("type", "unknown")
            data = task.get("data", {})
            func = task.get("func")
            
            if func:
                func(**data)
            else:
                # Default handlers for known task types
                self._default_handler(task_type, data)
            
            self._stats["processed"] += 1
            logger.debug(f"Processed task: {task_type}")
            return True
            
        except Exception as e:
            self._stats["failed"] += 1
            logger.error(f"Task failed: {e}", exc_info=True)
            return False
    
    def _default_handler(self, task_type: str, data: Dict[str, Any]) -> None:
        """Default handler for tasks without explicit function."""
        logger.info(f"Default handler for task: {task_type}")
        # No-op by default - specific handlers should be registered
    
    def _worker_loop(self) -> None:
        """Worker thread loop."""
        while self._running:
            task = None
            with self._lock:
                if self._queue:
                    task = self._queue.popleft()
            
            if task:
                self._process_task(task)
            else:
                # Sleep briefly if no tasks
                import time
                time.sleep(0.1)
    
    def start_workers(self) -> None:
        """Start worker threads."""
        if self._running:
            return
        
        self._running = True
        for i in range(self.max_workers):
            worker = threading.Thread(target=self._worker_loop, daemon=True)
            worker.start()
            self._workers.append(worker)
        
        logger.info(f"Started {self.max_workers} task queue workers")
    
    def stop_workers(self) -> None:
        """Stop worker threads gracefully."""
        self._running = False
        for worker in self._workers:
            worker.join(timeout=5.0)
        self._workers.clear()
        logger.info("Stopped task queue workers")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self._lock:
            return {
                **self._stats,
                "queue_size": len(self._queue),
                "workers": len(self._workers),
                "running": self._running
            }
    
    def get_queue_size(self) -> int:
        """Get current queue size."""
        with self._lock:
            return len(self._queue)


# Global task queue instance
task_queue = TaskQueue()


def enqueue_memory_task(task_type: str, data: Dict[str, Any]) -> None:
    """Enqueue a memory-related task."""
    task_queue.enqueue(task_type=task_type, data=data)


def enqueue_agent_task(task_type: str, data: Dict[str, Any]) -> None:
    """Enqueue an agent-related task."""
    task_queue.enqueue(task_type=task_type, data=data)


def start_memory_workers() -> None:
    """Start the task queue workers."""
    task_queue.start_workers()


def stop_memory_workers() -> None:
    """Stop the task queue workers."""
    task_queue.stop_workers()


def get_memory_queue_stats() -> Dict[str, Any]:
    """Get queue statistics."""
    return task_queue.get_stats()


def get_memory_queue_size() -> int:
    """Get current queue size."""
    return task_queue.get_queue_size()
