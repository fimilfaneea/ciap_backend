"""Task Queue module - Background job processing with priority queue"""

from .manager import (
    TaskQueueManager,
    task_queue,
    TaskStatus,
    TaskPriority
)
from .handlers import (
    scrape_handler,
    analyze_handler,
    export_handler,
    batch_handler,
    register_default_handlers
)
from .utils import (
    TaskChain,
    TaskGroup,
    wait_for_task,
    schedule_recurring_task,
    create_workflow,
    get_workflow_status
)

__all__ = [
    # Manager
    "TaskQueueManager",
    "task_queue",
    "TaskStatus",
    "TaskPriority",

    # Handlers
    "scrape_handler",
    "analyze_handler",
    "export_handler",
    "batch_handler",
    "register_default_handlers",

    # Utilities
    "TaskChain",
    "TaskGroup",
    "wait_for_task",
    "schedule_recurring_task",
    "create_workflow",
    "get_workflow_status",
]
