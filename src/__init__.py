"""CIAP - Competitive Intelligence Automation Platform"""

__version__ = "0.4.0"  # Module 4 complete

# Expose top-level imports
from . import database
from . import config
from . import cache
from . import task_queue

__all__ = ["database", "config", "cache", "task_queue"]
