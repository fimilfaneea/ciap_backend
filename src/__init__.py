"""CIAP - Competitive Intelligence Automation Platform"""

__version__ = "0.5.0"  # Module 5 complete

# Expose top-level imports
from . import database
from . import config
from . import cache
from . import task_queue
from . import scrapers

__all__ = ["database", "config", "cache", "task_queue", "scrapers"]
