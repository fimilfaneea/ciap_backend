"""CIAP - Competitive Intelligence Automation Platform"""

__version__ = "0.6.0"  # Module 6 complete

# Expose top-level imports
from . import database
from . import config
from . import cache
from . import task_queue
from . import scrapers
from . import processors

__all__ = ["database", "config", "cache", "task_queue", "scrapers", "processors"]
