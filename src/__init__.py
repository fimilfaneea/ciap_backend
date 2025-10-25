"""CIAP - Competitive Intelligence Automation Platform"""

__version__ = "0.7.0"  # Module 7 complete

# Expose top-level imports
from . import database
from . import config
from . import cache
from . import task_queue
from . import scrapers
from . import processors
from . import analyzers

__all__ = ["database", "config", "cache", "task_queue", "scrapers", "processors", "analyzers"]
