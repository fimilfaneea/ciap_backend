"""CIAP - Competitive Intelligence Automation Platform"""

__version__ = "1.0.0"  # All 10 modules complete - Production ready!

# Expose top-level imports
from . import database
from . import config
from . import cache
from . import task_queue
from . import scrapers
from . import processors
from . import analyzers
from . import api
from . import services

__all__ = [
    "database",
    "config",
    "cache",
    "task_queue",
    "scrapers",
    "processors",
    "analyzers",
    "api",
    "services"
]
