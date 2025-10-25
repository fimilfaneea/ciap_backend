"""CIAP - Competitive Intelligence Automation Platform"""

__version__ = "0.3.0"  # Module 3 complete

# Expose top-level imports
from . import database
from . import config
from . import cache

__all__ = ["database", "config", "cache"]
