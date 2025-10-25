"""
API Module for CIAP
FastAPI REST API application and utilities
"""

from .main import app, ws_manager

# Re-export database dependency for convenience
from ..database import get_db

# Import route modules for access
from .routes import search, tasks, analysis, export

__all__ = [
    # Main application
    "app",
    "ws_manager",

    # Database dependency
    "get_db",

    # Route modules
    "search",
    "tasks",
    "analysis",
    "export"
]

__version__ = "0.8.0"
