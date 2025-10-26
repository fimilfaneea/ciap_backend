#!/usr/bin/env python
"""
CIAP API Server - Development Mode
Enables reload but reactor is reinstalled on each restart
"""
import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Install reactor (same as production)
def install_reactor():
    """
    Install AsyncioSelectorReactor for development mode

    Note: With reload enabled, this function runs in the worker subprocess
    every time code changes trigger a restart.

    On Windows, sets the event loop policy to use SelectorEventLoop.
    """
    if 'twisted.internet.reactor' in sys.modules:
        return

    try:
        # Set Windows event loop policy before installing reactor
        import asyncio
        import platform

        if platform.system() == 'Windows':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        import twisted.internet.asyncioreactor
        twisted.internet.asyncioreactor.install()
        print("[OK] AsyncioSelectorReactor installed (dev mode)")
    except Exception as e:
        print(f"[WARNING] Reactor installation failed: {e}")

install_reactor()

import uvicorn
from src.config.settings import settings

def main():
    """
    Development server with auto-reload

    Note: On file changes, server restarts and reactor is reinstalled
    """
    config = {
        "app": "src.api.main:app",
        "host": "127.0.0.1",  # Localhost only for dev
        "port": settings.API_PORT,
        "reload": True,  # Enable reload
        "reload_dirs": ["src"],  # Watch src directory only
        "reload_excludes": ["*.pyc", "__pycache__", ".git"],
        "log_level": "debug",  # Verbose logging
        "access_log": True,
    }

    print(f"\n{'='*60}")
    print(f"  CIAP API Server - DEVELOPMENT MODE")
    print(f"  Port: {config['port']}")
    print(f"  Auto-reload: ENABLED")
    print(f"  Watching: src/")
    print(f"{'='*60}\n")

    uvicorn.run(**config)

if __name__ == "__main__":
    main()
