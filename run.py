#!/usr/bin/env python
"""
CIAP API Server - Production Startup Script
Installs AsyncioSelectorReactor before FastAPI to enable Scrapy-Playwright
"""
import sys
import os

# Ensure project root is in path (for imports to work)
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ============================================================================
# CRITICAL: Install AsyncioSelectorReactor BEFORE any other imports
# This MUST be the first Twisted-related code that runs
# ============================================================================

def install_reactor():
    """
    Install Twisted's AsyncioSelectorReactor for Scrapy-Playwright compatibility

    This must happen before:
    - Any asyncio event loop starts
    - Any FastAPI/Uvicorn initialization
    - Any Twisted imports (except this installer)

    The reactor is a process-wide singleton and cannot be changed once installed.

    On Windows, this also sets the event loop policy to use SelectorEventLoop
    instead of the default ProactorEventLoop, as AsyncioSelectorReactor requires it.
    """
    # Check if reactor already installed (handles reload scenarios)
    if 'twisted.internet.reactor' in sys.modules:
        from twisted.internet import reactor
        reactor_type = f"{reactor.__class__.__module__}.{reactor.__class__.__name__}"
        print(f"[OK] Reactor already installed: {reactor_type}")

        # Verify it's the correct type
        if 'asyncioreactor' not in reactor_type:
            print(f"[WARNING] Wrong reactor type installed!")
            print(f"  Expected: AsyncioSelectorReactor")
            print(f"  Got: {reactor_type}")
            print(f"  Scrapy-Playwright may not work correctly.")
        return

    # Install AsyncioSelectorReactor
    try:
        # CRITICAL: On Windows, set event loop policy to use SelectorEventLoop
        # ProactorEventLoop (Windows default) is not compatible with AsyncioSelectorReactor
        import asyncio
        import platform

        if platform.system() == 'Windows':
            print("[INFO] Windows detected - setting SelectorEventLoop policy")
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        # Now install the reactor
        import twisted.internet.asyncioreactor
        twisted.internet.asyncioreactor.install()

        # Verify installation
        from twisted.internet import reactor
        reactor_type = f"{reactor.__class__.__module__}.{reactor.__class__.__name__}"
        print(f"[OK] Successfully installed AsyncioSelectorReactor")
        print(f"  Reactor: {reactor_type}")

    except Exception as e:
        print(f"[ERROR] CRITICAL: Failed to install AsyncioSelectorReactor: {e}")
        print(f"  Scrapy-Playwright will NOT work.")
        print(f"  Continuing anyway...")

# Install reactor NOW, before any other imports
install_reactor()

# ============================================================================
# Now safe to import application code
# ============================================================================

import uvicorn
from src.config.settings import settings

def main():
    """
    Start the CIAP API server with correct reactor configuration
    """
    # Server configuration
    config = {
        "app": "src.api.main:app",
        "host": "0.0.0.0",
        "port": settings.API_PORT,
        "reload": False,  # Disable reload in production
        "workers": 1,  # Single worker (reactor is not fork-safe)
        "log_level": "info",
        "access_log": True,
    }

    print(f"\n{'='*60}")
    print(f"  CIAP API Server Starting")
    print(f"  Port: {config['port']}")
    print(f"  Workers: {config['workers']}")
    print(f"  Reload: {config['reload']}")
    print(f"{'='*60}\n")

    # Start server
    uvicorn.run(**config)


if __name__ == "__main__":
    main()
