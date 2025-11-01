#!/usr/bin/env python
"""
Playwright Browser Setup Script
Run this after installing requirements.txt to download browser binaries
"""
import subprocess
import sys


def install_playwright_browsers():
    """Install Playwright browser binaries"""
    print("="*70)
    print("Installing Playwright Browser Binaries")
    print("="*70)
    print("\nThis is a one-time setup step.")
    print("Browser binaries (~150MB) will be downloaded to:")
    print("  Windows: %USERPROFILE%\\AppData\\Local\\ms-playwright")
    print("  Linux/Mac: ~/.cache/ms-playwright")
    print("\n" + "="*70 + "\n")

    try:
        # Install chromium browser
        print("[1/1] Installing Chromium browser...")
        result = subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            check=True,
            capture_output=False
        )

        print("\n" + "="*70)
        print("SUCCESS: Playwright browsers installed!")
        print("="*70)
        print("\nYou can now run the CIAP scrapers.")
        print("This setup only needs to be done once.\n")

        return 0

    except subprocess.CalledProcessError as e:
        print("\n" + "="*70)
        print("ERROR: Failed to install Playwright browsers")
        print("="*70)
        print(f"\nError: {e}")
        print("\nPlease try manually running:")
        print("  playwright install chromium")
        print()
        return 1

    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(install_playwright_browsers())
