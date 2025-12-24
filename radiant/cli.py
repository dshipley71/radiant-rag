"""
CLI entry point for Radiant Agentic RAG.

This module provides the command-line interface wrapper.
The main logic is in radiant.app.main().
"""

from radiant.app import main

__all__ = ["main"]

if __name__ == "__main__":
    import sys
    sys.exit(main())
