"""
Entry point for running the package as a module.

Usage:
    python -m radiant [command] [options]
"""

import sys
from radiant.cli import main

if __name__ == "__main__":
    sys.exit(main())
