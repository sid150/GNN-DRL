"""
Entry point for running backend as a module: python -m backend
This file delegates to main.py's CLI interface.
"""

import sys
from .main import main

if __name__ == "__main__":
    sys.exit(main())
