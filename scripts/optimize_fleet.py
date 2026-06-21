#!/usr/bin/env python3
"""Run fleet optimization."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from carbon_twin.cli import main

if __name__ == "__main__":
    main(["optimize"] + sys.argv[1:], standalone_mode=False)
