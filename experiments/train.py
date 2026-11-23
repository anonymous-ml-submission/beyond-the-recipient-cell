#!/usr/bin/env python3
"""Clean public entry point for an unchanged original research implementation."""
from pathlib import Path
import os
import runpy
import sys

repo = Path(__file__).resolve().parents[1]
script = repo / 'scripts/train_primary2025_context_v1.py'
if not script.is_file():
    raise SystemExit("Required original implementation is unavailable: " + str(script.relative_to(repo)))
sys.path.insert(0, str(repo / "scripts"))
sys.path.insert(0, str(repo))
os.chdir(repo)
sys.argv[0] = str(script)
runpy.run_path(str(script), run_name="__main__")
