"""Make `import harness` work when pytest runs from the repository root."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "tests"))  # for tests/helpers.py

# The toy repo has a deliberately failing test (the agent's job is to fix it).
collect_ignore = ["examples"]
