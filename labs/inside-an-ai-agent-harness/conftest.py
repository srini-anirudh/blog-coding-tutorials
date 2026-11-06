"""Make `import harness_lab` work when pytest runs from the repository root.

Importing harness_lab also puts the C10 tutorial on the path, for `import harness`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import harness_lab  # noqa: E402,F401
