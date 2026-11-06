"""Make `import mascoord` work when pytest runs from the repository root."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
