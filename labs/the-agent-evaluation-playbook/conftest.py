"""Put this folder on sys.path so ``import agenteval`` works when pytest runs from the repo root."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
