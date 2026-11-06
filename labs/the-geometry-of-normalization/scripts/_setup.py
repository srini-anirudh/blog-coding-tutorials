"""Put the lab folder on sys.path so scripts can `import normgeo` from anywhere."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
