"""Put the lab folder on sys.path so `import vitfamily` works from any working directory."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
