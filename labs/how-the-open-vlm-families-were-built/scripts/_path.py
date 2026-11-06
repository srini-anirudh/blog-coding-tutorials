"""Make `vlmfam` importable when a script runs from the repository root, and quiet the HF logs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import transformers  # noqa: E402

transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
