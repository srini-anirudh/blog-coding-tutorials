#!/usr/bin/env bash
# Run every script in order from the repository root. Outputs go to labs/how-vlms-read-documents/results/.
set -euo pipefail
cd "$(dirname "$0")/../../.."
S=labs/how-vlms-read-documents/scripts
python $S/01_token_budget.py --check-processor   # drop the flag if Qwen2-VL is not downloaded
python $S/02_small_text.py                       # trains the CRNN; the longest step
python $S/03_layout_embeddings.py
python $S/04_output_formats.py
python $S/05_teds.py
python $S/06_reading_order.py
python $S/07_scoring.py
