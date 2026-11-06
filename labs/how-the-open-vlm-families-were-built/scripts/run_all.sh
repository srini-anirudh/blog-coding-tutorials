#!/usr/bin/env bash
# Run every script from the repository root. 02, 03, 04 (part 3) and 06 need network access to the Hub.
set -euo pipefail
cd "$(dirname "$0")/../../.."
L=labs/how-the-open-vlm-families-were-built/scripts
for s in 01_tokens_per_image 02_check_processors 03_param_split 04_connectors 05_positions \
         06_tokens_per_page 07_mixing_and_losses 08_cross_family; do
  echo "== $s"
  python "$L/$s.py" > /dev/null
done
echo "outputs in labs/how-the-open-vlm-families-were-built/results/"
