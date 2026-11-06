#!/usr/bin/env bash
# Run every script in this lab from the repository root. Outputs go to labs/vlms-that-act/results/.
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY=${PY:-python}
L=labs/vlms-that-act/scripts
$PY $L/01_loop_and_state.py
$PY $L/02_observations.py
$PY $L/03_resolution.py
$PY $L/04_grounding_zoom.py      # trains a grounder (the longest step), then evaluates
$PY $L/05_action_levels.py
$PY $L/06_bc_dagger_rl.py        # trains BC, BC with 4x demos, DAgger and BC + RL policies
$PY $L/07_harness.py
