#!/bin/bash
# Every run behind the README's results, in order. Run from the repository root.
set -euo pipefail
D=labs/aligning-vision-language-models/scripts
python $D/prior_split_smolvlm.py --scenes 80 --describe 30   # downloads SmolVLM-256M on first use
python $D/chair_silence.py        # trains the toy SFT policy into runs/sft.pt on first use (about 2 minutes)
python $D/reward_audit.py --seeds 0 1 --steps 1500
python $D/mdpo.py --seeds 0 1
python $D/token_credit.py
python $D/judge_loop.py --seeds 0
python $D/abstention.py
