#!/bin/bash
# Run every script from the repository root and save its output in results/.
# Scripts 01 and 02 download wikitext-2 and two tokenizers; 07 downloads CIFAR-10 and wants a GPU.
set -euo pipefail
L=labs/one-transformer-two-modalities
python $L/scripts/01_front_doors.py      > $L/results/01_front_doors.txt
python $L/scripts/02_sequences.py        > $L/results/02_sequences.txt
python $L/scripts/03_position.py         > $L/results/03_position.txt
python $L/scripts/04_masks.py            > $L/results/04_masks.txt
python $L/scripts/05_identical_middle.py > $L/results/05_identical_middle.txt
python $L/scripts/06_inductive_bias.py   > $L/results/06_inductive_bias.txt
python $L/scripts/07_vit_vs_cnn.py --out $L/results/vit_vs_cnn.json > $L/results/vit_vs_cnn.txt
