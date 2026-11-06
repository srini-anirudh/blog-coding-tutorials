# hardware: AMD EPYC 7R13, 2 threads, torch 2.15.0+cu132 per run, 4 runs in parallel
# command: python labs/thinking-in-tokens/scripts/01_parity_depth.py
# 2000 AdamW steps, batch 256, fresh random strings each step; accuracy on 2000 held-out strings

| n bits | format | layers | tokens written | depth bound L*K | final answer acc | first step at >=0.99 | seed |
|---|---|---|---|---|---|---|---|
| 4 | direct | 2 | 1 | 2 | 1.000 | 500 | 0 |
| 4 | cot | 2 | 4 | 8 | 1.000 | 500 | 0 |
| 8 | direct | 2 | 1 | 2 | 0.514 | never | 0 |
| 8 | cot | 2 | 8 | 16 | 1.000 | 500 | 0 |
| 16 | direct | 2 | 1 | 2 | 0.507 | never | 0 |
| 16 | direct | 6 | 1 | 6 | 0.507 | never | 0 |
| 16 | cot | 2 | 16 | 32 | 1.000 | 500 | 0 |
| 32 | direct | 2 | 1 | 2 | 0.521 | never | 0 |
| 32 | cot | 2 | 32 | 64 | 1.000 | 500 | 0 |
