# hardware: AMD EPYC 7R13, 1 thread (pure Python)
# command: python labs/thinking-in-tokens/scripts/03_vote_theory.py
# simulated: P(plurality of N draws from q is the right answer), 20,000 trials per cell

| answer marginal q | is right the mode? | N=1 | N=3 | N=5 | N=11 | N=21 | N=41 | N=101 |
|---|---|---|---|---|---|---|---|---|
| right 40%, the other 60% spread thinly over 20 wrong answers (3% each) | yes | 0.399 | 0.483 | 0.680 | 0.907 | 0.989 | 1.000 | 1.000 |
| right 40%, one specific wrong answer at 45%, 15% spread over 5 others | no | 0.399 | 0.409 | 0.441 | 0.425 | 0.402 | 0.366 | 0.297 |
| right 60%, the other 40% on one wrong answer | yes | 0.600 | 0.644 | 0.684 | 0.753 | 0.827 | 0.903 | 0.979 |
| systematic error: the same dropped minus sign 70%, right 20%, 10% scattered | no | 0.203 | 0.131 | 0.097 | 0.028 | 0.005 | 0.000 | 0.000 |
