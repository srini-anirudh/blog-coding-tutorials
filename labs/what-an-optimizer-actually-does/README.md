# Lab: What an Optimizer Actually Does

Code companion to the post [What an Optimizer Actually Does](https://srini-anirudh.github.io/blogs/what-an-optimizer-actually-does/).

The post says every optimizer fits one template: `theta <- theta - lr * P^-1 m`. Here `m` is the gradient smoothed in
time and `P` is the metric you descend in. This lab writes each optimizer in the post as that update rule, in plain
PyTorch, and then checks the post's claims one at a time on small problems you can run on a laptop CPU.

C1 ([`c01-pytorch`](../../c01-pytorch/)) already builds plain SGD and AdamW, checks them against `torch.optim`, and
times `optimizer.step()` on CPU. This lab does not repeat that. It adds the variants the post compares: Nesterov, the EMA
form of momentum, Adagrad, RMSProp, coupled L2, the placement of epsilon, Lion, Adafactor, 8-bit AdamW and Muon. The
training runs use C4's TinyLLM ([`c04-llm/tinyllm`](../../c04-llm/)) and its copy of Tiny Shakespeare, imported rather
than rebuilt.

## Layout

```
labs/what-an-optimizer-actually-does/
├── optlab/
│   ├── optimizers.py      # SGD (+ Nesterov, EMA form), Adagrad, RMSProp, Adam (every switch), Lion,
│   │                      # Adafactor, AdamW8bit, Muon, Hybrid
│   ├── newton_schulz.py   # Muon's iteration, its scalar map, exact spectral steepest descent, Shampoo's direction
│   ├── quant.py           # block-wise 8-bit storage with a linear and a log code
│   ├── landscapes.py      # the 2-D surfaces from the post's first figure, plus a rotated bowl
│   ├── train.py           # one short TinyLLM run with any optimizer
│   └── common.py          # hardware labels, result files, model and data helpers
├── scripts/               # one script per claim, listed below
├── benchmarks/step_cost.py
├── results/               # real outputs; each file starts with its hardware and command
└── tests/
```

## Setup

From the repository root, after `pip install -e ".[dev]"`. Set `OMP_NUM_THREADS` on a shared machine. The CPU numbers
below used 8 threads (4 threads for the parallel training runs). Nothing is downloaded.

## Run everything

```bash
python -m pytest labs/what-an-optimizer-actually-does -m "not slow and not gpu and not network"   # about 10 s

python labs/what-an-optimizer-actually-does/scripts/01_landscapes.py        # ~1.5 min
python labs/what-an-optimizer-actually-does/scripts/02_momentum.py          # ~10 s
python labs/what-an-optimizer-actually-does/scripts/03_adaptive.py          # ~20 s
python labs/what-an-optimizer-actually-does/scripts/04_weight_decay.py      # ~1.5 min
python labs/what-an-optimizer-actually-does/scripts/05_matrices.py          # ~5 s
python labs/what-an-optimizer-actually-does/scripts/06_ledger.py            # ~5 s
python labs/what-an-optimizer-actually-does/scripts/07_state_compression.py # ~40 s
python labs/what-an-optimizer-actually-does/scripts/08_train_compare.py     # 27 training runs: ~40 min in one process; or one process per optimizer (--opt), in parallel
python labs/what-an-optimizer-actually-does/benchmarks/step_cost.py         # CPU sizes
python labs/what-an-optimizer-actually-does/benchmarks/step_cost.py --gpu   # H100 sizes, needs CUDA
```

`08_train_compare.py --opt AdamW` runs one optimizer's sweep. `--table` merges the parts into the table.
`--seed-check` repeats three settings with two more seeds. That is how
the runs below were made: one process per optimizer, in parallel.

## Which script covers which section of the post

| Post section | Script |
|---|---|
| One question, three knobs (the six-pen figure) | `01_landscapes.py` |
| The Newton budget, the memory ledger | `06_ledger.py` |
| Picking a geometry | `05_matrices.py` sections 1 and 5 |
| Smoothing in time | `02_momentum.py` |
| Scaling in space, bias correction, update RMS, epsilon | `03_adaptive.py` |
| Weight decay, coupled and not | `03_adaptive.py` section 5, `04_weight_decay.py` |
| Spending less state (Adafactor, 8-bit, Lion) | `07_state_compression.py`, `06_ledger.py`, `08_train_compare.py` |
| Matrices, not vectors (Shampoo, Muon, Newton–Schulz) | `05_matrices.py` |
| Where a step actually costs you | `benchmarks/step_cost.py`, `06_ledger.py` |
| Choosing one, the update-to-weight ratio | `08_train_compare.py` |

## Results

Every number names its hardware. Raw output is in `results/`; each file starts with a hardware line and the command.
The machine was shared and heavily loaded (load average above 100 on 192 cores), so treat CPU timings as orders of
magnitude. Losses, step counts and ratios do not depend on load.

Tests: 22 passed, offline, on AMD EPYC 7R13, 2 threads, torch 2.15.0 (about 6 s). There are no `network`, `slow` or
`gpu` tests.

### AMD EPYC 7R13, 8 threads (training runs: 4 threads), torch 2.15.0

| Claim in the post | Measurement | Script |
|---|---|---|
| The ranking depends on the landscape | Steps to f - f* < 1e-3, each at its best lr. Axis-aligned bowl (kappa 29): RMSProp 6, Nesterov 10, AdamW 60, SGD 78. Same bowl rotated 45 degrees: RMSProp 67, Nesterov 10, AdamW 85, SGD 64. RMSProp loses its edge when the bowl rotates; Nesterov does not care. | `01` |
| Rosenbrock: diagonal scaling does not help | SGD and RMSProp do not reach tolerance in 3000 steps at any lr tried. Momentum 309 steps, AdamW 415, Muon-style 231. AdamW still gets there, at the top of the lr grid (1.0). | `01` |
| Accumulator momentum has an effective lr 1/(1-b) larger | Constant gradient 1, b = 0.9: accumulator m = 10.0000, EMA m = 1.0000 | `02` |
| EMA window | Mean age b/(1-b): 9.0, 99.0, 999.0 for b = 0.9, 0.99, 0.999 | `02` |
| kappa becomes sqrt(kappa) | kappa = 100: GD contracts 0.9802 per step (theory 0.9802), heavy ball 0.8190 (theory 0.8182); steps to 1e-6: 612 vs 85. kappa = 10^4: 0.9998 vs 0.9807 (theory 0.9802); 61,039 vs 972 steps. | `02` |
| Below b* over-damped, above b* rings | kappa = 100, lr tuned per b. b* = 0.669 is fastest (82 steps to 1e-6). b = 0.5: 193 steps, no oscillation. b = 0.9: 257 steps, 6 sign flips of the slowest coordinate. b = 0.95: 16 flips. | `02` |
| Nesterov damps overshoot | Overshoot of the slowest coordinate 0.121 vs 0.153 for heavy ball; 213 vs 258 steps. Both still ring (54 and 56 sign flips), so the effect is modest. | `02` |
| Bias correction, first step | Uncorrected first step is 3.162x lr, corrected 1.000. With a constant gradient the uncorrected step keeps growing: peak 6.57x at step 12, still above 1.5x at step 587. | `03` |
| Adam's update RMS is about 1 | RMS of m_hat/sqrt(v_hat): identical for gradient scales 1e-4, 1, 1e4. But it depends on signal to noise: 0.23 for pure noise (theory 0.229), 0.73 at snr 1, 0.995 at snr 10. | `03` |
| Adagrad decays like 1/sqrt(t) | Step / lr at t = 1, 10, 100, 1000, 10,000: 1.000, 0.316, 0.100, 0.0316, 0.0100. RMSProp settles at 1.0 (its uncorrected first step is 10x). | `03` |
| epsilon placement | Constant g = 1e-10: m/(sqrt(v)+eps) gives 9.9e-3 (= g/eps, an SGD step with lr/eps); m/sqrt(v+eps) gives 1.0e-6 (= g/sqrt(eps)). The two differ by 4 orders of magnitude at g = 1e-10. | `03` |
| Coupled L2 under-decays large-gradient weights | wd 0.1, lr 1e-3, 1000 steps from 1.0, gradient noise std 10 vs 0.01. AdamW: 0.9042 and 0.9057 (pure decay 0.9048). Adam + L2: 0.9895 and 0.2618. | `03` |
| Equilibrium RMS(theta) = sqrt(lr / 2 wd) | Idealized recursion: 0.0385 for lr 3e-4, wd 0.1 (formula 0.0387). Halving lr scales it by 0.711 (sqrt(1/2) = 0.707). | `04` |
| Same, in real AdamW | Scale-invariant layer: RMS(W) is 1.17x to 1.41x the formula with beta1 = 0.9 and 1.32x to 1.76x with beta1 = 0. The scaling with lr and wd holds roughly (wd 0.1 to 2.0 at lr 1e-2: weights shrink 4.3x, formula says 4.5x). The absolute level is above the formula because successive updates are correlated. | `04` |
| Cosine schedule switches decay off | Last tenth of a cosine-to-zero run applies 0.16% of the total decay; the first tenth applies 19.8%. | `04` |
| Steepest descent under the spectral norm is -UV^T | <G, D> per unit spectral norm for a 64 x 176 gradient: -UV^T -4.744 (equals the nuclear-norm bound), 5-step Newton-Schulz -3.901, SGD direction -1.629, sign descent -1.636, best of 2000 random directions -0.255 | `05` |
| Newton-Schulz is a scalar map on singular values | Largest gap between the matrix iteration and the scalar map: 2.2e-7 (fp64) | `05` |
| Muon's coefficients lift small singular values fast | Steps to lift 1e-3 to 0.7: Muon 6, textbook cubic 17. From 1e-2: 4 vs 11. That is about 3x fewer, not 10x. | `05` |
| The output band | After 5 steps, starts in [3e-3, 1] end in [0.682, 1.202]. Starts down to 1e-3 can end as low as 0.47, and 1e-4 ends at 0.05. | `05` |
| bf16 is enough | 256 x 704 matrices: bf16 vs fp32 relative difference 0.026 (random) and 0.041 (spectrum 1/i); 93% to 95% of singular values in (0.7, 1.3) | `05` |
| Rescale so RMS(update) is about 0.2 | RMS(0.2 sqrt(max) O) is 0.172 to 0.188 on four shapes, because the singular values average a bit below 1 | `05` |
| Shampoo's quarter powers | L^-1/4 G R^-1/4 equals (L kron R)^-1/4 vec(G) to 1.6e-15. It does not equal (L kron R)^-1/2 (relative gap 0.90). See "A note for the post" below. | `05` |
| Optimizer state, measured | TinyLLM, 771,200 parameters, one step: SGD 0, momentum 4, Lion 4, AdamW 8, 8-bit AdamW 2.049, Adafactor 0.056, Muon + AdamW 4.176 bytes per parameter (95.6% of this model's parameters are Muon-eligible; the post says 70% to 85% at scale). Same bytes as `torch.optim.AdamW` (8.000). | `06` |
| The 7B ledger | Arithmetic on the post's assumptions, not a 7B run. Per rank on 8 GPUs: 112, 38.5, 26.25, 14 GB for replicated, ZeRO-1, 2, 3. Fused step 28 B/param, intensity 0.43 FLOP/B, 689x below the H100 ridge; 58.5 ms unsharded. Offload 84 GB at 64 GB/s: 1.31 s. | `06` |
| Muon cost | 5 steps cost 20n + 10n^2/m FLOP/param: 122,880 for 4096 x 4096 (0.51% of 6T at T = 4M), 97,161 for 11008 x 4096 (0.40%). The post's 3.3 d/T = 0.34% leaves out the 10n^2/m term. | `06` |
| Muon gather traffic | 2 B/param on the hidden weights: 9.8 to 13.0 GB per step for 7B depending on the hidden share (70% to 93%); 24 to 32 ms on NVLink at 400 GB/s, 245 to 325 ms at 40 GB/s. The post's 13 GB needs 93% hidden. | `06` |
| Adafactor's rank-1 V | Real Adam state after 200 steps: V is rebuilt with relative error 0.15 to 0.47, but the update direction keeps cosine 0.964 to 0.996 with Adam's (mean 0.984). Storage 45,056 to 480 values on the MLP matrices. | `07` |
| 8-bit state | Update error when both m and v are quantized: per-tensor linear 11,909 (relative), block-2048 linear 602, log code 0.027 either per-tensor or per-block. With a linear code 2.8% of nonzero v entries round to 0, and m/(sqrt(v)+eps) becomes m/eps there. | `07` |
| Outliers need blocks | One entry of m multiplied by 1000: per-tensor scale zeroes 81.2% of the other entries, block-wise zeroes 0.3% | `07` |

### Training comparison (`08_train_compare.py`)

TinyLLM from C4, 771,200 parameters, byte-level Tiny Shakespeare, 300 steps of 16 x 128 tokens, warmup then cosine to
10%, clip 1.0, weight decay 0.1 (Lion 1.0). Validation loss on 10 held-out batches, seed 0, best of a learning-rate
sweep. AMD EPYC 7R13, 4 threads, torch 2.15.0, fp32 (Muon's Newton-Schulz also in fp32 on CPU).

| Optimizer | Val loss per lr in the sweep | Best | State B/param | Median RMS(dW)/RMS(W) |
|---|---|---|---|---|
| Muon (2-D) + AdamW | 2.305, 1.878, 1.724, 1.754, 1.917 (lr 1e-3 to 1e-1) | 1.724 | 4.18 | 2.2e-3 |
| 8-bit AdamW | 1.959, 1.918, 2.224 | 1.918 | 2.05 | 2.8e-3 |
| AdamW | 1.964, 1.931, 2.018 | 1.931 | 8.00 | 2.6e-3 |
| Adafactor (no momentum) | 2.014, 2.000, 2.445 | 2.000 | 0.06 | 1.0e-2 |
| Adam + L2, coefficient 1e-3 | 2.149, 2.026, 2.288 | 2.026 | 8.00 | 4.5e-3 |
| Lion | 3.188, 2.386, 2.089, 2.253 (lr 1e-4 to 3e-3) | 2.089 | 4.00 | 1.7e-3 |
| SGD + momentum | 2.564, 2.514, 2.610 | 2.514 | 4.00 | 6.3e-4 |
| Adam + L2, coefficient 0.1 | 3.256, 3.387, 3.231 | 3.231 | 8.00 | 1.7e-1 |

How to read it. Learning-rate sweeps are 3 to 5 points, one seed, one 300-step schedule. Differences of a few hundredths
(AdamW against 8-bit AdamW) are inside the seed spread in `results/08_seed_check.txt`. Three seeds at the best lr: AdamW 1.931, 1.942, 1.958; 8-bit AdamW 1.918, 1.933, 1.937; Muon + AdamW 1.724, 1.723, 1.734 (seeds 1 and 2 are in that file). AdamW and 8-bit AdamW overlap. Muon's
gain is much larger than that noise, but this is a 0.77 M-parameter byte model for 300 steps, so it shows that the
mechanism works and says nothing about the paper-scale claims (fewer steps to a loss target at billions of
parameters). The coupled-L2 run with the same number as AdamW's weight decay is wrecked, and the update-to-weight
ratio (0.17, 60x the healthy runs) shows it, which is the diagnostic the post recommends.

### Optimizer step cost (`benchmarks/step_cost.py`)

NVIDIA H100 80GB HBM3, torch 2.15.0+cu132, CUDA 13.2, one GPU, fp32 state, medians of 20 steps:

| Measurement | Result |
|---|---|
| Fused AdamW on one flat tensor, 16.8 M / 67.1 M / 268.4 M parameters | 0.277 / 0.803 / 2.672 ms, i.e. 1.70 / 2.34 / 2.81 TB/s achieved (28 B/param moved) against 3.35 TB/s peak |
| 217 tensors, 341 M parameters: `optlab.Adam` Python loop / torch for-loop / foreach / fused | 27.3 / 29.5 / 11.5 / 5.0 ms |
| 5 Newton-Schulz steps in bf16, 4096 x 4096 and 11008 x 4096 | 3.68 and 8.87 ms, 561 and 494 TFLOP/s achieved |

The fused step is bandwidth-bound (it approaches the memory peak as the tensor grows, with FLOPs far below the roofline),
and launch overhead is the gap between the 5 ms fused step and the 27 ms loop. Newton-Schulz is matmul-bound.

AMD EPYC 7R13, 8 threads (the machine was heavily loaded; orders of magnitude only): fused AdamW on 1 M to 17 M
parameters 1.4 to 28 GB/s, 73 tensors / 7.5 M parameters: 27.8 ms Python loop, 23.5 ms for-loop, 36.4 ms foreach,
7.8 ms fused; Newton-Schulz in fp32 on 1024 x 1024 took 492 ms. See `results/step_cost_cpu.txt`.

### A note for the post

Four places where the measured behaviour differs from the post's text. They are small, and the post's main claims hold.

1. Shampoo: `L^-1/4 G R^-1/4` equals `(L kron R)^-1/4 vec(G)`, not `(L kron R)^-1/2 vec(G)`. The fourth roots come from
   taking the inverse square root of `(L kron R)^1/2`, the Kronecker estimate of the gradient covariance.
2. Muon overhead: 3.3 d/T is the 20n term only. With the 10n^2/m term, 4096-wide layers cost 0.4% to 0.5% at
   T = 4M tokens.
3. Newton-Schulz: Muon's coefficients need about 3x fewer steps than the textbook cubic from singular values of
   1e-3 to 1e-2 (6 vs 17, 4 vs 11), not 10x, and 5 steps leave starts below 3e-3 under the [0.7, 1.3] band.
4. Adam's update RMS is 1 only when the gradient signal dominates the noise. For pure noise it is
   sqrt((1-b1)/(1+b1)) = 0.23.

### What was not measured

- The paper-scale claims: Muon's token efficiency at billions of parameters, Lion and Adafactor on T5 or large LMs, and
  8-bit AdamW on standard recipes. The training table is one seed per cell on a 0.77 M-parameter model.
- The 7B memory and traffic numbers are arithmetic, not measurements on a 7B run. No multi-GPU run was made, so ZeRO
  sharding, NVLink and inter-node traffic, offload and distributed Muon are not measured.
- The 8-bit code table is a simple log table, not the dynamic-exponent code of Dettmers et al. Adafactor uses a fixed lr
  and beta2 instead of the paper's schedules. Shampoo, SOAP, Adam-mini, Sophia, Dion, MARS and Scion are not implemented
  (Shampoo's direction and Kronecker identity are checked in `05`).
- Muon in bf16 was tested for accuracy (`05`) and timed on the H100 (`step_cost`) but the training runs used fp32 on CPU.
- Gradient accumulation in bf16, stochastic rounding and checkpoint resume behaviour are not tested.

