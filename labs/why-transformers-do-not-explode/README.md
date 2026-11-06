# Lab: Why Transformers Do Not Explode

This is the code companion to the post
[Why Transformers Do Not Explode](https://srini-anirudh.github.io/blogs/why-transformers-do-not-explode/).
The post argues that a deep Transformer survives training because a stack of mechanisms each holds one failure
closed. This lab builds each mechanism in small PyTorch files and measures it, on random networks and on a
character-level Transformer trained on Tiny Shakespeare.

What you can run:

- One matmul multiplies variance by `n_in * sigma_w^2`. Sixty plain layers under naive, Xavier and Kaiming init,
  forward and backward, for ReLU, tanh and linear. The zero-drift random walk that criticality still leaves.
- The mean-field correlation map of a random tanh network: the order-to-chaos line, `chi_1`, and the depth scale,
  checked against real networks. Jacobian spectra under Gaussian and orthogonal weights.
- Residual streams growing like `sqrt(L)`, the identity term in the gradient, and the path ensemble expanded term by
  term.
- LayerNorm's Jacobian, the per-layer gradient profile of Post-LN and Pre-LN at init, and training with and without
  warmup.
- DeepNorm's constants, and the size of one update `||Delta F||` against depth.
- Softmax saturation: entropy against the softmax Jacobian, what QK-norm and soft-capping bound, and a learning-rate
  sweep with and without QK-norm.
- Adam under stress: the first step, stale second moments, `eps`, global clipping, scale invariance under
  normalization, and gradient noise against batch size.
- Number formats: the limits table, rounding stagnation, master weights, underflow of real gradients, and dynamic
  loss scaling.
- SP against muP: a coordinate check and a learning-rate sweep across widths.

Everything runs on a CPU. The post quotes large-scale results (DeepNet at 1,000 layers, ViT-22B, OPT and PaLM
spikes, muTransfer to GPT-3 6.7B, FP8 training of DeepSeek-V3). None of those were reproduced. Where a script shows
the same mechanism small, it says so.

## Layout

```
labs/why-transformers-do-not-explode/
├── stablelab/
│   ├── common.py      # seeding; results files start with the hardware and the command
│   ├── signal.py      # gains, plain stacks, log-variance walk, mean-field maps, Jacobian spectra
│   ├── model.py       # TinyTransformer: pre / post / DeepNorm, init schemes, QK-norm, soft-cap; char data; train()
│   ├── optim.py       # InspectableAdam, the Figure 10 gradient stream, global-norm clipping
│   ├── numerics.py    # format table, underflow counting, DynamicLossScaler
│   └── mup.py         # MLP in SP and muP (Tensor Programs V, Table 3), coordinate check
├── scripts/           # 01..09, one per part of the post, and run_all.sh
├── results/           # real outputs of every script
└── tests/
```

## Setup and tests

From the repository root, after `pip install -e ".[dev]"`. The lab needs only `torch` and `numpy`. Scripts 04 to 09
read Tiny Shakespeare (1.1 MB). They use `c04-llm/data/tinyshakespeare.txt` if it exists, and otherwise download it
into `labs/why-transformers-do-not-explode/data/`, which stays out of git.

```bash
python -m pytest labs/why-transformers-do-not-explode -m "not slow and not gpu and not network"   # 32 tests, about 10 s
```

The tests use a small synthetic corpus, so they need no download. There are no `network`, `slow` or `gpu` tests.

## Run everything

```bash
bash labs/why-transformers-do-not-explode/scripts/run_all.sh
```

Or one at a time:

```bash
python labs/why-transformers-do-not-explode/scripts/01_variance_and_init.py      # 20 s
python labs/why-transformers-do-not-explode/scripts/02_signal_propagation.py     # 25 s
python labs/why-transformers-do-not-explode/scripts/03_residual_stream.py        # 10 s
python labs/why-transformers-do-not-explode/scripts/04_norm_placement.py --steps 400 --warmup 100   # about 35 min
python labs/why-transformers-do-not-explode/scripts/05_deepnorm_update.py        # 15 s
python labs/why-transformers-do-not-explode/scripts/06_attention_logits.py       # about 10 min; --no-train: seconds
python labs/why-transformers-do-not-explode/scripts/07_adam_under_stress.py      # 10 s
python labs/why-transformers-do-not-explode/scripts/08_number_formats.py         # 45 s
python labs/why-transformers-do-not-explode/scripts/09_mup_transfer.py           # about 20 min
```

Timings are wall-clock on a shared machine and are only a guide.

## Which script covers which part of the post

| Post section | Script |
|---|---|
| 02 Part I: Variance through a matrix multiply (gain, Xavier, Kaiming, `calculate_gain`) | `01_variance_and_init.py` |
| 03 Part II: Stacking L of them (log-variance walk, criticality) | `01_variance_and_init.py` |
| 03 Part II: Dynamical signal propagation, orthogonal init and dynamical isometry | `02_signal_propagation.py` |
| 04 Part III: Residual streams change everything (`sqrt(L)`, identity term, path ensemble) | `03_residual_stream.py` |
| 05 Part IV: Where the normalizer goes | `04_norm_placement.py` |
| 06 Part V: Scaling the residual branch (DeepNorm, `1/sqrt(2L)`) | `05_deepnorm_update.py`, and `03_residual_stream.py` for the stream |
| 07 Part VI: Attention has its own failure mode | `06_attention_logits.py` |
| 08 Part VII: The optimizer is part of the dynamical system | `07_adam_under_stress.py` |
| 09 Part VIII: Instability that lives in the number format | `08_number_formats.py` |
| 10 Part IX: A debugging taxonomy for loss spikes | not built here; see C5's `c05-pretraining/scripts/break_it.py` and `pretrain/guards.py` |
| 11 Part X: Making hyperparameters survive a scale-up | `09_mup_transfer.py` |

## Results

Every number names its hardware. Raw output is in `results/`, under the name of the script that wrote it. Each file
starts with the hardware line and the command.

### AMD EPYC 7R13, 8 threads, torch 2.15.0 (CPU only; the machine was shared, so timings are rough)

Tests: 32 passed offline in about 9 s. They do not write to `results/`.

| Script | Measurement | Result |
|---|---|---|
| 01 | Variance after one matmul, predicted `n_in * sigma_w^2` against measured, nine settings | within 3% in every case (for example 4 against 4.03) |
| 01 | 60 plain ReLU layers, width 512, per-layer forward gain: naive / Xavier / Kaiming | 0.026 / 0.4995 / 0.999 |
| 01 | Same stack, tanh with Kaiming: forward second moment / input-gradient second moment | 0.61 / 199 (forward fine, backward 200x too large) |
| 01 | 64 critical linear nets, `std ln-var` at L = 100: width 64 / width 512 | 1.67 / 0.61 (prediction `s*sqrt(L)`: 1.77 / 0.62) |
| 02 | Random tanh net, sigma_b 0.2, critical sigma_w from `chi_1 = 1` | 1.3041 |
| 02 | Depth scale xi in the ordered phase, sigma_w 0.8 / 1.0: theory against fit | 1.68 against 1.66 / 4.0 against 4.2 |
| 02 | Depth scale in the chaotic phase, sigma_w 1.5 / 2 / 3: theory against fit | 14.5 against 23.5 / 7.1 against 12.3 / 4.5 against 8.3 (fit 1.6 to 1.8x longer) |
| 02 | Jacobian, width 200, L = 128, share of singular values in (0.5, 2): tanh orthogonal / tanh Gaussian / ReLU orthogonal | 0.53 / 0.015 / 0.015 |
| 03 | Residual stream norm at L = 96, `||x_L|| / ||x_0||`: plain residual / `1/sqrt(2L)` / LayerScale 1e-4 | 9.56 / 1.21 / 1.00 |
| 03 | `sqrt(1 + L v)` with v fit on the first 12 blocks, predicting L = 96 | 9.29 predicted, 9.56 measured |
| 03 | Plain linear stack, variance gain 1.1, at L = 96 | stream norm 114x |
| 03 | Gradient at the bottom of 128 layers: plain gain 0.7 / residual with dead branches | 1.3e-20 / 1.0 |
| 04 | Gradient norm, top block / bottom block at init, 12 layers: Post-LN default / Post-LN Xavier / Pre-LN default | 1.02 / 1.80 / 0.33 |
| 04 | Val loss after 400 steps, lr 1e-3 / 3e-3, Post-LN default init (no warmup) | 2.060 / 2.005 |
| 04 | Same, Post-LN with Xavier init, no warmup, and with 100 warmup steps | stuck at 3.365 / 3.364, and 3.365 / 3.385 |
| 04 | Same, Pre-LN default init, lr 1e-3 / 3e-3: no warmup, then warmup 100 | 2.212, 2.197 / 2.322, 2.082 |
| 05 | One Adam step, change of final hidden state, L = 48 against L = 3: Post-LN / DeepNorm / Pre-LN | grows 22.3x / 3.0x / 6.5x |
| 06 | Softmax of 16 logits: Jacobian size at entropy 0.05 nats; at max p within 1e-3 of 1 | 7.2x below uniform; 63x below uniform |
| 06 | Pre-LN, 4 layers, lr 3e-2, final max abs logit and val loss: QK-norm off / on | 47,183, 2.670 / 18.3, 1.897 |
| 06 | Same at lr 1e-3, 3e-3, 1e-2, max abs logit: off / on | 30 / 7.3, 37 / 7.5, 70 / 8.8 |
| 07 | Peak update after 5,000 zero gradients, in units of lr: beta2 0.999 / 0.99 / 0.95 | 6.5 / 2.1 / 1.1 |
| 07 | Update size in the quiet stretch (mean `|update| / lr`): beta2 0.999 / 0.95 | 0.0019 / 0.183 |
| 07 | Cosine of clipped gradient with original: global-norm clip / per-coordinate clamp | 1.000000 / 0.59 |
| 08 | FP16 gradient values flushed to zero at S = 1 / 2^8; BF16 at S = 1 | 0.014% (25% subnormal) / 0 / 0 |
| 08 | Loss after 3,000 SGD steps (start 8.8e-5): BF16 only / BF16 + FP32 master / FP32 | 8.8e-5 (no movement) / 2.5e-5 / 2.7e-5 |
| 08 | Dynamic loss scale, 300 steps, start 2^32 | settles at 2^18, 16 steps skipped (5.3%) |
| 09 | Best log2 lr, width 64 to 2048: SP | -6, -7, -7, -7, -8, -8 |
| 09 | Best log2 lr, width 64 to 2048: muP | -6, -6, -6, -5, -5, -5 |
| 09 | Tune at width 64, reuse at width 2048: loss with SP / with muP / best at that width | 2.042 / 1.725 / 1.668 (SP) and 1.690 (muP) |

Two results disagree with a simple reading of the post, and the lab reports them as measured:

- Script 04 does not show warmup rescuing Post-LN. With Xavier init (the setting Xiong et al. analyse), Post-LN with a
  12-layer, d = 128 model never leaves the unigram level in 400 steps, with or without 100 warmup steps. With the
  post's default `N(0, 0.02)` init it trains without trouble. Warmup mattered only for Pre-LN at lr 3e-3 (2.322 to 2.082); elsewhere the change was 0.05 or less.
  This is one seed, one depth and 400 steps. It does not test whether longer warmup would change the Xavier result.
- Script 06 shows QK-norm bounding the logits, but not at `sqrt(d_h) = 4`. With learnable LayerNorm gains the max
  logit grew to 18 at lr 3e-2. The post's bound holds for unit gains only. QK-norm still prevented the collapse: the
  no-QK-norm run ended with a max logit near 47,000 and mean entropy near 7e-4 nats.

The muP shift of +1 grid step (a factor of 2) over a 32x width range is small but not zero. SP shifted by -2 steps
(4x). The muP transfer loss is 0.035 worse than the best at width 2048; the SP transfer loss is 0.374 worse. One seed.


## What was not measured

- Nothing at the post's scale. DeepNet's 1,000-layer and 200-layer translation models, the ViT-22B divergence, the
  Wortsman et al. sweep across model sizes, OPT/PaLM/GLM/BLOOM spikes, muTransfer to BERT-large and GPT-3 6.7B, and
  FP8 training were not run. Their numbers in the post are the papers' numbers.
- Part IX, the loss-spike decision tree, has no script here. C5 already injects bad batches, NaNs and loss spikes
  and recovers from them (`c05-pretraining/scripts/break_it.py`). Replaying a batch on an earlier checkpoint and
  per-rank loss logging were not built.
- z-loss, sigma-Reparam and QK-clip are described in the post but not implemented. Soft-capping and QK-norm are.
- Activation gradients inside the backward pass in FP16 (the Micikevicius et al. measurement) were not measured.
  Script 08 counts underflow in parameter gradients only, and its loss-scale check looks at parameter gradients,
  not at every intermediate of the backward pass.
- The Pre-LN "curse of depth" and massive activations need trained large models and were not measured.
- No GPU was used. Nothing in the lab depends on speed.
