# Lab: The Geometry of Normalization

Runnable code for the post
[The Geometry of Normalization: BatchNorm, LayerNorm & RMSNorm](https://srini-anirudh.github.io/blogs/the-geometry-of-normalization/).

The post argues that every normalization layer answers one question: which numbers do I average over, and what do I
do with the answer? This lab builds the operators from one function and checks the post's central claims with
numbers. It covers the axis atlas, the two invariances, the three-term gradient, and weight decay as a learning-rate
knob. It also covers BatchNorm's noise at small batch, norm placement and depth, QK-Norm, and the VLM interface.
Most of it is small synthetic experiments. Two parts use small public checkpoints: SmolLM2-135M and SmolVLM-256M.

Related tutorials in this repo: `c04-llm/tinyllm/norm.py` is the RMSNorm used in the C4 model, and
`c03-attention/scripts/stability_demo.py` shows QK-Norm and soft-capping bounding attention logits. This lab does not
repeat those. It adds the parts that are specific to this post.

## Layout

```
labs/the-geometry-of-normalization/
├── normgeo/
│   ├── ops.py        # normalize(x, dims), the atlas, hand-derived backward passes, RMSNorm options, DyT, decay groups
│   ├── blocks.py     # a tiny causal transformer with six norm placements: post, pre, peri, lns, deepnorm, dyt
│   ├── geometry.py   # RoPE, the rotation-angle formula, a gradient-scaling hook
│   └── report.py     # writes results/<script>.txt with a hardware line and the command
├── scripts/          # 01 to 10, one per group of claims (table below)
├── tests/
└── results/          # real outputs of every script
```

## Setup

From the repository root, after `pip install -e ".[dev]"`. Script 06 reads `c04-llm/data/tinyshakespeare.txt`
from this repo. Scripts 05 and 10 download CIFAR-10, SmolLM2-135M and SmolVLM-256M-Instruct from the Hugging Face Hub
on first use (about 0.9 GB). Set `OMP_NUM_THREADS` on a shared machine. The numbers below used 8 threads.

## Run everything

```bash
# Tests. The offline set takes about a minute on 8 threads (30 tests).
python -m pytest labs/the-geometry-of-normalization -m "not slow and not gpu and not network"
python -m pytest labs/the-geometry-of-normalization -m "slow or network"     # 2 tests, about a minute; they download

# Scripts. Each prints a table and writes results/<name>.txt (and .json where useful).
python labs/the-geometry-of-normalization/scripts/01_axis_atlas.py      # seconds
python labs/the-geometry-of-normalization/scripts/02_anatomy.py         # seconds
python labs/the-geometry-of-normalization/scripts/03_effective_lr.py    # about 10 s
python labs/the-geometry-of-normalization/scripts/04_batchnorm.py       # seconds
python labs/the-geometry-of-normalization/scripts/05_small_batch.py     # about 7 min, CIFAR-10 download
python labs/the-geometry-of-normalization/scripts/06_depth.py           # about 10 min when two settings run side by side
python labs/the-geometry-of-normalization/scripts/06_depth.py --lr 1e-2 --warmup 0 --tag hot   # same, harsher schedule
python labs/the-geometry-of-normalization/scripts/07_qk_norm.py         # seconds
python labs/the-geometry-of-normalization/scripts/08_llm_nuances.py     # about 15 s
python labs/the-geometry-of-normalization/scripts/09_interface.py       # seconds
python labs/the-geometry-of-normalization/scripts/10_real_models.py     # about 10 min, downloads two models
```

No script needs a GPU. Nothing in the post is about GPU speed that this lab could measure at small scale, so no GPU
runs were made.

## Which script covers which section

| Post section | Script | What it shows |
|---|---|---|
| §01 Story 3: scale invariance and the effective learning rate (Fig 01) | `03_effective_lr.py` | Forward unchanged under W -> kW, gradient scales as 1/k, angular step as 1/k². Weight decay sets the equilibrium step |
| §02 The axis atlas (Fig 02) | `01_axis_atlas.py` | Five operators from one function, checked against torch. Only BatchNorm mixes samples |
| §03 Anatomy, invariances and the gradient (Fig 03) | `02_anatomy.py` | Re-scaling and re-centring invariance, the three-term backward, the two projections |
| §04 BatchNorm and its discontents (Fig 04) | `04_batchnorm.py`, `05_small_batch.py` | Estimator noise, train/eval gap, padding, small-batch CIFAR-10 |
| §05 LayerNorm and RMSNorm geometry (Fig 05) | `02_anatomy.py` part 4 and 5 | The circle in the plane vs the full sphere, rank D-1 vs D, eager CPU timing |
| §07 Placement, §08 the curse of depth (Figs 06, 07) | `06_depth.py` | Stream RMS, relative update, gradient ratio and layer-removal cost for six placements |
| §08 and §11 item 6 in a real model (Fig 09) | `10_real_models.py` part A | Residual norms, massive activations and layer-removal cost in SmolLM2-135M |
| §09 QK-Norm and the softmax (Fig 08) | `07_qk_norm.py` | Entropy collapse and gradient decay without QK-Norm. RoPE and RMSNorm ordering |
| §10 Living without normalization | `06_depth.py` (`dyt`) | DyT has no per-token rescaling, so it does not control the stream's scale |
| §11 Nuances that bite in LLMs | `08_llm_nuances.py` | eps placement, bf16 reductions, Gemma's (1 + w), decay groups, folding, the redundant pre-norm under QK-Norm |
| §12 The modality interface (Figs 10, 11) | `09_interface.py`, `10_real_models.py` parts B and C | Representational inertia, the small-gain post-projector norm and its gradient hook, SmolVLM's norm gap |

## Results

Every number below was measured on AMD EPYC 7R13, 8 threads, torch 2.15.0+cu132, CPU only. Raw output is in
`results/`, under the name of the script. The machine was shared with other jobs, so the one timing (02, part 5)
is an order of magnitude only. Nothing else depends on load.

### Operators, invariances and gradients (scripts 01, 02)

| Measurement | Result |
|---|---|
| Our `normalize(x, dims)` vs torch BatchNorm, LayerNorm, InstanceNorm, GroupNorm, RMSNorm | largest difference 4.8e-07 (fp32) |
| Output of sample 0 when only sample 1 is changed | BatchNorm moves by 2.14. The other four move by 0 |
| Max change after x -> 7.3x (fp64, eps 1e-5): LayerNorm / RMSNorm | 8.2e-06 / 8.0e-06 |
| Max change after x -> x + 3 (fp64): LayerNorm / RMSNorm | 1.3e-15 / 3.0 |
| Hand-derived backward vs autograd (fp64): LayerNorm / RMSNorm | 8.9e-16 / 8.9e-16 |
| Mean cosine of the input-gradient with the ones vector, after LayerNorm backward / RMSNorm backward | 2e-17 / 0.10 |
| Mean cosine with x-hat after backward, both operators | about 3e-07 |
| 3D cloud, eps 1e-12: LayerNorm / RMSNorm | norm exactly sqrt 3 for both. x+y+z = 0 for LayerNorm only |
| Rank of 4096 outputs in 16 dims: LayerNorm / RMSNorm / input | 15 / 16 / 16 |
| Forward + backward, 2048 x 1024 fp32, eager torch kernels: LayerNorm / RMSNorm | 4.13 / 4.83 ms (ratio 1.17, shared machine) |

The timing row does not support "RMSNorm is cheaper" for torch's eager CPU kernels. The post's claim is about fused
kernels, which this lab did not measure.

### Scale invariance and the effective learning rate (script 03)

Model: a fixed head on top of `LayerNorm(Wx)`, only W trained, CPU, fp32, seed 0.

| Measurement | Result |
|---|---|
| Loss at W scaled by k = 0.25, 0.5, 1, 2, 4 | 1.5978 for every k |
| k times the gradient norm at those k | 0.578 for every k |
| k squared times the angular step of one SGD step | 0.0275 to 0.0303 across the five k |
| Cosine between the gradient and W | below 5e-05 |
| Check of ‖W'‖² = ‖W‖² + lr² ‖g‖² over 3000 SGD steps, no decay | largest relative error 2.6e-05 |
| 3000 SGD steps, no weight decay: ‖W‖ 3.94 -> 5.86, mean angular step first 200 / last 200 steps | 2.07e-02 / 1.03e-02 |
| Weight decay 1e-3: ‖W‖ end, last-200 angular step, predicted sqrt(2 lr wd) | 4.47, 1.81e-02, 2.00e-02 |
| Weight decay 1e-2: ‖W‖ end, last-200 angular step, predicted sqrt(2 lr wd) | 2.66, 5.77e-02, 6.32e-02 |

### BatchNorm (scripts 04 and 05)

| Measurement | Result |
|---|---|
| Std of the batch mean at batch size 2 / 8 / 32 / 512, population std 3 | 2.115 / 1.062 / 0.530 / 0.133 (theory 3/sqrt(B): 2.121 / 1.061 / 0.530 / 0.133) |
| Mean absolute gap between train-mode and eval-mode outputs, batch 2 / 8 / 32 / 512 | 0.718 / 0.324 / 0.157 / 0.039 |
| Error of the running mean estimate at batch 2 / 512 | 0.582 / 0.016 |
| Padded batch of 8 sequences, 59% padding: BatchNorm mean with pads counted vs masked, true mean 1.0 | 0.413 vs 1.016 |
| Real sequences contributing at position 0 / 16 / 32 / 63 | 8 / 4 / 2 / 1 |

CIFAR-10 subset (10,000 train, 2,000 test images, 2 epochs, one seed, small CNN, `results/05_small_batch.txt`):

| Norm | Batch | Test accuracy (eval) | Test accuracy, test-batch statistics |
|---|---|---|---|
| BatchNorm | 2 | 0.407 | 0.445 |
| BatchNorm | 8 | 0.485 | 0.508 |
| BatchNorm | 64 | 0.435 | 0.489 |
| GroupNorm | 2 | 0.433 | not applicable |
| GroupNorm | 8 | 0.413 | not applicable |
| GroupNorm | 64 | 0.288 | not applicable |
| No norm | 2 | 0.391 | not applicable |
| No norm | 8 | 0.408 | not applicable |
| No norm | 64 | 0.363 | not applicable |

With 2,000 test images the standard error is about 1.1 points, and there is one seed and two epochs. The runs do
not show BatchNorm failing at small batch: batch 2 is not clearly worse than batch 64. Learning rate is tied to
batch size, so batch and step count change together. Read this table as a smoke test of the code path, not as
evidence for or against the post.

### Placement and depth (script 06)

A 24-block, width-64 character model on Tiny Shakespeare, 400 AdamW steps, seed 0, one run per placement. Columns
are the RMS of the residual stream after block 24, the mean relative update of the last six blocks, the validation
loss, and the mean validation-loss increase when one block is removed (average over the first 8 / last 8 blocks).
Setting A is lr 2e-3 with 50 warmup steps (`results/06_depth.txt`). Setting B is lr 1e-2 with no warmup
(`results/06_depth_hot.txt`).

| Placement | A: stream RMS | A: late update | A: val loss | A: drop cost first 8 / last 8 | B: val loss | B: drop cost first 8 / last 8 |
|---|---|---|---|---|---|---|
| post | 1.04 | 0.532 | 3.080 | 0.000 / 0.033 | 3.348 | 0.000 / 0.026 |
| pre | 7.11 | 0.102 | 2.250 | 0.195 / 0.003 | 2.493 | 0.273 / 0.006 |
| peri | 15.07 | 0.117 | 2.238 | 0.311 / 0.005 | 2.747 | 0.131 / 0.011 |
| lns (LayerNorm Scaling) | 2.88 | 0.031 | 2.219 | 0.232 / 0.000 | 2.369 | 0.354 / 0.009 |
| deepnorm | 1.24 | 0.144 | 2.074 | 0.034 / 0.013 | 3.276 | 0.000 / 0.035 |
| dyt | 86.70 | 0.038 | 3.346 | 0.000 / 0.000 | 3.346 | 0.000 / 0.000 |

What this does and does not show:

- At init, the Pre-LN stream RMS grows from 0.03 (embeddings) to 1.72 over 24 blocks. After training it grows to 7.11
  in setting A and 31.8 in setting B. Post-LN stays at about 1.
- In Pre-LN the last 8 blocks barely matter after training (drop cost 0.003 against 0.195 for the first 8). This is the
  curse-of-depth pattern from §08, at toy scale.
- LayerNorm Scaling lowered the stream growth (2.88 against 7.11) and gave the best Pre-LN-family loss in both
  settings. It did not make the late blocks more useful here (drop cost 0.000 / 0.009). The paper's gain from using
  deep layers was not reproduced.
- Post-LN had the highest loss among the normalized placements in both settings. Its gradient ratio between block 1
  and block 24 was 1.2e-05 in setting A, so the early blocks received almost no gradient. This matches the post's
  Post-LN story, but it is one seed.
- DeepNorm trained best in setting A (2.074) and poorly in setting B (3.276) without warmup. We did not tune it.
- DyT stayed at 3.346 in both settings, far above the 2.1 to 2.5 of the Pre-LN family, so it barely learned. It was not tuned (alpha starts at 0.5,
  no sweep), and the post says DyT needed alpha tuning for large models. This run shows only that its stream RMS
  grows to 87 in setting A and 250 in setting B, because it has no per-token rescaling. It is not evidence that DyT
  fails in general.
- The ratio of gradient norms (`grad ratio`) in the files is block 1 over block 24 on one batch.

### QK-Norm (script 07)

8 heads, 256 keys, head dim 64, random q and k scaled by s.

| s | Max logit, no norm / QK-Norm | Entropy, no norm / QK-Norm | Norm of dL/dS, no norm / QK-Norm |
|---|---|---|---|
| 1 | 5.3 / 4.3 | 5.05 / 5.06 | 36.1 / 35.4 |
| 4 | 85.0 / 4.3 | 0.33 / 5.06 | 83.0 / 35.4 |
| 32 | 5441.9 / 4.3 | 0.00 / 5.06 | 9.0 / 35.4 |

Uniform attention over 256 keys has entropy 5.55. RoPE keeps the norm (error 3.8e-06). `RMSNorm(RoPE(q))` equals
`RoPE(RMSNorm(q))` to 7.2e-07 with unit gain. With a random per-channel gain the two orders differ by up to 2.7.

### Details that bite (script 08)

| Measurement | Result |
|---|---|
| eps inside vs outside the sqrt, eps 1e-6, max relative difference at input RMS 1e-4 / 1e-2 / 1 | 0.91 / 5.8e-03 / 6.9e-07 |
| bf16 RMSNorm of 4096-dim vectors with std 100, max relative error against fp64: upcast / no upcast | 6.9e-03 / 1.06e-02 |
| `mean(x²)` as a plain bf16 running sum | 4768 against a true 10290 (54% off) |
| Gemma-style (1 + w) gain loaded into a plain-gain RMSNorm: output RMS, relative error | 0.050 against 1.004, error 0.996 |
| Decay only, 2000 AdamW steps from gain 0.5: plain gain / (1 + w) gain | 0.068 / 0.932 |
| Folding the gain into the next linear layer, fp64 | difference 4.2e-07, with or without the division after the GEMM |
| QK-Norm, pre-norm deleted and its gain folded into W_q (fp64) | difference 1.6e-07 on q. The v path differs by 2.01 relative |

The running-sum row is a worst case that a naive kernel could hit. Torch's CPU bf16 mean accumulates at higher
precision internally, so its "no upcast" row hardly differs from the upcast row. Do not read the first two rows as
a measured bf16 failure.

### Interface (script 09) and real models (script 10)

| Measurement | Result |
|---|---|
| Rotation by an update of norm 1 at 90 degrees, formula vs measured, ‖h‖ = 0.8 / 5 / 29 / 72 | 51.34 / 11.31 / 1.975 / 0.796 degrees. The formula agrees to 3 decimals |
| One random Pre-LN MLP block, update norm at every ‖h‖ | 4.70. Rotation: 80.4 / 43.2 / 9.2 / 3.7 degrees at ‖h‖ = 0.8 / 5 / 29 / 72 |
| Post-projector LayerNorm with gain T/sqrt(D): visual norm, encoder gradient norm | 0.80 (1.0x text), 1.02. With the backward hook: 0.80, 30.56. Bare projector: 41.4, 52.8 |
| SmolLM2-135M on 2048 Shakespeare tokens, fp32: loss | 3.269 |
| Residual stream mean norm after layer 0 / 12 / 24 / 27 | 2.78 / 143 / 381 / 565. After the final RMSNorm: 46.7 |
| Largest activation | 20,063 at (position 0, channel 507) from layer 12 on. The median absolute value is 1.6 to 8.5 |
| Loss increase when removing one layer, mean over layers 0-9 / 10-19 / 20-29 | 1.316 / 0.358 / 0.559 |
| Loss increase from removing layer 0 / 1 / 11 / 28 / 29 | +7.02 / +3.37 / +2.32 / +1.02 / +2.55 |
| SmolVLM-256M-Instruct, 16 CIFAR-10 images: vision tower token norm, after connector, text embedding norm | 15.4, 122.9, 2.72. Ratio visual / text after the connector: 45x |
| SmolVLM embedding table vs SmolLM2-135M: median cosine of shared rows, mean row norm | 0.992, 3.107 against 3.179 |
| SmolVLM image special tokens (`<image>`, `<fake_token_around_image>`, `<global-img>`) | norms 0.76 / 0.97 / 0.95, among the smallest 30 of 49,280 rows |

What this shows:

- The gap the post describes appears in a real small VLM: after the connector the visual tokens are 45 times the text
  embeddings, a gap in the post's 30 to 90 times range. This is one model and 16 low-resolution images.
- In SmolLM2-135M a massive activation sits at channel 507, first at position 8 (layers 3 to 9) and from layer 12 on at
  position 0, at about 20,000 against a median of 1 to 8. The stream grows steadily with depth until the final
  RMSNorm. Removing layers 0 and 1, layer 11 or the last two layers hurts most. The middle layers cost much less.
- The post reports that Qwen2-VL's vision boundary tokens became the largest vectors in the vocabulary. SmolVLM's
  image tokens did the opposite: they are among the smallest rows. The pattern did not transfer to this model.
- The embedding table was nearly unchanged by vision-language training in SmolVLM (cosine 0.992), unlike the
  inflation the post reports for Qwen2.5-VL. We did not check whether SmolVLM's backbone is initialised from
  SmolLM2-135M, so the comparison is between two public checkpoints and nothing more.

## What was not measured

- No paper-scale result was reproduced. The post quotes results at 130M to 130B parameters (LayerNorm Scaling's
  LLaMA-130M variance of about 175, OLMo 2 and Chameleon divergences, GLM-130B, Peri-LN at 3.2B, DyT across many
  domains, nGPT's 4-20x). This lab shows the mechanisms at toy scale only.
- The VLM interface numbers in the post (CLIP, SigLIP, Qwen2.5-VL, LLaVA-1.5, Kimi-VL, GLM-4.1V, and the Qwen2-VL
  boundary tokens) were not re-measured. Script 10 measures the same quantities in one different, much smaller model.
  The post-projector norm fix was not trained end to end. Script 09 shows only its forward scale and its gradient.
- Fused norm kernels, bf16 training runs, z-loss, sequence parallelism, FSDP and activation checkpointing were not run.
- Santurkar et al.'s noise-injection experiment (§01 Story 1) and the loss-landscape smoothness claim (Story 2) were
  not reproduced.
- The 28-operator zoo (§06) is not built. The lab covers BatchNorm, LayerNorm, InstanceNorm, GroupNorm, RMSNorm and
  DyT, plus six placements.
