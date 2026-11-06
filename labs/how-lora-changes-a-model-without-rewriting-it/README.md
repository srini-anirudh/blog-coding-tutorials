# Lab: How LoRA Changes a Model Without Rewriting It

Code companion to the theory post
[How LoRA Changes a Model Without Rewriting It](https://srini-anirudh.github.io/blogs/how-lora-changes-a-model-without-rewriting-it/).

The post says LoRA saves optimizer memory, not compute. It says a rank-r update is a bottleneck of r wires, that the
α/r scaling hides what high ranks can do, that NF4 spends its 16 codewords where Gaussian weights live, and that LoRA
learns less and forgets less than full fine-tuning. This lab builds each of those pieces from scratch in small PyTorch
files and runs them, so you can watch each claim happen at small scale.

The package `loralab/` holds a LoRA layer with every option the post discusses (four initialisations, α/r or α/√r,
DoRA), an NF4 quantizer with double quantization, the post's parameter ledger, and two weight-space diagnostics. The
LoRA, rsLoRA and DoRA layers are checked against Hugging Face PEFT. The NF4 codebook is built from normal quantiles and
checked against the published one.

Related tutorials: [C2](../../c02-hf-stack/) uses PEFT's LoRA through the HF stack, and
[C6](../../c06-post-training/) writes a minimal `LoRALinear` for SFT. This lab goes further into the mechanism:
initialisation, scaling, merging, quantization and the learn/forget trade-off.

## Layout

```
labs/how-lora-changes-a-model-without-rewriting-it/
├── loralab/
│   ├── lora.py       # LoRALinear (4 inits, alpha/r or alpha/sqrt(r), DoRA), merge, manual backward, multi-adapter batch
│   ├── quant.py      # NF4 from quantiles, blockwise absmax, double quantization, NF4Linear
│   ├── ledger.py     # exact parameter and memory arithmetic (Llama-2-7B, GPT-3 175B geometry)
│   ├── spectrum.py   # matrices with a chosen spectrum, Eckart-Young, LoRA-style factored search
│   ├── analysis.py   # DoRA's Delta M / Delta D, intruder dimensions
│   ├── data.py       # WikiText-2 and its reversed-word cipher
│   └── report.py     # results files that start with the hardware and the command
├── scripts/          # 01 to 07, listed below
├── tests/            # pytest
└── results/          # real outputs of every script (.txt and .json)
```

## Setup

From the repository root, after `pip install -e ".[dev]"`. Scripts 06 and 07 download SmolLM2-135M and WikiText-2 from
the Hugging Face Hub the first time. Everything else is offline.

## Run everything

```bash
cd labs/how-lora-changes-a-model-without-rewriting-it
python scripts/01_ledger.py          # memory and parameter ledger, ~30 s on CPU
python scripts/02_rank.py            # Eckart-Young and factored search, ~25 s
python scripts/03_lora_layer.py      # inits, backward formulas, saved activations, ~10 s
python scripts/04_scaling.py         # alpha/r vs alpha/sqrt(r) rank sweep, 3 seeds, ~4 min
python scripts/05_merge_serve.py     # merge, hot-swap drift, NF4 merge, mixed batch, ~5 s
python scripts/06_nf4.py             # NF4 vs uniform, blocks, double quant, SmolLM2-135M loss, ~10 min on CPU
python scripts/07_finetune.py --device cuda   # full vs LoRA vs DoRA vs QLoRA on SmolLM2-135M, one GPU

# Tests, from the repository root.
python -m pytest labs/how-lora-changes-a-model-without-rewriting-it -m "not slow and not gpu and not network"
python -m pytest labs/how-lora-changes-a-model-without-rewriting-it -m network
```

`07_finetune.py` also runs on CPU, but the default 600 steps for six configs takes hours there. For a quick CPU check:
`python scripts/07_finetune.py --configs lora8 --steps 20 --eval-rows 8 --name 07_cpu_check`.

## Which script covers which section of the post

| Post section | Script |
|---|---|
| The ledger nobody reads (Fig. 1) | `01_ledger.py` |
| Rank, from first principles (Fig. 2) | `02_rank.py` |
| LoRA: why B = 0 and A random; the backward pass (Fig. 3) | `03_lora_layer.py` |
| What this buys, in numbers (Fig. 4) | `01_ledger.py` |
| Scaling, initialization, and the α/r question | `04_scaling.py` |
| Which matrices: where the parameters live | `01_ledger.py` |
| Merging, swapping, serving (Fig. 6) | `05_merge_serve.py` |
| QLoRA: quantization, double quantization (Fig. 7) | `06_nf4.py` |
| QLoRA: the gradient path | `07_finetune.py` (`qlora8`), `tests/test_quant_ledger_spectrum.py` |
| DoRA: magnitude and direction (Fig. 8) | `07_finetune.py` (`dora8`, Delta M / Delta D) |
| PEFT versus full fine-tuning (Fig. 10) | `07_finetune.py` |

## Results

Every number names its hardware. Raw output is in `results/`; each file starts with the hardware and the command.
CPU = AMD EPYC 7R13, 8 threads, torch 2.15.0. GPU = NVIDIA H100 80GB HBM3, torch 2.15.0+cu132, CUDA 13.2.
Tests on AMD EPYC 7R13, 8 threads: 27 passed and 1 `network` test deselected in the offline run; the `network` test
passes (1 passed). There are no `slow` or `gpu` tests.

### Ledger (`01_ledger.py`, arithmetic plus one measured part; CPU)

| Quantity | Result |
|---|---|
| Llama-2-7B geometry, total parameters | 6.738 B |
| Full fine-tuning, 16 B per parameter | 112 GB for 7 B; the bf16 weights are 1/8 of it |
| Per layer, attention / MLP | 67.1 M / 135.3 M (MLP share 0.668) |
| LoRA r=8, Q,V only / all attention / all linear | 4.19 M / 8.39 M / 19.99 M trainable (0.062% / 0.124% / 0.297%) |
| Break-even rank, 4096 x 4096 | 2048 |
| Memory, r=16 all linear (no activations): full / LoRA / QLoRA | 107.81 / 14.12 / 4.12 GB |
| NF4 bits per parameter, single / double quantization | 4.500 / 4.127 (saves 0.373) |
| GPT-3 175B, Wq,Wv at r=4 | 18.87 M parameters; fp16 checkpoint 9272x smaller than the 350 GB model |
| Measured, SmolLM2-135M shapes, random weights, fp32: grads + Adam state, full / LoRA r=8 | 1614 MB / 29.3 MB, both 12 B per trainable parameter |
| Measured, checkpoint size, full / LoRA r=8 | 538 MB / 9.8 MB (55x smaller) |

The post's 10,000x checkpoint ratio for GPT-3 uses 35 MB; this lab's arithmetic gives 37.7 MB for Wq,Wv at r=4, hence 9272x.
The training-memory ratio (about 3.4x for GPT-3) is not recomputed here.

### Rank, the layer and the scaling trap (CPU)

| Measurement | Result |
|---|---|
| Truncation error equals the Eckart-Young tail (24 x 24, rank 12) | 22.64 vs 22.64 |
| Adam on B A (B=0 start) vs the Eckart-Young optimum | ratio 1.00 for 11 of 12 cases; the exception is rank 16 on the fast-decay 24 x 24 matrix, 4.96x worse (still tiny: 0.017 vs 0.0035) |
| Init: both random, max output change at step 0 | 2.47 (b_zero and a_zero: 0) |
| Init: both zero, loss after 300 steps | 1.707 (unchanged, dead); b_zero 1.2e-5 |
| First Adam step from B = 0, max change in A / B | 0 / 0.01 |
| Backward formulas vs autograd, max difference | 0.0 |
| Activations kept for backward, 1024 -> 1024, input 1.05 MB: trainable / frozen / frozen + LoRA r=8 | 1.049 / 0 / 1.057 MB |
| Rank sweep, full-rank target, share of reachable improvement done by step 200, rank 128: alpha/r / alpha/sqrt(r) | 85% / 97% |
| Same, rank 256 | 64% / 97% |
| Same, rank 2 or 8 | 98 to 99% for both rules |
| LoRA+ ratio 1 / 4 / 16, rank 32, error at step 50 | 4.062 / 3.619 / 3.519 (the three end within 0.01 of each other by step 1500; one seed) |

### Merge, swap, serve (`05_merge_serve.py`, CPU)

| Measurement | Result |
|---|---|
| Merged vs unmerged output, 1024 x 1024, r=16, fp32 | max difference 2.4e-6 |
| Merge/unmerge x2 adapters, 1000 cycles: relative drift of W, bf16 / fp32 | 0.35% / 6.7e-6 |
| Merge a LoRA update into an NF4 base, update 0.1% of ||W||: share kept | 1.5% |
| Same, update 5% of ||W|| / 20% of ||W|| | 28% / 98% |
| Mixed batch of 32 requests, 8 adapters, shared GEMM vs per-request merge | max difference 2.2e-6 |

The bf16 drift stops growing after about 100 cycles because rounding then reaches a fixed pattern; it does not keep growing.
The NF4 rows use the NF4 error of this layer's own weights (9.2% relative).

### Quantization (`06_nf4.py`, CPU)

| Measurement | Result |
|---|---|
| Built NF4 vs published codebook, max difference | 1.2e-7; zero is a codeword |
| Gaussian 4096 x 4096, relative error, blocks of 64: NF4 / uniform 16 | 0.0920 / 0.1005 |
| Same, one block for the whole tensor | 0.1397 / 0.2039 |
| With 0.1% outliers x20, blocks of 64: NF4 / uniform 16 | 0.1226 / 0.2097 |
| With 0.1% outliers x20, whole tensor: NF4 / uniform 16 | 0.8471 / 4.219 |
| Double quantization: stored bits per parameter | 4.5000 -> 4.1270; weight error 0.09199 -> 0.09200 |
| SmolLM2-135M, all linear weights, relative error: NF4 / uniform 16 | 0.0935 / 0.1054 |
| SmolLM2-135M, WikiText-2 test (64 rows x 512), loss: fp32 / NF4 / uniform 16 | 3.013 / 3.234 / 3.777 |

### Full fine-tuning vs LoRA vs DoRA vs QLoRA (`07_finetune.py`, NVIDIA H100 80GB HBM3, torch 2.15.0+cu132, CUDA 13.2)

SmolLM2-135M, 600 steps x batch 16 x 256 tokens (2.46 M tokens), fp32, one seed, 128 validation rows per domain.
Target domain: WikiText-2 with every word spelled backwards. Source domain: plain WikiText-2. Losses are nats per token.
Learning rate: 5e-5 for full fine-tuning, 5e-4 for every adapter. These were not tuned.

| Config | Trainable | Adam state MB | Peak GB | ms/step | Target loss before -> after | Source loss before -> after |
|---|---|---|---|---|---|---|
| full | 134.52 M | 1076 | 10.06 | 124 | 5.583 -> 2.685 | 3.113 -> 3.196 |
| LoRA r=8 | 2.44 M | 19.5 | 8.43 | 179 | 5.583 -> 2.992 | 3.113 -> 3.975 |
| LoRA r=64 | 19.54 M | 156.3 | 8.83 | 193 | 5.583 -> 2.661 | 3.113 -> 4.325 |
| LoRA r=256 | 78.15 M | 625.2 | 10.31 | 184 | 5.583 -> 2.521 | 3.113 -> 5.514 |
| DoRA r=8 | 2.60 M | 20.8 | 10.97 | 261 | 5.583 -> 2.976 | 3.113 -> 3.943 |
| QLoRA r=8 | 2.44 M | 19.5 | 8.55 | 214 | 5.719 -> 3.010 | 3.355 -> 4.333 |

What this run shows:

- Rank buys target loss. LoRA at r=8, 64 and 256 reaches 2.992, 2.661 and 2.521. The reversed-word task is new to
  the model, which is the post's "adding capability" case, and low rank is the slowest to learn it.
- At about the same target loss, full fine-tuning (2.685) and LoRA r=64 (2.661) differ in source loss by a wide margin:
  +0.08 for full, +1.21 for LoRA. In this run LoRA forgot more, not less. This is the opposite of the direction in
  Biderman et al. (2024) that the post quotes. The learning rates differ by 10x between full and adapters, and the post
  says the comparison is sensitive to the learning rate, so this run does not settle the question. It does not
  reproduce "LoRA forgets less". A run that sweeps the learning rate for both methods was not done.
- Optimizer state is exactly 12 B per trainable parameter in the table (19.5 MB for 2.44 M parameters), matching the ledger.
- Merged and unmerged target losses agree to 5 digits for LoRA and DoRA (e.g. LoRA r=8 2.99185 both ways).
- Intruder dimensions (top-10 left singular vectors with max |cos| < 0.5 against the pretrained matrix, mean per
  adapted matrix): full 0.00, LoRA r=8/64/256 0.12/0.20/0.28, DoRA 0.09, QLoRA 0.08. The direction matches
  Shuttleworth et al. (2024), but the counts are small and the run has one seed.
- Delta M vs Delta D (q and v projections, correlation over layers and checkpoints): full +0.31, LoRA r=8 +0.67,
  DoRA +0.47, QLoRA +0.61. The DoRA paper reports a negative relationship for full fine-tuning. This run did not
  reproduce that sign, and the setting differs from the paper's (small model, short run, new task).
- Step time and peak memory on this small model do not show LoRA's savings: full fine-tuning is the fastest per step
  (124 ms) and peak memory is within 10.06 to 10.97 GB for every config, because activations and the 49152-token
  logits dominate at batch 16 x 256. The ledger's 14 GB vs 108 GB gap needs a 7 B model. The python LoRA wrapper and
  DoRA's per-step norm also cost time. QLoRA is slower than LoRA, as the post says.
- QLoRA starts from a worse model (target 5.719 and source 3.355 before training vs 5.583 and 3.113), then trains to 3.010,
  close to bf16-style LoRA r=8 at 2.992.

## What was not measured

- The paper-scale results the post quotes: GPT-3 175B tables (Hu et al.), the QLoRA 65B claim, DoRA benchmarks,
  Biderman et al.'s 20 B-token runs, S-LoRA throughput and the GPT-2 adapter latency numbers. Only the mechanisms were
  rebuilt, at small scale.
- The fine-tuning comparison used one seed, one learning rate per method, and 2.46 M tokens. Differences of a few
  hundredths of a nat should not be read as real. The learning-rate sweep that would settle the forgetting comparison
  was not run.
- QLoRA here is simulated: weights are stored as NF4 codes but dequantized by a PyTorch function on every forward, not
  by the bitsandbytes kernel. Speed and memory of real QLoRA were not measured, and paged optimizers were not built.
- bf16 training and the memory of a 7 B model were not run. The 7 B numbers are arithmetic.
- AdaLoRA, PiSSA, MiLoRA, adapters, prefix and prompt tuning, IA3 and BitFit are not built. VLM adaptation is covered
  by [C8](../../c08-vlm-training/).
- The Figure 2 and Figure 8 matrices are synthetic here as well. Nothing in the lab measures a real fine-tuning delta's spectrum.

## Theory

- [How LoRA Changes a Model Without Rewriting It](https://srini-anirudh.github.io/blogs/how-lora-changes-a-model-without-rewriting-it/)
