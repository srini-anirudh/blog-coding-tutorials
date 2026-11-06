# Lab: Evolution of ML Architectures

Code companion for the post
[Evolution of ML Architectures: From Perceptrons to State-Space Models](https://srini-anirudh.github.io/blogs/evolution-of-ml-architectures/).

The post compares five ways to mix positions: the MLP, the convolution, the recurrence, attention and the
selective state space model. It compares them on three axes: path length, training parallelism and the state
carried at inference. This lab turns the post's central claims into small programs that run on a CPU and print
real numbers:

- The sizing arithmetic. Each closed-form count is checked against a real PyTorch or Hugging Face module.
  Llama-3-8B comes out at 8,030,261,248 parameters from six numbers.
- The products of Jacobians behind vanishing gradients in deep stacks, RNNs and LSTMs.
- Every backward-pass formula in the post, checked against autograd.
- The convolution's prior: a model trained on shapes in the left half of an image is tested on the right half.
- One long-range recall task for five mixers: RNN, LSTM, attention, a time-invariant SSM and a selective SSM.
- A fixed-size state against a KV cache, for recall and for decode cost.
- The scan: sequential, parallel, the convolution view and the Mamba-2 matrix view.

## Layout

```
labs/evolution-of-ml-architectures/
├── archlab/
│   ├── counts.py      # parameters, bytes, FLOPs; layer and CNN builders to check them against
│   ├── gradients.py   # Jacobian products: deep stacks, RNN, LSTM, norm placement
│   ├── vjp.py         # the post's backward formulas, written by hand
│   ├── scan.py        # sequential and parallel scans, LTI kernel, semiseparable matrix
│   ├── seqmodels.py   # RNN, LSTM, attention, LTI SSM and selective SSM mixers; the recall task
│   ├── vision.py      # MLP, CNN, tiny ViT; the shifted-shapes task; receptive field
│   └── env.py         # threads, seeds, and the hardware line at the top of every results file
├── scripts/           # one script per claim, listed below
├── results/           # what the scripts printed
└── tests/
```

## Setup

From the repository root, after `pip install -e ".[dev]"`. Nothing is downloaded: the Hugging Face models in
`01_count_params.py` are built on the `meta` device from their configs, so no weights are fetched or allocated.
Set `OMP_NUM_THREADS` on a shared machine.

## Run everything

```bash
python -m pytest labs/evolution-of-ml-architectures -m "not slow and not gpu and not network"

python labs/evolution-of-ml-architectures/scripts/01_count_params.py       # seconds
python labs/evolution-of-ml-architectures/scripts/02_image_priors.py       # a few minutes
python labs/evolution-of-ml-architectures/scripts/03_gradient_paths.py     # seconds
python labs/evolution-of-ml-architectures/scripts/04_long_range_memory.py  # the long one, see below
python labs/evolution-of-ml-architectures/scripts/05_state_vs_cache.py     # under a minute
python labs/evolution-of-ml-architectures/scripts/06_scans.py              # seconds
python labs/evolution-of-ml-architectures/scripts/07_backward_derived.py   # seconds
python labs/evolution-of-ml-architectures/scripts/08_training_cost.py      # under a minute
```

`04_long_range_memory.py` trains 15 small models. The two SSMs use a Python-level scan, so they are the slow
part. The results in this folder came from running the three distances as three processes at once, then a longer
run of the two models that had not yet converged:

```bash
for f in 16 64 128; do
  python labs/evolution-of-ml-architectures/scripts/04_long_range_memory.py --fillers $f --tag _f$f &
done; wait
for f in 64 128; do
  python labs/evolution-of-ml-architectures/scripts/04_long_range_memory.py --fillers $f \
    --models lstm selective_ssm --steps 5000 --tag _f${f}_5000steps &
done; wait
```

`--quick` runs a 60-step smoke test at distance 8.

## Which script covers which section of the post

| Post section | Script |
|---|---|
| Three axes that define an architecture | `04_long_range_memory.py`, `05_state_vs_cache.py`, `08_training_cost.py` |
| The dense layer; why the MLP is universal and useless on its own | `02_image_priors.py` (part 1), `01_count_params.py` |
| Convolution: equivariance, invariance, receptive field | `02_image_priors.py` |
| The depth problem and the residual fix | `03_gradient_paths.py` (part 1) |
| Recurrence: the gradient is a matrix power; the sequential wall | `03_gradient_paths.py` (part 2), `08_training_cost.py` (part 2) |
| Gates: LSTM, GRU and the constant error carousel | `03_gradient_paths.py` (part 3), `04_long_range_memory.py` |
| Attention: why the 1/sqrt(d_k) | `07_backward_derived.py` |
| The Transformer block; which term dominates at which N/d | `01_count_params.py` (part 6), `08_training_cost.py` (part 1) |
| Inference economics: the KV cache | `01_count_params.py` (part 5), `05_state_vs_cache.py` (part 2) |
| Linearization: linear attention is an RNN | `06_scans.py` (part 5, reuses C3's code), `05_state_vs_cache.py` (part 1) |
| State spaces: HiPPO, S4 and Mamba; the selectivity problem | `06_scans.py`, `04_long_range_memory.py` |
| Hybrids: a fixed state cannot store N tokens losslessly | `05_state_vs_cache.py` (part 1) |
| How big is a model, really? | `01_count_params.py` |
| The backward pass, derived | `07_backward_derived.py` |
| Lab: the same image through three architectures | `02_image_priors.py` |
| Lab: the same sentence through four architectures | `04_long_range_memory.py` |
| Lab: the complexity explorer | `01_count_params.py` (part 6), `08_training_cost.py` |

Two mechanisms are already built in the tutorials, so this lab links to them instead of rebuilding them.
FlashAttention's tiling and online softmax are in [C3](../../c03-attention/). The full Llama-style decoder,
GQA and the KV cache are in [C4](../../c04-llm/). `01_count_params.py` uses C4's `count_parameters` as a third
check on Llama-3-8B, and `06_scans.py` uses C3's recurrent linear attention.

## Results

Every number names its hardware. Raw output is in `results/`, one file per script, each starting with the
hardware line and the command. All runs were on AMD EPYC 7R13 CPUs with torch 2.15.0, on machines shared with
other jobs, so timings are orders of magnitude only. Counts, errors and accuracies do not depend on load.

Tests on AMD EPYC 7R13, torch 2.15.0: 30 passed (offline, `-m "not slow and not gpu and not network"`, 17 seconds); see `results/tests.txt`. There are no `network`, `slow` or `gpu` tests.

### Arithmetic (exact, no hardware dependence)

| Measurement | Result |
|---|---|
| Llama-3-8B: formula / HF `LlamaForCausalLM` / C4 `count_parameters` | 8,030,261,248 / same / same |
| Llama-3-70B by the same formula | 70.55 B |
| Parameters per layer at d = 1024: MLP / conv 3x3 / Elman / GRU / LSTM / Transformer / Mamba | 1.05 / 9.44 / 2.10 / 6.29 / 8.39 / 12.59 / 6.67 M |
| Mamba mixer beyond its 6 d^2 projections (HF `MambaMixer`, d = 1024, state 16) | +6.0% |
| GPT-2 small: total / token embedding | 124.4 M / 38.6 M (31%) |
| AlexNet (2012 widths, ungrouped): conv / dense | 3.75 M / 58.63 M |
| VGG-16: total / dense | 138.36 M / 123.64 M |
| Llama-3-8B: bf16 weights / 16 B-per-parameter training state | 16.1 GB / 128.5 GB |
| Llama-3-8B KV cache per token / at 8k / at 128k | 128 KiB / 1.0 GiB / 16.0 GiB |
| Llama-3-70B KV cache at 128,000 tokens: MHA / GQA | 335.5 GB / 41.9 GB |
| Attention / matmul FLOPs at d = 4096: N = 4k / 24k / 128k | 0.17 / 1.00 / 5.33 |

### Gradients and backward formulas (AMD EPYC 7R13; float64 for the formula checks)

| Measurement | Result |
|---|---|
| Hand-written VJPs vs autograd (19 formulas: linear, 5 activations, softmax, CE, LayerNorm, RMSNorm, conv, BPTT, LSTM, attention, FlashAttention row term, SSM scan, Delta chain, MoE router) | largest error 1.1e-14 |
| 50-layer stack, input-gradient ratio: sigmoid plain / sigmoid residual | 0 (underflow) / 1.14 |
| 20-layer stack: tanh plain / ReLU plain | 1.2e-5 / 1.8e-8 |
| Linear RNN, sigma_max(W_h) = 0.9: gradient at distance 100 | 2.66e-5 (= 0.9^100) |
| tanh RNN, sigma_max = 1.0: gradient at distance 100 | 4.5e-12 |
| LSTM cell path at distance 100, forget bias 0 / +1 / +3 | 5.6e-30 / 2.6e-14 / 1.7e-2 |
| Logit std for unit-variance q, k at d_k = 128: unscaled / scaled | 11.52 / 1.02 |
| Trace of the softmax Jacobian, 64 keys: unscaled / scaled | 0.20 / 0.96 |

### Image priors (`02_image_priors.py`)

Trained on shapes in columns 0 to 5, tested on the same columns and on unseen columns 6 to 11. 1500 steps, 3 seeds,
chance is 0.25. AMD EPYC 7R13, 8 threads, torch 2.15.0.

| Model | Parameters | Accuracy, training columns | Accuracy, unseen columns |
|---|---|---|---|
| MLP (256 to 128 hidden) | 33,412 | 1.000 | 0.277 |
| CNN (three 3x3 convolutions, global max pool) | 14,180 | 1.000 | 1.000 |
| Tiny ViT (4x4 patches, 2 layers) | 18,276 | 0.998 | 0.993 |

After moving the object 4 pixels right, the pooled features of the MLP change by 97% and its prediction stays the
same for 29% of images. The CNN's features change by 0.3% and its prediction always stays. The CNN feature map
moves with the object exactly: the difference from the shifted map is 0 inside the borders, and 0 everywhere with
circular padding. Receptive field of L stacked 3x3 convolutions measured by gradient: 3, 5, 7, 9, 11 columns,
which is 1 + 2L.

The ViT result does not match the post's story. The post says a ViT must learn translation robustness from data.
This ViT shares one patch embedding across positions, averages its tokens, and starts with small position
embeddings, so it generalised to unseen positions anyway. Whether a ViT with a class token and large
position embeddings fails here was not tested.

### Long-range recall (`04_long_range_memory.py`)

One seed per cell. Chance is 0.062. Width 64, 2 layers, AdamW lr 2e-3 one-cycle, batch 32. AMD EPYC 7R13,
8 threads, torch 2.15.0. The runs shared the machine with each other and with other jobs, so `ms/step` is a rough
guide only. Accuracy is on 2,048 fresh sequences.

1000 training steps:

| Model | Parameters | 16 filler tokens | 64 filler tokens | 128 filler tokens |
|---|---|---|---|---|
| RNN | 19,216 | 0.064 | 0.074 | 0.057 |
| LSTM | 69,136 | 1.000 | 0.068 | 0.056 |
| Attention | 36,560 to 43,728 | 1.000 | 1.000 | 1.000 |
| Time-invariant SSM | 59,664 | 0.249 | 0.071 | 0.057 |
| Selective SSM | 61,712 | 1.000 | 0.074 | 0.058 |

5000 training steps, LSTM and selective SSM only (`--steps 5000`, files ending `_5000steps`):

| Model | 64 filler tokens | 128 filler tokens |
|---|---|---|
| LSTM | 0.059 | 0.066 |
| Selective SSM | 1.000 | 1.000 |

In the 5000-step selective SSM runs, the mean learned step size Delta (layer 1, averaged over channels) is
1.544 at the key against 0.083 at the filler for 64 filler tokens, and 0.363 against 0.048 for 128. That is the
post's "the gate opens on the key and nearly closes on the filler". In the 1000-step run at 16 filler tokens,
where the model also reached 1.000, the two values were about equal (0.020 against 0.019), so a correct answer
does not by itself mean the gate has separated.

What this shows and what it does not. The RNN never learns the task, and neither does the LSTM at 64 tokens
within 5000 steps. Attention learns at every distance with no sign of difficulty. The selective SSM needs more
steps than attention but learns it. The time-invariant SSM did not solve it at 1000 steps (0.249 at 16 tokens);
the task places the key at a fixed offset from the end, so a fixed kernel could solve it in principle, and a
longer run might have. The LSTM at 64 and 128 tokens is a failure at this budget, not a proof it cannot learn:
the post says gating makes long dependencies possible, and this lab did not tune the LSTM (no larger learning
rate sweep, no longer schedule than 5000 steps). Models are tiny, runs use one seed, and no model was tuned.

### State against cache (`05_state_vs_cache.py`, `06_scans.py`, `08_training_cost.py`)

| Measurement | Result |
|---|---|
| Recall from a 64 x 64 fixed state: n = 64 / 128 / 256 / 512 / 1024 pairs | 1.000 / 0.984 / 0.770 / 0.342 / 0.106 |
| Recall from the KV cache with softmax, same pairs | 1.000 at every n |
| Recall of 256 pairs, fixed state 32^2 / 64^2 / 128^2 | 0.148 / 0.770 / 1.000 |
| Decode one token, AMD EPYC 7R13, 4 threads: attention at 1k / 64k context | 0.21 / 29.9 ms |
| Decode one token, same machine: SSM step (state 2048 x 16) | 0.037 ms at any context |
| Sequential vs parallel scan, N = 4096: max difference / rounds | 1.4e-6 / 12 instead of 4096 |
| LTI recurrence vs FFT convolution | 2.9e-6 |
| Mamba-2: scan vs semiseparable matrix M x | 1.3e-5 |
| Attention core / FFN forward time, d = 256, AMD EPYC 7R13, 4 threads: N = 256 / 1024 / 4096 | 0.39 / 1.19 / 10.1 |
| nn.RNN over 4096 steps vs one matmul of equal work, same machine | 8x slower (7.8 to 8.7x at N = 256 to 4096) |

## Where the numbers differ from the post

- The recurrence section says that at lambda = 0.9 a dependency 100 steps back arrives at 0.0027 of its strength.
  0.9^100 is 2.66e-5, and the linear-RNN measurement matches that. 0.0027 is 0.9^56.
- The sizing section says that raising the vocabulary from 32k to 256k at d = 4096 adds 1.7 B parameters. The
  arithmetic gives 0.92 B with tied embeddings and 1.84 B untied.
- The sizing section says the embedding table is under 1% of a 70B model. For Llama-3-70B the input embedding is
  1.49% and embedding plus untied head is 3.0%.
- The Mamba block is "roughly 6 d^2". With the Hugging Face mixer at d = 1024 the extra parts add 6.0%, so the
  block is 6.36 d^2.

## What was not measured

- No GPU was used. The post's GPU claims were not measured: decode is memory-bandwidth bound on an H100,
  FlashAttention is 2 to 4x faster, Mamba-2 is 2 to 8x faster than Mamba-1. The CPU decode timings show the
  trend in cost, not the bandwidth argument.
- No paper result at scale was reproduced. That includes ImageNet numbers for VMamba, Swin and ConvNeXt; Long
  Range Arena; Jamba, Samba and Griffin; Mixtral's active parameters; and the Zoology recall curves. Part 1 of
  `05_state_vs_cache.py` shows the same state-size trade at toy scale, with a linear-attention state and no
  training.
- Pre-LN against post-LN: at initialisation, a 24-block toy shows flat gradients for both placements. The
  post's claim is about training without warmup, and that run is not in this lab.
- Vision SSM scan orders (Vim, VMamba's four-way scan) are not built.
- The training runs in `04_long_range_memory.py` use one seed per cell, 1000 steps (5000 for the second table),
  and no tuning. They show which mixers learned the task at that budget, not the best each could reach.
- The recall script has a `--device cuda` option. It was never run: the shared GPUs were busy and the lab is not
  about GPU speed. The CPU results above are the only ones.
- `ms/step` in the recall tables were measured while several jobs shared the machine. Do not compare them.
