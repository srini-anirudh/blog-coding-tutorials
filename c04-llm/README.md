# C4: Build a Modern LLM From Scratch

Code for the tutorial [Build a Modern LLM From Scratch](https://srini-anirudh.github.io/blogs/build-a-modern-llm-from-scratch/) in the Coding Tutorials series.

## What you build

`tinyllm` is a small decoder-only language model with the same architecture as Llama:

- token embeddings with optional weight tying
- RMSNorm
- RoPE
- grouped-query attention (GQA), with a readable reference version and a fast SDPA version
- SwiGLU
- pre-norm decoder blocks, a final norm and an LM head
- shifted cross-entropy loss and depth-scaled initialization
- greedy, temperature, top-k and top-p sampling
- a KV cache, and generation with or without it
- an optional mixture-of-experts block
- a native checkpoint format, plus conversion to and from Hugging Face `LlamaForCausalLM`

The attention follows the GQA and RoPE design from [C3](../c03-attention), as a trimmed copy inside this package. `tests/test_matches_c3.py` checks that both produce the same output.

## Public contract

Other tutorials rely on this interface. C5 imports it through `--model tinyllm`.

```python
from tinyllm import ModelConfig, TinyLLM

config = ModelConfig(vocab_size=256, hidden_size=256, num_layers=6, num_heads=8,
                     num_kv_heads=2, intermediate_size=704, max_seq_len=256)
model = TinyLLM(config)
out = model(input_ids, labels=None, past_key_values=None)  # out.logits, out.loss, out.past_key_values
ids = model.generate(input_ids, max_new_tokens=50, temperature=0.8, top_k=40)
```

## Layout

```text
c04-llm/
├── README.md
├── conftest.py            puts this folder on sys.path for pytest
├── configs/               tiny, smollm2-135m and llama3-8b as JSON
├── tinyllm/
│   ├── config.py          ModelConfig and presets
│   ├── norm.py            RMSNorm
│   ├── rope.py            RoPE tables and rotation
│   ├── attention_naive.py reference attention, every step written out
│   ├── attention_sdpa.py  the same maths through the SDPA kernel
│   ├── attention.py       GroupedQueryAttention
│   ├── mlp.py             SwiGLU
│   ├── moe.py             optional mixture of experts
│   ├── block.py           one decoder block
│   ├── model.py           TinyLLM
│   ├── loss.py            shifted cross-entropy
│   ├── init.py            weight initialization
│   ├── params.py          parameter and cache size formulas
│   ├── sampling.py        temperature, top-k, top-p
│   ├── kv_cache.py        KVCache
│   ├── generation.py      generate, with or without the cache
│   ├── checkpoint.py      native save and load
│   ├── hf.py              Hugging Face conversion
│   ├── tokenizer.py       byte tokenizer, vocabulary of 256
│   └── utils.py           device, seed and timing helpers
├── scripts/               demos, one per topic
├── benchmarks/            scripts that print timing tables
├── tests/                 pytest
└── results/               real outputs of the commands below
```

## Setup

Run everything from the repository root.

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

Tested with Python 3.12, PyTorch 2.15 and transformers 5.22. `tinyllm/hf.py` uses transformers 5 argument names.

Scripts use the thread count that PyTorch picks. Set `OMP_NUM_THREADS` to change it. The results below used `OMP_NUM_THREADS=8`.

## Tests

```bash
python -m pytest c04-llm -m "not slow and not gpu and not network"   # offline, about 20 seconds
python -m pytest c04-llm -m network                                  # downloads SmolLM2-135M
python -m pytest c04-llm -m gpu                                      # needs a CUDA GPU, skipped otherwise
```

The tests cover shapes, causality, gradients, cache equivalence, checkpoint round trips, training resume, parameter formulas, and agreement with PyTorch, C3 and Hugging Face. `tests/test_matches_c3.py` skips itself if `../c03-attention` is missing.

## Scripts

Run `train_sanity.py` first. It downloads Tiny Shakespeare to `c04-llm/data/` and saves a checkpoint to `c04-llm/runs/tiny-shakespeare/`. Both folders stay out of git. `generate.py`, `moe_demo.py`, `compare_heads.py`, `walkthrough.py --step sampling` and two cases of `break_it.py` need them, and they tell you if they are missing.

| Command | What it does | Saved output |
|---|---|---|
| `python c04-llm/scripts/train_sanity.py --device cpu` | Trains the tiny preset for 1000 steps, as a sanity check | `results/train_sanity.txt` |
| `python c04-llm/scripts/walkthrough.py --step all` | One small experiment per tutorial step. `--step rope` runs one. | `results/walkthrough.txt`, `results/sampling_logits.json` |
| `python c04-llm/scripts/inspect_shapes.py` | Prints the shape and scale of every intermediate tensor | `results/inspect_shapes.txt` |
| `python c04-llm/scripts/param_count.py` | Checks the parameter formula against built modules, and compares cache sizes | `results/param_count.txt` |
| `python c04-llm/scripts/compare_hf.py --pretrained HuggingFaceTB/SmolLM2-135M` | Compares against `LlamaForCausalLM`, with random and with real weights | `results/compare_hf.txt` |
| `python c04-llm/scripts/generate.py --compare` | One sample for each sampling setting | `results/generate_compare.txt` |
| `python c04-llm/scripts/generate.py --prompt "ROMEO:" --temperature 0.8 --top-k 40` | One sample. `--no-cache` gives the same text, more slowly. | `results/generate_sample.txt` |
| `python c04-llm/scripts/break_it.py --case all --pretrained HuggingFaceTB/SmolLM2-135M --device cpu` | Five intentional bugs and their symptoms | `results/break_it.txt` |
| `python c04-llm/scripts/moe_demo.py --steps 300` | Trains an MoE model with and without the load-balancing loss | `results/moe_demo.txt` |
| `python c04-llm/scripts/compare_heads.py --steps 400 --device cpu` | Trains MHA, GQA and MQA on the same data | `results/compare_heads.txt` |

## Benchmarks

| Command | What it measures | Saved output |
|---|---|---|
| `python c04-llm/benchmarks/bench_attention.py --device cpu` | Train-step time, naive against SDPA | `results/bench_attention_cpu.txt` |
| `python c04-llm/benchmarks/bench_generation.py --device cpu` | Generation speed with and without the cache, tiny preset | `results/bench_generation_cpu.txt` |
| `python c04-llm/benchmarks/bench_generation.py --device cpu --preset smollm2-135m --new-tokens 128` | The same at the size of SmolLM2-135M, with random weights | `results/bench_generation_smollm2_cpu.txt` |

Both benchmarks accept `--dtype bf16`. On CUDA, `bench_attention.py` also reports peak allocated memory, and `bench_generation.py` turns off cuDNN attention unless you pass `--allow-cudnn-sdp` (see the GPU section).

## Results

All numbers come from the files in `results/`. Hardware: **AMD EPYC 7R13, 8 threads, CPU only**, with PyTorch 2.15.0 and transformers 5.22.2.

Every script and benchmark was run twice. Every seeded value matched exactly: losses, samples, logits, and the saved checkpoint down to its hash. Wall-clock times did not match, because the machine also runs other jobs. The tables show both runs where time is the result. The second run is in `results/timing_rerun.txt`. Read the speed-ups as rough.

| Test run | Result |
|---|---|
| Offline | 75 passed, in about 18 s |
| Network, SmolLM2-135M parity | 1 passed |
| GPU | 5 skipped, because no GPU was visible. They have never run. |
| C5 tests that train this model | 3 passed |

The full output is in `results/pytest.txt`.

### Training and checkpoint

| Measure | Value |
|---|---|
| Parameters, tiny preset | 4,295,936 |
| Validation loss, step 0 | 5.578 (a uniform guess gives ln 256 = 5.545) |
| Validation loss, step 1000 | 1.500 (lowest logged: 1.496 at step 800) |
| Training time for 1000 steps | 670 s and 737 s in the two runs, about 11,100 to 12,200 tokens/s |
| Checkpoint size, `model.safetensors` | 16.4 MiB |

### KV cache

| Model | Run | No cache | Cache | Speed-up |
|---|---|---|---|---|
| Tiny, 496 new tokens | 1 | 56.5 tok/s | 274.1 tok/s | 4.9x |
| Tiny, 496 new tokens | 2 | 65.7 tok/s | 241.6 tok/s | 3.7x |
| SmolLM2-135M size, random weights, 128 new tokens | 1 | 7.9 tok/s | 17.5 tok/s | 2.2x |
| SmolLM2-135M size, random weights, 128 new tokens | 2 | 6.4 tok/s | 13.8 tok/s | 2.1x |

### Naive attention against SDPA

Tiny preset, batch 8, forward and backward. SDPA wins clearly at 1024 tokens in both runs. At 128 and 256 tokens the gap is within the noise, and at 512 the size of the win is not stable.

| Sequence length | Run 1: naive, SDPA | Speed-up | Run 2: naive, SDPA | Speed-up |
|---|---|---|---|---|
| 128 | 141.7 ms, 116.7 ms | 1.21x | 122.5 ms, 134.3 ms | 0.91x |
| 256 | 276.0 ms, 232.1 ms | 1.19x | 300.9 ms, 279.8 ms | 1.08x |
| 512 | 1361.5 ms, 441.1 ms | 3.09x | 1277.2 ms, 664.6 ms | 1.92x |
| 1024 | 4644.9 ms, 1097.9 ms | 4.23x | 5742.4 ms, 1266.3 ms | 4.53x |

### MHA, GQA and MQA

Same data and seed, 400 steps. This is one seed on a small corpus, so these loss gaps are too small to rank the options. The cache size is the clear difference.

| Config | KV heads | Attention parameters | Cache per token | Validation loss |
|---|---|---|---|---|
| MHA | 8 | 1,572,864 | 12,288 B | 1.618 |
| GQA | 4 | 1,179,648 | 6,144 B | 1.621 |
| GQA | 2 | 983,040 | 3,072 B | 1.619 |
| MQA | 1 | 884,736 | 1,536 B | 1.622 |

### Agreement with Hugging Face

| Check | Result |
|---|---|
| Random tiny model, max logit difference | 0.000e+00 |
| Random tiny model, 64 greedy tokens | identical |
| Naive backend against Hugging Face, max logit difference | 7.153e-07 |
| SmolLM2-135M loaded into TinyLLM, max logit difference | 0.000e+00 |
| SmolLM2-135M, 24 greedy tokens | identical |
| Parameter formula against built modules | equal for all three presets, up to 8,030,261,248 for Llama 3 8B |

### Mixture of experts

8 experts, top-2, 300 steps, one seed. A balanced router gives each expert 0.125 of the load.

| `router_aux_loss_coef` | Busiest expert | Language model loss |
|---|---|---|
| 0.0 | 0.49 | 2.010 |
| 0.01 | 0.18 | 1.912 |

### Intentional bugs

| Bug | Symptom |
|---|---|
| `is_causal=True` on a chunked prefill | Max logit difference 8.96e-01, against 7.15e-07 with the shifted mask |
| Cached decode with positions restarting at 0 | First wrong token is new token 1, and the text degrades into repeats |
| Wrong RoPE channel pairing on SmolLM2-135M | Loss 3.385 nats/token, against 2.076 |
| Unshifted loss | Loss falls to 0.0048 and the model only copies its input |
| Residual projections without the depth scaling, 48 layers | Residual RMS 0.690 at layer 48, against 0.059 |

## GPU results

Hardware: **NVIDIA H100 80GB HBM3**, PyTorch 2.15.0, CUDA 13.2, one GPU, bf16 unless noted. The host is a shared machine, so the timings carry the same caveat as the CPU ones. Commands and raw output: `results/bench_attention_gpu_*.txt`, `results/bench_generation_*gpu*.txt`, `results/pytest_gpu.txt`.

```bash
python -m pytest c04-llm -m gpu
python c04-llm/benchmarks/bench_attention.py --device cuda --dtype bf16
python c04-llm/benchmarks/bench_generation.py --device cuda --dtype bf16
```

### GPU tests

All 5 tests in `tests/test_gpu.py` pass. They force the FlashAttention kernel, so PyTorch raises an error instead of falling back, and compare it with the naive attention for MHA, GQA and MQA. They also check a single cached decode step, and a chunked prefill in float32 that uses the shifted mask.

### Naive attention against SDPA on the GPU

Tiny preset, batch 8, forward and backward, bf16. Peak memory is `torch.cuda.max_memory_allocated`.

| Sequence length | Naive | SDPA | Speed-up | Naive peak | SDPA peak |
|---|---|---|---|---|---|
| 128 | 25.1 ms | 20.3 ms | 1.23x | 163 MiB | 117 MiB |
| 256 | 26.1 ms | 20.0 ms | 1.30x | 342 MiB | 184 MiB |
| 512 | 27.3 ms | 20.1 ms | 1.36x | 1001 MiB | 323 MiB |
| 1024 | 27.1 ms | 20.0 ms | 1.36x | 3460 MiB | 594 MiB |

The memory gap is the clear result. The naive path stores a score matrix per head, so its peak grows much faster than linearly with the length (about 3.5x from 512 to 1024 tokens), while SDPA's grows about 1.8x. The times are nearly flat from 128 to 1024 tokens, which is what a step limited by the many small kernel launches of a 6-layer model would look like. This model is too small to be limited by attention arithmetic, so the 1.2x to 1.4x speed-up says little about large models. In float32 the same run gives speed-ups of 1.24x to 1.56x, with a 1024-token peak of 3242 MiB for naive and 1045 MiB for SDPA.

An earlier run of this benchmark, made while the host was heavily loaded, gave times that varied by 2x between neighbouring lengths. It was discarded.

### Generation on the GPU

Batch 1, 16-token prompt, 496 new tokens, random weights.

| Model | Cache | tok/s | Per step |
|---|---|---|---|
| Tiny (4.3M) | off | 176.3 | about 5.5 ms |
| Tiny (4.3M) | on | 197.4 | about 5.0 ms |
| SmolLM2-135M size | off | 39.3 | about 25 ms |
| SmolLM2-135M size | on | 40.4 | about 24 ms |

The cache gives almost no speed-up here, and that is a measured result, not a bug: `tests/test_cache.py` shows the two paths produce the same tokens. At batch 1 the step time is the same whether the forward pass covers 17 or 511 tokens, which is consistent with a step limited by launching kernels from Python rather than by arithmetic. I did not profile this case. If that is right, avoiding the recomputation saves nothing. The CPU numbers above show the cache helping because there the arithmetic is the cost. A larger batch, a longer context or a bigger model would shift this, and none of those were measured.

### A slow default for decode

With PyTorch's default backend choice, a decode step of the tiny model takes about 50 ms instead of 5 ms, and the SmolLM2-size model takes about 73 ms instead of 24 ms. A profile of one step showed the cause: `aten::_cudnn_attention_forward` spends about 9.7 ms of CPU per call and about 4 microseconds on the GPU. PyTorch chooses cuDNN attention for the decode shape here, which is one query against a growing cache with grouped heads. The prefill step (5.5 ms) and the training benchmark were not slowed. I did not profile which kernel they use, and I did not find out why the cuDNN call is slow on the host.

`tinyllm.utils.disable_cudnn_sdp()` turns that backend off, and `bench_generation.py` calls it on CUDA unless you pass `--allow-cudnn-sdp`. The library does not call it for you, because it sets a global PyTorch flag. If you generate on a GPU, call it once first. Both settings are saved: `results/bench_generation_gpu.txt` and `results/bench_generation_smollm2_gpu.txt` with it off, and the `_cudnn_default` files with PyTorch's default.

### Not measured on the GPU

Batch sizes above 1, long contexts, float16, multi-GPU runs and any comparison of the FlashAttention kernel against other kernels on its own.
