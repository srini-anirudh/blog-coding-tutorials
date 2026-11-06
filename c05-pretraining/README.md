# C5: Train an LLM From Scratch

A small but complete LLM pretraining system. The package `pretrain` turns raw stories into token shards, packs them into
fixed windows, and trains a Llama-architecture decoder with AdamW, a warmup and cosine schedule, gradient accumulation,
bf16, clipping and activation checkpointing. It saves full checkpoints and resumes them bitwise. It logs every number that
matters, estimates MFU, scales out with DDP, FSDP and ZeRO, and it survives the failures it causes on purpose.

Tutorial: https://srini-anirudh.github.io/blogs/train-an-llm-from-scratch/

The default model is Hugging Face `LlamaForCausalLM`, so this folder runs on its own. `--model tinyllm` trains the model you
built in C4 by importing `../c04-llm`. That path is proven end to end below.

## Layout

```
c05-pretraining/
├── pretrain/
│   ├── corpus.py           # download TinyStories, split into documents
│   ├── tokenizer.py        # GPT-2 from the Hub, or a byte-level BPE we train
│   ├── shards.py           # uint16 token shards with EOS after every document
│   ├── packing.py          # padding against packing, utilisation, document ids and masks
│   ├── loader.py           # map-style windows, rank-sharded sampler, resumable loader
│   ├── streaming.py        # worker and rank sharding for an IterableDataset
│   ├── model.py            # build HF Llama or C4's TinyLLM, count params, sample
│   ├── optim.py            # AdamW with decay and no-decay groups
│   ├── schedule.py         # warmup, cosine decay, minimum LR
│   ├── step.py             # one optimiser step: accumulation, autocast, clipping, finite check
│   ├── activation_ckpt.py  # keep block inputs, recompute the rest in backward
│   ├── checkpoint.py       # atomic writes, checksums, fallback on corruption
│   ├── rng.py              # capture and restore every random generator
│   ├── metrics.py          # JSONL metrics
│   ├── mfu.py              # FLOPs per token, measured peak, utilisation
│   ├── distributed.py      # rank, world size, DDP, all-reduce by hand
│   ├── fsdp.py             # FSDP2 wrapping and per-rank byte counts
│   ├── zero.py             # ZeRO stage 1 and the DeepSpeed config
│   ├── guards.py           # bad batch, NaN, loss spike, OOM retry
│   ├── trainer.py          # the loop that calls all of the above
│   ├── config.py           # ModelArgs and TrainConfig
│   └── utils.py            # device, seeding, memory readings
├── scripts/                # runnable demos, listed below
├── benchmarks/             # packing, loader, precision, checkpointing, MFU
├── configs/                # tinystories.yaml, smoke.yaml, ddp_cpu.yaml, deepspeed_zero3.json
├── results/                # real outputs. The first line of each file states its hardware
└── tests/
```

## Tensor shapes

| Tensor | Shape | Where |
|---|---|---|
| One window, input and target | `(L,)` | `loader.PackedWindows` |
| Micro-batch input and target | `(B_micro, L)` | `loader.ResumableLoader` |
| Logits | `(B_micro, L, V)` | `step.train_step` |
| Sequences per optimiser step | `B_micro * N_accum * world_size` | `step.py` |

## Run

From the repository root, after `pip install -e ".[dev]"`. PyTorch uses every core by default, so set `OMP_NUM_THREADS` on a
shared machine. Numbers below used `OMP_NUM_THREADS=8`.

Build the data first. `data_cache/` and `runs/` stay out of git. This downloads the first 128 MiB of TinyStories V2 and takes
about a minute and a half.

```bash
python c05-pretraining/scripts/prepare_data.py
python c05-pretraining/scripts/prepare_data.py --tokenizer gpt2   # optional: GPT-2's tokenizer instead of our BPE
```

Train. On a CPU pass `--precision fp32`: bf16 is emulated on CPUs without native support, and was 11 times slower here.
Running the same command again resumes from the newest valid checkpoint. A checkpoint stores its config, so a different
model or setting needs its own `--out-dir` or `--fresh`.

```bash
python c05-pretraining/scripts/train.py --config c05-pretraining/configs/tinystories.yaml --precision fp32 --device cpu --out-dir runs/tinystories
python c05-pretraining/scripts/train.py --config c05-pretraining/configs/tinystories.yaml --precision fp32 --device cpu --model tinyllm --out-dir runs/tinyllm
python c05-pretraining/scripts/sample.py --ckpt c05-pretraining/runs/tinystories/ckpt_000800.pt
torchrun --nproc_per_node 2 c05-pretraining/scripts/train.py --config c05-pretraining/configs/ddp_cpu.yaml --threads 4
torchrun --nproc_per_node 2 c05-pretraining/scripts/train.py --config c05-pretraining/configs/ddp_cpu.yaml --threads 4 --model tinyllm --out-dir runs/ddp_cpu_tinyllm
```

Scripts:

```bash
python c05-pretraining/scripts/inspect_tokens.py       # ids, EOS, boundaries, windows
python c05-pretraining/scripts/naive_train.py          # the simplest loop that works
python c05-pretraining/scripts/plot_schedule.py        # the LR curve, written to runs/lr_schedule.png
python c05-pretraining/scripts/check_accumulation.py   # accumulation against one big batch
python c05-pretraining/scripts/compare_hf_trainer.py   # our loop against HF Trainer
python c05-pretraining/scripts/compare_init.py         # why TinyLLM ends lower than HF Llama (needs both main runs)
python c05-pretraining/scripts/resume_test.py          # add --model tinyllm for C4's model
python c05-pretraining/scripts/break_it.py --case all  # too_high_lr, nan_gradient, bad_batch, spike, oom, corrupt_checkpoint
torchrun --nproc_per_node 2 c05-pretraining/scripts/ddp_equivalence.py
torchrun --nproc_per_node 2 c05-pretraining/scripts/fsdp_train.py --steps 5
torchrun --nproc_per_node 2 c05-pretraining/scripts/zero_train.py
torchrun --nproc_per_node 2 --max-restarts 1 c05-pretraining/scripts/dead_process_demo.py
```

Benchmarks:

```bash
python c05-pretraining/benchmarks/bench_packing.py
python c05-pretraining/benchmarks/bench_loader.py
python c05-pretraining/benchmarks/bench_precision.py --config c05-pretraining/configs/smoke.yaml --device cpu
python c05-pretraining/benchmarks/bench_checkpointing.py --device cpu   # add --model tinyllm for C4's model
python c05-pretraining/benchmarks/bench_mfu.py --device cpu --metrics runs/tinystories/metrics.jsonl
```

GPU only. `deepspeed_train.py` also needs `pip install deepspeed`, which is not in this repository's dependencies.

```bash
torchrun --nproc_per_node 4 c05-pretraining/scripts/train.py --config c05-pretraining/configs/tinystories.yaml --out-dir runs/gpu_ddp
torchrun --nproc_per_node 4 c05-pretraining/scripts/fsdp_train.py --steps 100 --hidden-size 1024 --num-layers 12
torchrun --nproc_per_node 4 c05-pretraining/scripts/deepspeed_train.py --stage 3 --steps 50
python c05-pretraining/benchmarks/bench_precision.py --device cuda
python c05-pretraining/benchmarks/bench_checkpointing.py --device cuda
python c05-pretraining/scripts/break_it.py --case oom --device cuda
```

## Test

```bash
python -m pytest c05-pretraining -m "not slow and not gpu and not network"   # 34 passed, about 70 s
python -m pytest c05-pretraining -m network                                  # 2 passed
python -m pytest c05-pretraining -m gpu                                      # needs two GPUs, skipped without them
```

The tests cover shapes, accumulation against a big batch, HF Trainer and schedule equivalence, clipping, activation
checkpointing gradients, bitwise resume on both models, corrupt checkpoints, every guard, the FLOP formula against
`FlopCounterMode`, and real two-process DDP on gloo.

## Results

Hardware for every number here: **AMD EPYC 7R13, 8 threads**, CPU only, PyTorch 2.15.0. That CPU has no native bf16. Ranks in
the two-process runs share those 8 threads. The GPU numbers are in their own section at the end.

### Training

Same config for both main runs: 5,401,856 parameters, `L = 256`, 8,192 tokens per optimiser step, 800 steps, fp32, seed 1337.
The checkpoint is 61.9 MiB, because it also holds AdamW's two moment tensors.

| Run | First loss | Val loss at 200 | Val loss at 400 | Val loss at 800 | Median tokens/s | Time | Result file |
|---|---|---:|---:|---:|---:|---|---|
| HF Llama | 8.3875 | 3.6379 | 3.0108 | 2.6135 | 6,953 | 17 min | `train_tinystories.log` |
| C4 TinyLLM | 8.3952 | 3.0837 | 2.6083 | 2.2611 | 7,098 | 17 min | `train_tinyllm.txt` |
| HF Llama, 2-rank DDP, smaller model | 8.3468 | 3.9259 | - | - | about 20,000 | 45 s | `train_ddp_cpu.log` |
| C4 TinyLLM, 2-rank DDP, smaller model | 8.3465 | 3.7687 | - | - | about 17,500 | 48 s | `train_ddp_cpu_tinyllm.log` |

The two DDP runs use `configs/ddp_cpu.yaml`: 1,250,432 parameters and 200 steps. Both of those models write readable stories.
See `sample.txt` and `sample_tinyllm.txt`.

TinyLLM ends 0.35 lower on validation loss, although both models compute the same function (C4's `compare_hf.py` shows a
logit gap of 0). The cause is the starting point. HF draws every weight with std 0.02. C4 also divides `o_proj` and
`down_proj` by `sqrt(2 * num_layers)`. `scripts/compare_init.py` trains HF Llama again with only that scaling added:

| Step | HF Llama, flat init | C4 TinyLLM | HF Llama, scaled init |
|---:|---:|---:|---:|
| 50 | 5.3276 | 4.4215 | 4.4133 |
| 200 | 3.6379 | 3.0837 | 3.0695 |
| 400 | 3.0108 | 2.6083 | 2.5965 |
| 800 | 2.6135 | 2.2611 | 2.2529 |

With the scaled init, HF Llama follows TinyLLM to within 0.01. So the gap comes from the initialisation, not the code.
Source: `results/compare_init.txt`. This is one seed per run.

### Checks against a trusted result

| Check | Result | File |
|---|---|---|
| Our loop against HF Trainer, 20 steps | max weight gap 0.00e+00 | `compare_hf_trainer.txt` |
| Accumulation against a big batch, unequal targets | ours 3.73e-09, mean of means 8.25e-03 | `check_accumulation.txt` |
| 2-rank DDP against a big batch | 3.73e-09, ranks identical | `ddp_equivalence.txt` |
| Resume against an uninterrupted run, both models | bitwise equal | `resume_test.txt`, `resume_test_tinyllm.txt` |
| FLOP formula against `FlopCounterMode` | ratio 0.98 | `bench_mfu.txt` |
| FSDP against DDP, 5 steps on gloo | max weight gap 3.70e-07 | `fsdp_train.txt` |
| ZeRO-1 against DDP, 5 steps on gloo | max weight gap 0.00e+00 | `zero_train.txt` |

### Benchmarks

| Benchmark | Result | File |
|---|---|---|
| Packing, `L = 256` | greedy packing 100.0%, pad to `L` 73.0% and cuts 8.1% of tokens | `bench_packing.txt` |
| Loader | 0 workers 1,910 batches/s, 2 workers 718, 4 workers 1,161 | `bench_loader.txt` |
| fp32 against bf16, smoke config | 41.9 against 460.8 ms per step, largest loss gap 0.0010 | `bench_precision.txt` |
| Activation checkpointing, HF Llama | autograd saved 571.7 to 104.1 MiB, peak growth 749.7 to 528.2 MiB, step time 1.10x | `bench_checkpointing.txt` |
| Activation checkpointing, TinyLLM | autograd saved 572.6 to 105.1 MiB, peak growth 768.5 to 370.2 MiB, step time 1.28x | `bench_checkpointing_tinyllm.txt` |
| MFU | 6,972 tokens/s is 0.259 TFLOP/s, 35.4% of a measured 0.731 TFLOP/s matmul peak | `bench_mfu.txt` |
| FSDP per-rank bytes, 2 ranks on gloo | params 4.75 to 2.38 MiB, optimizer state 9.50 to 4.76 MiB | `fsdp_train.txt` |
| ZeRO-1 per-rank optimizer state, 2 ranks | 9.54 to 4.76 MiB | `zero_train.txt` |

Read these with care:

- The loader is fastest with 0 workers here. Reading a memory-mapped window is so cheap that sending it between processes
  costs more. Workers pay off when a sample needs real work, or when the training step leaves the main process idle.
- Peak growth is process memory on a CPU, and the allocator adds noise to it. The autograd figure is the steadier one.
- The MFU peak is the best matmul this CPU reached on this shared machine, not a datasheet value. It moved between 0.62 and
  0.73 TFLOP/s across runs, so the MFU is a rough figure.
- bf16 is this slow because this CPU emulates it. The GPU section measures bf16 separately.

### Failure modes

All real, from `results/break_it.txt` and `results/dead_process.txt`.

| Failure | What happens | Recovery |
|---|---|---|
| Too-high LR, held for 150 steps | lr=3e-3 reaches 4.394. lr=3e-1 stalls at 5.959. lr=3 goes past 100, with or without clipping | Lower the LR. Clipping does not save Adam, which rescales the gradient anyway |
| NaN gradient | The step is skipped and weights do not move. Three in a row raise `TrainingDiverged` | Reload the last good checkpoint |
| Bad batch | `BadBatchError: input ids in [17, 5000], vocab is [0, 4096)` | Fail before the model runs |
| Loss spike | Random tokens at step 100: loss 10.586 against a recent median of 5.660, flagged | Logged. The next step is back to 5.624 |
| OOM | Real allocator failure at a 1 GiB cap. Splits from 1 micro-batch of 128 to 16 of 8 | The same 32,768 tokens finish |
| Corrupt checkpoint | The checksum fails on a truncated file | Falls back to the older checkpoint and resumes |
| Dead process | Rank 1 exits at step 15 | torchrun restarts both ranks, which resume from step 10 and finish |

The dead process case first failed for real. torchrun's static rendezvous keeps its store across a restart, so the new rank 1
read the old address of rank 0 and got `Connection refused`. `init_distributed` now uses a new key prefix per restart attempt.

### GPU

Hardware: **NVIDIA H100 80GB HBM3**, PyTorch 2.15.0, CUDA 13.2. Every run used one GPU or two, with no other job on them.
These are the same configs as above, at full length. Files are `results/gpu_*`.

| Run | Val loss at 800 | Median tokens/s | Median step ms | Time | File |
|---|---:|---:|---:|---:|---|
| HF Llama, 1 GPU, bf16 | 2.6142 | 148,000 | 55.4 | 52 s | `gpu_train_1gpu.log` |
| HF Llama, 2 GPUs, DDP on NCCL | 2.3521 | 270,500 | 60.5 | 54 s | `gpu_train_ddp.log` |
| C4 TinyLLM, 2 GPUs, DDP on NCCL | 2.0171 | 273,500 | 59.3 | 53 s | `gpu_train_ddp_tinyllm.log` |

The 1-GPU run reaches 2.6142, against 2.6135 for the same run on the CPU in fp32. DDP keeps the per-GPU batch, so two GPUs
see twice the tokens per step (16,384). That is why the DDP runs end lower. It is not the same run as the 1-GPU one. Two GPUs
give 1.83 times the tokens per second of one.

| Check or benchmark | Result | File |
|---|---|---|
| DDP on NCCL against one big batch | 3.73e-09, ranks identical | `gpu_ddp_equivalence.txt` |
| Resume on a GPU | bitwise equal | `gpu_resume_test.txt` |
| fp32 against bf16, main config | 45.8 against 60.5 ms per step, largest loss gap 0.0079 | `gpu_bench_precision.txt` |
| Activation checkpointing | autograd saved 571.7 to 104.1 MiB, peak growth 620.0 to 188.1 MiB, step time 1.71x | `gpu_bench_checkpointing.txt` |
| MFU | 147,784 tokens/s is 5.487 TFLOP/s, 0.8% of a measured 687.7 TFLOP/s bf16 matmul peak | `gpu_bench_mfu.txt` |
| Real CUDA out-of-memory at a 1 GiB cap | splits from 1 micro-batch of 128 to 8 of 16, all 32,768 tokens finish | `gpu_break_oom.txt` |
| FSDP2 against DDP, 2 GPUs, 141M params, 100 steps | per GPU: params 544.0 to 272.0 MiB, optimizer state 1088.0 to 544.0 MiB, peak 2769.1 to 1854.4 MiB | `gpu_fsdp_train.txt` |
| GPU-marked test | 1 passed | `gpu_pytest.txt` |

Read these with care:

- This 5.4M-parameter model is far too small for an H100. Each step is a few short kernels, so the GPU waits on kernel
  launches. That is why MFU is 0.8%, and why bf16 is slower than fp32 here: autocast adds casts that cost more than the
  smaller matmuls save. Expect the opposite on a model large enough to keep the GPU busy.
- FSDP ends with a max weight gap of 9.52e-04 from DDP after 100 steps. On the CPU the gap was 3.70e-07 after 5. FSDP sums
  gradients in a different order (reduce-scatter instead of all-reduce), and AdamW carries the rounding difference forward
  for 100 steps. The losses agree to about 4e-05.
- The FSDP model is built for this check (width 1024, 12 layers). It is not the model trained above.

Not measured: DeepSpeed ZeRO stages 1 to 3. DeepSpeed is not installed, and `scripts/deepspeed_train.py` has never executed on
any hardware. Treat it as untested.
