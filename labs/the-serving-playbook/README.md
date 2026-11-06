# Lab: The Serving Playbook

A small LLM serving engine, written from scratch, that runs the mechanisms the post explains: continuous
batching, a paged KV cache with copy-on-write and chained-hash prefix caching, an iteration-level scheduler with
a token budget, chunked prefill, preemption, CUDA-graph decode and speculative sampling. Every mechanism is a few
hundred lines of Python and PyTorch. The engine runs real models and its output is checked token for token
against plain greedy decoding.

Post: https://srini-anirudh.github.io/blogs/the-serving-playbook/

No vLLM, SGLang or other server is used or started. The model weights come from tutorial C4 (`../../c04-llm/tinyllm`,
which loads SmolLM2 from the Hugging Face Hub). This lab adds a different forward pass over those weights.

## What is in the box

```
labs/the-serving-playbook/
├── servekit/
│   ├── block_manager.py   # free list, block tables, ref counts, copy-on-write, chained-hash prefix cache
│   ├── model.py           # Llama forward pass over a ragged batch that reads and writes a paged KV cache
│   ├── scheduler.py       # token budget, sequence slots, watermark, chunked prefill, preemption by recompute
│   ├── engine.py          # the loop (schedule, build batch, forward, sample), load shedding, CUDA-graph decode
│   ├── alloc_sim.py       # reserve-max vs first-fit vs paged KV memory on a request trace
│   ├── spec.py            # the speculative accept / resample rule
│   └── metrics.py         # TTFT, TPOT, goodput, workloads, results-file headers
├── scripts/               # one script per experiment, listed below
├── tests/                 # pytest
└── results/               # real outputs; each starts with a hardware line and the command
```

Three simplifications to keep in mind:

- Attention reads the paged cache with plain PyTorch indexing (`cache[block_table]`), then calls PyTorch's fused
  attention. The post describes a hand-written kernel that gathers inside the kernel. Ours pays an extra copy.
- Sampling is greedy and every request runs for exactly `max_new_tokens`. That makes runs reproducible and lets the
  tests compare against plain decoding.
- The engine is eager PyTorch. On a small model, Python and kernel launches cost more than the GPU work. This
  matters for how to read the timings, and the results below say where.

## Setup

From the repository root, after `pip install -e ".[dev]"`. The GPU scripts need one CUDA GPU. The real checkpoint
(`HuggingFaceTB/SmolLM2-360M`, 135M for the test) downloads from the Hub on first use.

## Run everything

```bash
# Tests. The first command needs no network and no GPU.
python -m pytest labs/the-serving-playbook -m "not slow and not gpu and not network"
python -m pytest labs/the-serving-playbook -m network     # SmolLM2-135M vs Hugging Face generate()
python -m pytest labs/the-serving-playbook -m gpu         # CUDA-graph decode equals eager decode

# CPU only
python labs/the-serving-playbook/scripts/02_kv_memory.py          # sections 5, 6, 21
python labs/the-serving-playbook/scripts/05b_schedule_trace.py    # section 8, as a schedule
python labs/the-serving-playbook/scripts/09_speculative.py        # section 14

# One GPU
python labs/the-serving-playbook/scripts/01_roofline.py           # sections 1, 14
python labs/the-serving-playbook/scripts/03_batching.py           # sections 2, 3
python labs/the-serving-playbook/scripts/04_prefix_cache.py       # sections 7, 19, 21, 22
python labs/the-serving-playbook/scripts/05_chunked_prefill.py    # sections 8, 9
python labs/the-serving-playbook/scripts/06_blocks_and_preemption.py   # sections 4, 21
python labs/the-serving-playbook/scripts/07_goodput.py            # section 23
python labs/the-serving-playbook/scripts/08_cuda_graphs.py        # section 15
```

Each script prints a table and writes the same text to `results/`. Use `--model tiny` on the CPU for a quick smoke
test (random weights; the numbers mean nothing).

## Which script covers which part of the post

| Post section | Script | What it shows |
|---|---|---|
| 1. One request, one GPU | `01_roofline.py` | One 8B-sized layer's GEMMs take the same time from 1 to 256 token rows, then grow with the rows |
| 2. Static batching, 3. Continuous batching | `03_batching.py` | Same 256 requests, same model: slot squatting versus refilling slots every iteration |
| 4. The scheduler | `servekit/scheduler.py`, `tests/test_scheduler.py`, `06_...py` (part C) | Budget, slots, watermark, preemption by recompute |
| 5. The KV allocation problem, 6. Paged KV cache | `02_kv_memory.py` | KV bytes per token; three allocators on one request trace |
| 7. Prefix caching | `04_prefix_cache.py` | Shared system prompt computed once; request order changes the work |
| 8. Chunked prefill, 9. Prefill/decode interference | `05b_schedule_trace.py`, `05_chunked_prefill.py` | The schedule, and what an iteration of each size costs |
| 14. Speculative decoding | `09_speculative.py`, `01_roofline.py` | The rule is exact; tokens per pass match the formula; verifying 5 rows per sequence is free only at small batch |
| 15. CUDA graphs | `08_cuda_graphs.py` | Eager versus replayed decode step |
| 19. VLM serving: hash the pixels | `04_prefix_cache.py` (part C) | Token ids alone hand one image's KV to another |
| 21. vLLM up close | `06_...py` (parts A, B), `02_kv_memory.py`, `04_...py` | `num_gpu_blocks`, sharing by reference count, recompute versus swap |
| 23. Goodput | `07_goodput.py` | Throughput keeps rising while goodput collapses; shedding keeps part of it |

## Results

Every number names its hardware. Raw output is in `results/`.

GPU rows: NVIDIA H100 80GB HBM3, torch 2.15.0+cu132, CUDA 13.2, one GPU, bf16. CPU rows: AMD EPYC 7R13, 8 threads,
torch 2.15.0+cu132. The GPU was driven by a shared host with a busy CPU, so eager-PyTorch timings are higher than
they would be on a quiet machine. Each script is one run unless it says "median".

### Correctness

| Check | Result |
|---|---|
| Offline tests (CPU) | 21 passed (`results/tests.txt`) |
| SmolLM2-135M through the engine vs Hugging Face `generate()`, chunked prefill, 3 prompts (network test, CPU) | identical tokens |
| Engine vs plain no-cache greedy decoding, continuous, static, chunked to 5 tokens, preemption, prefix cache (CPU, fp32) | identical tokens in every case |
| CUDA-graph decode vs eager vs CPU (random tiny model, GPU test) | identical tokens |

### Section 1 and 14: the batch-size free lunch (`01_roofline.py`, H100)

One Llama-3-8B decoder layer's seven GEMMs, random bf16 weights, 416 MiB.

| Token rows B | 1 | 16 | 64 | 128 | 256 | 512 | 1,024 | 4,096 |
|---|---|---|---|---|---|---|---|---|
| Time (µs) | 184 | 189 | 183 | 196 | 212 | 330 | 619 | 2,509 |
| Achieved TFLOP/s | 2.4 | 36.9 | 152 | 286 | 527 | 677 | 722 | 712 |

The time is flat from 1 to 256 rows, then grows with B. At B = 1 the layer moves 2.37 TB/s, so decode is bandwidth
bound. Past about 256 rows the GEMMs are compute bound at roughly 710 to 730 TFLOP/s, below the 989 TFLOP/s peak
the post quotes. The knee sits near 256 rows. The post's balance point of 295 FLOP/byte comes from the datasheet. The measured
bandwidth (2.4 TB/s) and compute (about 730 TFLOP/s) are both lower than the datasheet, which puts the measured balance
point near 300 FLOP/byte as well.

Speculative verification (gamma = 4, so 5 rows per sequence): cost of 5B rows divided by B rows is 1.00 at B = 4,
0.98 at B = 16, 1.32 at B = 64 and 3.42 at B = 256. Speculation is free at small batch and costs real compute
once the batch fills the machine.

The whole SmolLM2-360M decode step in our eager engine takes about 30 ms from batch 1 to batch 128 (34 to 4,161
tokens/s). That is the engine's launch cost, not GPU work. See the CUDA-graph row below.

### Section 2 and 3: static versus continuous batching (`03_batching.py`, H100, SmolLM2-360M)

256 requests at t = 0, prompts 32 to 512 tokens, outputs 8 to 512 tokens (mean 120), greedy.

| Scheduler | Slots | Wall (s) | Output tok/s | Mean latency (s) | Decode steps | Mean sequences still wanting a token per step |
|---|---|---|---|---|---|---|
| Static | 32 | 130.9 | 235 | 62.8 | 3,509 | 8.7 |
| Continuous | 32 | 47.8 | 643 | 19.3 | 1,186 | 25.7 |
| Static | 128 | 38.6 | 796 | 15.4 | 981 | 31.1 |
| Continuous | 128 | 28.7 | 1,070 | 9.6 | 548 | 55.7 |

In the first static group of 32, the number of sequences that still wanted tokens fell from 32 to 27 (step 8), 19
(step 32), 9 (step 128), 5 (step 256) and 1 (step 406). That is the decay in the post's Figure 2.
Because the engine is launch bound (each step costs about the same whatever the batch), wall time follows the
step count. Continuous batching needs 2.96 times fewer steps than static at 32 slots, and the wall time is 2.7
times shorter.

### Section 5, 6 and 21: KV memory (`02_kv_memory.py`, CPU simulation)

A synthetic trace of 3,000 requests (prompts 50 to 1,000 tokens, outputs 16 to 2,000 tokens log-uniform, mean 414,
clients ask for max_tokens 2,048) against a 60 GB pool of Llama-3-8B KV (457,763 tokens at 128 KiB per token).
This is a simulation of memory, not a measurement of a real system. Time is counted in decode steps.

| Allocator | Decode steps to drain the trace | Mean concurrency | KV written / KV reserved | Steps blocked by fragmentation |
|---|---|---|---|---|
| Reserve max_model_len (8,192) per sequence | 23,586 | 52.6 | 0.125 | 0 |
| Contiguous, prompt + requested max_tokens, first fit | 9,002 | 137.8 | 0.398 | 7,224 |
| Paged, block size 16, with 14 preemptions | 3,960 | 313.3 | 0.993 | 0 |

The paged allocator drains the trace in 5.96 times fewer steps than reserve-max. First fit blocked the head of the
queue in 7,224 steps while it had enough total free memory but no hole big enough. That is external fragmentation.

Sharing by reference count: 8 samples of a 2,000-token prompt hold 133 blocks (266 MiB) after each writes one
token, versus 1,008 blocks (2,016 MiB) with private copies. After 100 tokens each: 181 blocks versus 1,056.

### Section 7 and 19: prefix caching (`04_prefix_cache.py`, H100, SmolLM2-360M)

64 requests, each a shared 2,048-token system prompt plus its own 64-token question, 32 output tokens, all at t = 0.

| dtype | Prefix cache | Prefill tokens computed | Wall (s) | TTFT p50 (ms) | Same 32 tokens as without the cache |
|---|---|---|---|---|---|
| bf16 | off | 135,168 | 6.71 | 3,981 | |
| bf16 | on | 4,096 | 4.30 | 2,261 | 0.84 of requests |
| fp32 | off | 135,168 | 7.03 | 3,462 | |
| fp32 | on | 4,096 | 4.32 | 2,331 | 1.00 of requests |

The cache removes 97% of the prefill tokens. In bf16, 16% of requests differ in some token after 32 steps, in fp32
none do. The cached path computes the same values with different tile shapes, and bf16 rounding is enough to flip
a near-tie in the argmax. In fp32, and in the CPU tests, the outputs are identical, so the cache is exact and the
difference is rounding. The wall-time gain is smaller than the token saving because the 4,096 suffix tokens still
run through a launch-bound engine.

Request order (block manager only, pool of 10 blocks): the order A B C D A B C D needs 640 prefill tokens, the
grouped order A A B B C C D D needs 384, which is 0.60 of it.

Part C, with the same placeholder ids for two different images: a cache keyed on token ids alone reports 80 cached
tokens to the second image's request (wrong, they belong to the first image). With the image hash in the block key it
reports 0.

### Section 8 and 9: chunked prefill (`05b_schedule_trace.py`, `05_chunked_prefill.py`)

`05b_schedule_trace.py` runs the scheduler alone (CPU). A 7,680-token prompt joins 32 decoding sequences. With no
chunking the largest iteration is 7,712 tokens and the prompt takes 1 iteration. With a 2,048, 512 or 256-token
budget the largest iterations are 2,048, 512 and 256 tokens and the prompt takes 4, 17 and 36 iterations. The
decodes ride along in every one of them (31 to 32 per iteration). The counts exceed 7,680 / C because the 32
decodes use part of each budget.

`05_chunked_prefill.py` times the same workload on the H100 with a random-weight model of Llama-3-8B shape (the
speed does not depend on weight values). 32 sequences decode while the long prompt arrives.

| Chunk | Long prompt TTFT (ms) | Decode ITL p50 (ms) | ITL p99 (ms) | ITL max (ms) | Mixed iterations |
|---|---|---|---|---|---|
| none | 325 | 31.8 | 88.5 | 294 | 1 |
| 2,048 | 559 | 33.8 | 132 | 220 | 6 |
| 1,024 | 1,009 | 34.1 | 109 | 188 | 11 |
| 512 | 2,332 | 33.8 | 162 | 220 | 24 |
| 256 | 5,805 | 33.8 | 210 | 210 | 51 |

Prefilling the 7,680-token prompt alone: one iteration takes 243 ms (31,606 tokens/s). With chunks of 2,048, 1,024,
512, 256 it takes 525, 719, 1,336 and 2,621 ms (14,625 to 2,930 tokens/s).

How to read this honestly: chunking lowers the worst stall (294 ms to 188 to 220 ms), as the post says. It does not
reach the post's clean picture, because here every iteration costs at least 100 ms regardless of its size. A
256-token chunk with 32 decodes takes about 123 ms, a 2,048-token chunk about 144 ms. The GPU work for 256 tokens
of an 8B model is a few milliseconds. The rest is eager-PyTorch launch time on a busy host. Splitting the prompt
into more iterations therefore costs more total time, and the p99 inter-token latency gets worse with small
chunks. This is the post's "C too small" rule and its CUDA-graph section meeting in one table. A production
engine hides that cost with graphs and an overlapped scheduler. This engine does not, and does not reproduce the
post's Sarathi-Serve numbers.

### Section 15: CUDA graphs (`08_cuda_graphs.py`, H100, SmolLM2-360M)

One decode step for B sequences at context 256, median wall time per step including scheduling and sampling.
Eager runs the paged forward pass kernel by kernel from Python. The graph path replays one captured graph per batch
size bucket (B padded up to 1, 2, 4, 8, ..., 128) with every block table padded to the maximum length.

| Batch B | Eager (ms) | Graph (ms) | Speedup | Eager tok/s | Graph tok/s | Same tokens as eager |
|---|---|---|---|---|---|---|
| 1 | 31.2 | 3.87 | 8.0x | 32 | 258 | 1.00 |
| 4 | 31.2 | 4.04 | 7.7x | 128 | 989 | 1.00 |
| 16 | 32.0 | 4.46 | 7.2x | 500 | 3,586 | 1.00 |
| 64 | 31.9 | 5.94 | 5.4x | 2,009 | 10,781 | 0.94 |
| 128 | 33.7 | 7.73 | 4.4x | 3,798 | 16,553 | 0.94 |

The eager step costs about 31 ms at every batch size, so the GPU is idle most of the time. The graph removes that
cost and the step then grows with the batch, as the roofline says it should. Graphs and eager decode give the same
tokens at small batch. At 64 and 128 sequences 6% of sequences differ in some token after 40 steps. Padding changes the
shapes of the reductions, and in bf16 that is enough to flip a near-tie. The GPU test checks equality on a tiny model.

### Section 4 and 21: where `num_gpu_blocks` comes from, and recompute versus swap (`06_blocks_and_preemption.py`, H100)

Random-weight model of Llama-3-8B shape (weights measured after loading, not computed from the parameter count).

| Quantity | Value |
|---|---|
| GPU memory | 79.2 GiB |
| Weights, bf16 | 14.97 GiB |
| Activation peak, 8,192-token profiling pass | 1.01 GiB |
| KV per block (16 tokens) | 2,048 KiB |
| `gpu_memory_utilization` 0.85 / 0.90 / 0.95 | 26,281 / 28,308 / 30,335 blocks |
| Sequences of 4k tokens at 0.85 / 0.95 | 102 / 118 |

Going from 0.85 to 0.95 adds 7.9 GiB of KV, which is 16 more 4k-token sequences. This is the post's arithmetic,
measured.

Preempting one 8,000-token sequence (1,000 MiB of KV):

| Way back | Time |
|---|---|
| Recompute (prefill again) | 253 ms |
| Swap to pinned host memory and back | 637 ms (561 out at 1.9 GB/s, 76 in at 13.8 GB/s) |
| Swap at the 50 GB/s each way the post assumes (arithmetic, not measured) | 42 ms |

On this machine the device-to-host copy ran at only 1.9 GB/s, so swap lost by 2.5 times, the opposite of the post's
8-times win for swap. The conclusion depends on link bandwidth, and the measured link was far below the post's
assumption. The post's 327 ms recompute figure assumes 400 TFLOP/s achieved. We measured 253 ms, about 507 TFLOP/s
for the 128 TFLOP prefill.

Shrinking the pool under 48 requests (prompts 64 to 256, outputs 64 to 512):

| KV blocks | Preemptions | Wall (s) | Output tok/s |
|---|---|---|---|
| 4,000 | 0 | 24.5 | 391 |
| 600 | 5 | 18.5 | 516 |
| 300 | 25 | 36.4 | 263 |

The 600-block run is faster than the 4,000-block run. The cause is not clear from this data. One likely reason is
that at 4,000 blocks all 48 sequences decode together and each eager step is launch bound at about the same cost
whatever the batch, so a smaller pool should not win. We did not investigate, and one run per row is not enough to
call the difference real. The 300-block row shows the expected cost: 25 preemptions and 64% more prefill tokens
(11,241 versus 6,854).

### Section 23: goodput (`07_goodput.py`, H100, SmolLM2-360M, CUDA-graph decode)

Poisson arrivals for about 15 s per load level, prompts 128 to 1,024 tokens, outputs 32 to 256. SLO: TTFT at most
1,000 ms and mean TPOT at most 80 ms. "Shed" drops a request that has waited more than 1,000 ms in the queue.
Goodput is requests per second that met both SLOs. Each point is one run, so expect noise of a few req/s.

| Offered (req/s) | Accept all: throughput | Accept all: goodput | Shed: throughput | Shed: goodput |
|---|---|---|---|---|
| 2 | 2.1 | 2.1 | 2.1 | 2.1 |
| 4 | 3.9 | 3.9 | 3.9 | 3.9 |
| 8 | 7.7 | 5.8 | 8.0 | 8.0 |
| 12 | 10.3 | 0.4 | 11.8 | 10.8 |
| 16 | 11.4 | 0.3 | 13.7 | 11.2 |
| 24 | 13.7 | 0.4 | 18.1 | 12.0 |
| 32 | 17.8 | 0.1 | 24.0 | 20.5 |
| 48 | 22.0 | 0.5 | 22.8 | 6.5 |

Throughput rises with load all the way to 48 req/s, while goodput for "accept all" peaks at about 4 req/s and is near
zero from 12 req/s on. Past the knee every admitted request waits too long, so the system is busy and delivers
almost nothing within the SLO. Shedding keeps goodput at 8 to 12 req/s through 24 req/s. The 20.5 req/s at 32 and the
drop to 6.5 at 48 show the noise and the limit of a simple queue-time cutoff.

### Not reproduced or not measured

- Anything at the scale of the papers (the 60 to 80% waste figure from PagedAttention, Sarathi-Serve, DistServe, Mooncake
  and so on). Our KV waste numbers come from a synthetic trace, not from measuring vLLM.
- Section 10 and 11 (disaggregation, KV transfer), 12 and 13 (distributed serving, topology), 16 (kernel fusion),
  17 (quantized serving), 18 (multi-node routing), SGLang's radix-tree structure and structured output, and the
  scheduler overlap in section 22 are in the post but not built here. Request grouping by shared prefix is shown with the
  block manager only (part B of `04_prefix_cache.py`), not with a radix tree.
- VLM serving is shown only as the image-hash cache key. There is no encoder pool and no image preprocessing.
- The attention path is not a fused paged-attention kernel, so we cannot measure the cost of the gather.
- CUDA graphs cover pure-decode iterations only. Mixed prefill iterations run eagerly, as in the post.
- All timings are one run on a shared host, in eager PyTorch. The absolute numbers are launch bound for the small
  model. Ratios between schedulers are meaningful. Absolute throughput is not comparable with a production engine.
