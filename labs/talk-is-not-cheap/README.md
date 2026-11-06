# Lab: Talk Is Not Cheap

Code companion for the post
[Talk Is Not Cheap: Communication from PCIe to InfiniBand](https://srini-anirudh.github.io/blogs/talk-is-not-cheap/).

The post builds a cost model for moving bytes between GPUs and uses it to derive the collectives. This lab lets you
run those ideas. You can:

- fit the α–β model to a real ping-pong, on CPU processes and on two H100s;
- write naive, tree, ring, recursive-doubling and Rabenseifner AllReduce as explicit send/receive schedules;
- check that every schedule gives the exact sum, and count its bytes against the 2(p−1)/p·n bound;
- run the same schedules for real with `torch.distributed`, and see where the model's ranking holds and where it fails;
- price a flat ring against a hierarchical AllReduce on a two-level machine;
- model gradient bucketing, ECMP hash collisions and MoE AllToAll imbalance;
- measure HBM, PCIe and NVLink copies, NCCL's LL, LL128 and Simple protocols, and what overlap costs a matmul.

The post quotes cluster-scale numbers (64 to 1024 ranks, InfiniBand and EFA fabrics). None of those were reproduced
here. The largest real run is 8 CPU processes, or 2 GPUs in one node. Everything at cluster scale is a model, and the
results say so.

## Layout

```
labs/talk-is-not-cheap/
├── commcost/
│   ├── ladder.py      # the bandwidth ladder, break-even FLOP/byte, PCIe encoding, link counts
│   ├── alphabeta.py   # T = α + βn, the fit, n½, closed-form cost of each AllReduce
│   ├── schedules.py   # every collective as rounds of point-to-point transfers
│   ├── simulate.py    # run a schedule on numpy: check it, count bytes, price it per link
│   ├── dist_run.py    # run the same schedule with isend/irecv on gloo or NCCL
│   ├── workers.py     # what each spawned process does: ping-pong, AllReduce timing, NCCL sweeps, overlap
│   ├── overlap.py     # backward-pass timeline with gradient buckets
│   ├── fabric.py      # ECMP balls in bins, packet spraying, MoE routing imbalance
│   └── report.py      # results files with a hardware line and the command
├── scripts/           # 01 to 09, listed below
├── tests/
└── results/           # real outputs, each starting with "# hardware:" and "# command:"
```

The schedule is the central idea. One list of `Transfer(src, dst, lo, hi, op)` rounds is executed three ways: on
numpy arrays to check the answer and count bytes, under a link-cost function to price it, and over real processes.
So the bytes the tests count are the bytes the real run sends.

## Setup

From the repository root, after `pip install -e ".[dev]"`. Only `torch` and `numpy` are needed. No downloads.

## Run everything

```bash
# Tests. The offline set spawns gloo processes and takes about 20 s.
python -m pytest labs/talk-is-not-cheap -m "not slow and not gpu and not network"
python -m pytest labs/talk-is-not-cheap -m gpu            # needs 2 GPUs

# Models and arithmetic. Each takes a few seconds on one core.
python labs/talk-is-not-cheap/scripts/01_ladder.py
python labs/talk-is-not-cheap/scripts/03_allreduce_schedules.py
python labs/talk-is-not-cheap/scripts/05_hierarchical.py
python labs/talk-is-not-cheap/scripts/06_bucketing.py
python labs/talk-is-not-cheap/scripts/07_ecmp.py
python labs/talk-is-not-cheap/scripts/08_alltoall_moe.py

# Real communication on CPU processes (gloo over loopback). About 4 minutes on 8 cores.
python labs/talk-is-not-cheap/scripts/02_pingpong.py
python labs/talk-is-not-cheap/scripts/04_allreduce_measured.py

# Real communication on 2 GPUs (NCCL). About 6 minutes in total.
CUDA_VISIBLE_DEVICES=0,1 python labs/talk-is-not-cheap/scripts/02_pingpong.py --backend nccl
CUDA_VISIBLE_DEVICES=0,1 python labs/talk-is-not-cheap/scripts/04_allreduce_measured.py --backend nccl -p 2 --max-log2 28
CUDA_VISIBLE_DEVICES=0,1 python labs/talk-is-not-cheap/scripts/09_gpu_links.py
```

## Which script covers which section of the post

| Post section | Script |
|---|---|
| Communication is another memory hierarchy (the ladder, break-even intensity) | `01_ladder.py`, `09_gpu_links.py` part 1 |
| Why communication costs time; the α–β model, n½ | `02_pingpong.py` |
| Representative values, bandwidth-delay product | `01_ladder.py` |
| PCIe (lane arithmetic, 128b/130b) | `01_ladder.py`, `09_gpu_links.py` part 2 |
| NVLink, NVSwitch (9× cliff, full-mesh link counts) | `01_ladder.py`, `09_gpu_links.py` part 3 |
| Collective communication; AllReduce = ReduceScatter + AllGather | `03_allreduce_schedules.py` |
| Deriving AllReduce: naive, tree, ring, the lower bound, Rabenseifner | `03_allreduce_schedules.py`, `04_allreduce_measured.py` |
| What NCCL does: LL, LL128, Simple (nuance 08) | `09_gpu_links.py` part 4 |
| All-to-All, MoE imbalance (nuances 10, 11) | `08_alltoall_moe.py` |
| Hierarchical collectives (nuance 12); in-network reduction (nuance 13) | `05_hierarchical.py`, `03_allreduce_schedules.py` part 6 |
| Overlap and bucketing (nuances 14, 15) | `06_bucketing.py`, `09_gpu_links.py` part 5 |
| Congestion: ECMP and packet spraying (nuance 16) | `07_ecmp.py` |

## Results

Every number names its hardware. Raw output is in `results/`, under the name of the script that wrote it.

Tests: 80 passed offline on AMD EPYC 7R13 (`results/tests.txt`). The one `gpu` test passed on 2 x NVIDIA H100 80GB
HBM3 (`results/tests_gpu.txt`). There are no `network` or `slow` tests.

### Measured on 2 x NVIDIA H100 80GB HBM3, torch 2.15.0, CUDA 13.2, NCCL 2.30.7

The two GPUs are in one node with NVLink and peer access on. The node was shared with other jobs on other GPUs, and its
CPUs were heavily loaded. Small-message times depend on the CPU, so treat them as upper bounds.

| Measurement | Result | Spec quoted in the post |
|---|---|---|
| HBM copy on one GPU, read + write, 1 GiB | 2,883 GB/s | 3,350 GB/s |
| GPU 0 to GPU 1 copy, 1 GiB | 384 GB/s | 450 GB/s per direction |
| Host to GPU, pinned / pageable, 1 GiB | 14.4 / 12.1 GB/s | 63 GB/s (Gen5 x16) |
| GPU to host, pinned / pageable, 1 GiB | 1.9 / 1.3 GB/s | 63 GB/s |
| NCCL send/recv ping-pong fit | α = 61 µs, 1/β = 288 GB/s, n½ = 16.8 MiB | α ≈ 2 µs, 370 GB/s |
| `all_reduce`, 1 KiB, library | 0.09 ms | |
| Our schedules over NCCL, implied α per round | 160 to 220 µs | |
| `all_reduce` bus bandwidth at 1 GiB: LL / LL128 / Simple | 120 / 301 / 278 GB/s | LL is 50% efficient on the wire |
| 20 bf16 8192³ matmuls alone | 31.6 ms (614 TFLOP/s) | |
| 1 GiB `all_reduce` alone | 3.4 ms | |
| Both, one after the other / on separate streams | 34.3 / 32.3 ms | |
| Matmul slowdown while AllReduces run beside it (CUDA events) | +0.7% (first run: +1.3%) | "a few percent" |

How to read it:

- **The software α is the whole story for small messages.** The fitted NCCL α is 61 µs, about 30 times the post's 2 µs.
  That is not NVLink. It is Python, `ProcessGroupNCCL` and a busy CPU, which is the post's nuance 02. Our own schedules
  pay 160 to 220 µs per round, so on 2 GPUs the library's fused `all_reduce` beats every schedule we wrote at every size.
- **LL plateaus at about 40% of LL128.** LL stops near 120 GB/s from 16 MiB upwards, while LL128 and Simple keep
  climbing to roughly 300 GB/s. That is nuance 08, measured. In the first full run (`results/09_gpu_links_run1.txt`)
  Simple reached 321 GB/s at 1 GiB, so LL128 against Simple at the top is within run-to-run noise.
- **Overlap hides most of the AllReduce.** Separate streams save 2.0 of the 3.4 ms. The matmul slowed by about 1% while
  the AllReduce ran. This is 2 GPUs over NVLink, where NCCL uses few channels. At larger scale the cost can be higher;
  this run does not show it.
- **PCIe was far below spec on this node.** Host to GPU was 14 GB/s and GPU to host under 2 GB/s, pinned, on both GPUs.
  We did not find why. A shared node with a busy host is the likely cause, but it was not confirmed. Do not read these
  as H100 PCIe numbers.

### Measured on AMD EPYC 7R13, 8 single-thread processes, gloo over TCP loopback

The host had a load average near 600 on 192 CPUs during this run. Every time below is inflated by scheduling delay.

| Measurement | Result |
|---|---|
| 2-process ping-pong fit | α = 219 µs, 1/β = 2.1 GB/s, n½ = 453 KiB |
| All 5 AllReduce versions, 8 sizes, 8 ranks | exact sum on every rank, all 40 runs |
| Fastest of our schedules: 4 KiB to 1 MiB / 4 MiB / 16 MiB | tree / Rabenseifner / ring |
| 16 MiB: naive / tree / ring / Rabenseifner / library, ms | 293 / 183 / 119 / 139 / 92 |
| Times the α–β model (from the ping-pong) picked the measured winner | 1 of 8 sizes |
| α implied per round, refit to each algorithm's own timings | 0.8 ms (naive) to 3.3 ms (ring) |

The shape of the post's argument shows up: the full-vector schemes win small, the bandwidth-optimal ones win large.
The model's details do not. It says Rabenseifner should win everywhere, because it has the tree's round count and the
ring's bytes. In practice the tree wins small messages, because only half its ranks talk in each round, and 8
processes on one loaded host contend for the same CPUs and the same loopback stack. The α–β model assumes independent
links. A loaded host is the opposite. Earlier trial runs on the same host, at lower load, fitted α between 45 and
226 µs. So α here measures the load as much as the transport.

### Models and arithmetic (no measurement)

Exact byte counts from the simulator, and α–β times. These come from the post's own formulas and spec values.

| Claim checked | Result |
|---|---|
| Ring and Rabenseifner bytes per rank | exactly 2(p−1)/p·n at p = 4 to 64 |
| Naive root at p = 64, n = 100 MB | 12.6 GB at the root, 0.2 GB elsewhere |
| Simulator time against closed form, all 4 algorithms | equal |
| Tree/ring crossover at p = 64, α = 12 µs, 45 GB/s | 5.9 MiB (the post's text says about 100 KiB) |
| Same, α = 2 µs | 1.0 MiB (moves left, as the post says) |
| Ring α term at p = 1024, α = 12 µs | 24.6 ms |
| Flat ring vs hierarchical, 1 GiB, 4 nodes x 8 GPUs | 41.8 vs 8.2 ms |
| Busiest NIC bytes cut by hierarchy, 4 x 8 | 10.33x, equal to G(P−1)/P·N/(N−1) |
| Total bytes on the fabric cut by hierarchy, 4 x 8 | 1.29x |
| Hierarchy with in-node links as slow as the network | 41.8 vs 41.7 ms: the traffic cut alone buys almost nothing |
| NVLink hops busy in a flat ring, 4 x 4 | 11% of each round |
| Best bucket, p = 64 at 45 and 22.5 GB/s / p = 1024 | 800 MB, 800 MB / 3,200 MB |
| Exposed tail at p = 1024 with 25 MB buckets vs the best bucket (291-tensor 7B model) | 5.66 s vs 0.35 s |
| Busiest of 64 uplinks with 64 hashed flows, mean of 20,000 trials | 3.95 flows (ln k / ln ln k gives 2.92) |
| Same, with 1000 sprayed packets per flow | 1.08x an even share |
| AllToAll vs hierarchical AllReduce on 4 x 8: busiest NIC bytes | 0.75n vs 0.19n |
| MoE router skew 1.0: busiest rank / mean; with capacity 1.25 | 1.99x; 1.54x and 51% of tokens dropped |

Four places where the lab disagrees with the post's text:

1. **Tree/ring crossover.** By the post's own formulas, the ones Figure 14 uses, the ring overtakes the tree near
   5.9 MiB at p = 64, α = 12 µs, 45 GB/s. That is about 60 times the "roughly 100 KiB" in the text.
2. **ECMP.** ln k / ln ln k is the leading term of an asymptotic result. At k = 64 the busiest link carries about 4
   flows on average, not 2.9. The real penalty is worse than the post says.
3. **Hierarchy.** It cuts the busiest NIC's traffic by about G, which is what sets the time. It barely changes the
   total bytes on the fabric: the flat ring squeezes them through the N boundary NICs, and the hierarchical version
   spreads them over all G·N NICs. So "8× less fabric traffic" is true per NIC, not for the fabric as a whole.
4. **Overlap cost.** At 2 GPUs we measured about 1% matmul slowdown, below "a few percent". This does not contradict
   the post at scale. It is just what this setting shows.

## What was not measured

- Anything across nodes: InfiniBand, EFA, RDMA, GPUDirect, rails, switches, SHARP, PFC and incast. There is one node.
- Any collective on more than 2 GPUs. The ring, tree and hierarchical comparisons on GPUs are not meaningful at p = 2,
  so the GPU results only cover the protocols, the copies and overlap.
- NCCL's own algorithm choice (`NCCL_ALGO`). With 2 GPUs in one node there is nothing to compare.
- The post's latency ladder (Figure 1) and its per-component breakdown of α (Figure 2). They are not measurable from
  PyTorch.
- Bucketing on a real DDP run. `06_bucketing.py` is a timeline model only.
- The CPU runs on a quiet machine. Both CPU hosts were heavily loaded during this lab.
