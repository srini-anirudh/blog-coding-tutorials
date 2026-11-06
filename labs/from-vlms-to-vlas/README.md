# Lab: From VLMs to VLAs

A code companion for the post
[From VLMs to VLAs: Vision-Language-Action Models for Robot Control](https://srini-anirudh.github.io/blogs/from-vlms-to-vlas/).

The post asks how a vision-language model is turned into a policy that sends fast, continuous motor commands. This
lab builds the post's central mechanisms from scratch and runs them on small tasks, so you can watch each one happen.
Everything runs on a CPU in a few minutes.

- Compounding error in behaviour cloning, and the two remedies: fewer decisions (chunks) and recovery data (DAgger).
- Multimodal demonstrations: an MSE head drives into the obstacle. Bins and flow matching go around it.
- Action tokens: 256 bins with percentile ranges, RT-2-style reuse of rare vocabulary ids, the information each bin
  token carries as the control rate rises, and the FAST tokenizer (DCT, rounding, byte-pair encoding).
- A π0-style action expert: two weight sets sharing attention, a block-wise causal mask and a prefix cache that is
  reused for ten flow-matching steps.
- A 50 Hz control loop for chunk size against inference latency (the post's Figure 3).
- A toy of knowledge insulation: train a new action expert with and without a stop-gradient into the backbone.
- Trial statistics: intervals, power and the number of robot trials a comparison needs.

No real VLM is turned into a VLA here, and no robot or simulator is used. The tasks are small and synthetic, so the
numbers show the direction of each effect, not the papers' results. Building a VLM itself is covered by
[C7](../../c07-vlm/) and attention by [C3](../../c03-attention/).

## Layout

```
labs/from-vlms-to-vlas/
├── vlalab/
│   ├── envs.py         # LaneTask (compounding error) and ObstacleTask (bimodal demos), with scripted experts
│   ├── heads.py        # MSE regression, 256-bin softmax and flow-matching heads, and a training loop
│   ├── tokenizers.py   # UniformBinner, least-used vocabulary ids, DCT, BPE, FASTTokenizer, entropy per token
│   ├── expert.py       # block-wise causal mask and the shared-attention action expert with a prefix cache
│   ├── loop.py         # the 50 Hz chunked control loop: sync, async and blended execution
│   ├── insulation.py   # the knowledge-insulation toy: pointing pretraining, then robot training variants
│   ├── evalstats.py    # normal and Wilson intervals, power, trials needed, simulated checks
│   └── report.py       # writes results/ files with a hardware line and the command first
├── scripts/            # 01 to 07, listed below
├── results/            # the output of every script, as .txt and .json
└── tests/
```

## Setup

From the repository root, after `pip install -e ".[dev]"`. Only `03_action_tokens.py --vocab` and the `network` test
download anything: the SmolLM2-135M tokenizer and WikiText-2 from the Hugging Face Hub.

## Run everything

```bash
# Tests. The offline set takes about 10 s.
python -m pytest labs/from-vlms-to-vlas -m "not slow and not gpu and not network"
python -m pytest labs/from-vlms-to-vlas -m network     # tokenizer demo, needs the Hub
python -m pytest labs/from-vlms-to-vlas -m slow        # every script end to end with small settings

# Scripts. Each writes results/<name>.txt and results/<name>.json.
python labs/from-vlms-to-vlas/scripts/01_compounding_error.py
python labs/from-vlms-to-vlas/scripts/02_multimodal_demos.py
python labs/from-vlms-to-vlas/scripts/03_action_tokens.py --vocab   # drop --vocab to stay offline
python labs/from-vlms-to-vlas/scripts/04_chunk_latency_loop.py
python labs/from-vlms-to-vlas/scripts/05_action_expert.py
python labs/from-vlms-to-vlas/scripts/06_trial_statistics.py
python labs/from-vlms-to-vlas/scripts/07_knowledge_insulation.py
```

Wall times on AMD EPYC 7R13, 8 threads, on a shared machine: 28 s, 59 s, 53 s, 10 s, 23 s, 7 s and 1757 s. The last is the knowledge-insulation toy; it took 29 minutes on a machine with a load average
near 250, so it is over the 20-minute budget. `--pretrain-steps`, `--robot-steps` and `--seeds` shorten it.

## Which script covers which section of the post

| Post section | Script | What it shows |
|---|---|---|
| The control problem: why errors compound | `01_compounding_error.py` | The ε·T² worst case, chunking, behaviour cloning against DAgger |
| Demonstrations disagree with each other | `02_multimodal_demos.py` | MSE averages left and right and hits the obstacle |
| RT-2: actions as text tokens; Bins as tokens | `03_action_tokens.py` part A | Rounding error, rare ids reused, logit mask |
| Bins as tokens: the marginal information argument | `03_action_tokens.py` part B | Bits per token from 5 to 200 Hz |
| Compressing chunks: FAST | `03_action_tokens.py` part C | Tokens per one-second chunk, reconstruction error |
| Generating chunks: diffusion and flow matching | `02_multimodal_demos.py`, `vlalab/heads.py` | π0's τ distribution, ten Euler steps |
| Predicting chunks; real-time chunking | `04_chunk_latency_loop.py` | Calls, pauses, reaction and jumps against chunk size |
| The action expert; the latency budget | `05_action_expert.py` | Exact prefix cache, 6.6× fewer milliseconds for ten steps |
| Pretrained knowledge can be damaged | `07_knowledge_insulation.py` | Joint flow-matching training erases the backbone's pointing skill; stop-gradient or co-training keeps it |
| Evaluating a robot policy: how many trials | `06_trial_statistics.py` | ±21.9 points at 20 trials, 385 trials for 50% against 60% |

## Results

Tests on AMD EPYC 7R13, 8 threads, torch 2.15.0: 22 offline tests pass, 1 `network` test passes and 7 `slow`
tests pass (every script end to end with small settings). The offline set takes about 12 s.

All numbers below are from AMD EPYC 7R13, 8 threads, torch 2.15.0, on a machine shared with other jobs. Timings are
medians and should be read as orders of magnitude. Every other number does not depend on load. Raw output is in
`results/`.

### Compounding error (`01`)

Part A simulates the post's worst case: each decision on demonstrated states is wrong with probability ε = 0.01, and
after the first wrong decision every step is a mistake. 200,000 Monte Carlo episodes match the exact sum.

| Steps T | Actions per decision H | Decisions | Mistaken steps | ε·T |
|---|---|---|---|---|
| 100 | 1 | 100 | 37.2 | 1 |
| 500 | 1 | 500 | 401.9 | 5 |
| 500 | 10 | 50 | 108.3 | 5 |
| 500 | 50 | 10 | 26.4 | 5 |

At T = 500 the bound ε·T²/2 = 1250 exceeds T itself, so the count saturates. The shape is still clear: one-second
chunks at 50 Hz cut the mistaken steps from about 402 to 26.

Part B trains real policies on a lane-keeping task. The expert cancels the lane's drift and steers back. The demos
start on the lane centre, so they never show a recovery (|y| stays under 0.005). At test time every policy gets the
same small push each step.

| Policy | Mistaken steps, T = 25 | T = 100 | T = 400 |
|---|---|---|---|
| Expert | 0.35 | 1.59 | 7.01 |
| Behaviour cloning | 4.02 | 20.0 | 88.4 |
| DAgger, one round of expert labels on the learner's states | 0.34 | 1.58 | 6.98 |

Behaviour cloning makes about 12 times more mistakes than the expert. One round of DAgger closes the gap. The ACT
chunking ablation (1% to 44%) is not reproduced.

### Multimodal demonstrations (`02`)

Half the demos pass the obstacle on the left, half on the right. 500 rollouts per head.

| Head | Collision rate | Left / right |
|---|---|---|
| MSE regression | 0.860 | 0.036 / 0.104 |
| MSE regression, chunk of 8 | 0.902 | 0.028 / 0.070 |
| 256 bins per dimension | 0.004 | 0.416 / 0.580 |
| Flow matching, 10 Euler steps | 0.000 | 0.400 / 0.600 |
| Flow matching, chunk of 8 | 0.028 | 0.422 / 0.550 |

At the start the MSE head's mean lateral step is 0.0014, where every demo steps ±0.0257. It predicts the average of
the two modes and drives into the obstacle. Chunking does not fix this. Bins and flow matching keep both modes.

One detail mattered for flow matching. With τ fed in as a single raw number and a 128-wide MLP, flow matching
collided in about 20% of rollouts, because its samples blurred between the modes. Sinusoidal τ features and a
256-wide, 4-layer MLP brought it to 0%. The regression and bin heads use the smaller MLP.

### Action tokens (`03`)

| Measurement | Result |
|---|---|
| Largest bin rounding error, 256 bins | 0.00195 of the range (half a bin, 1/512) |
| Test values inside the 1st–99th percentile range | 98.4% |
| SmolLM2 ids picked as action bins (256 least used on WikiText-2) | all 256 never occur in its 2.54 M tokens |
| H(b_t \| b_t−1), one arm dimension: 5 / 50 / 200 Hz | 6.51 / 4.67 / 2.77 bits (marginal H(b) 7.76) |
| FAST tokens per 7-D one-second chunk at 50 Hz, scale 10 | 56.1 on average (38–76), against 350 bin tokens |
| FAST arm RMSE from rounding alone, scale 5 / 10 / 20 | 0.0133 / 0.0078 / 0.0045 of the range |

The conditional entropy is a plug-in estimate, which is biased low when pairs are sparse. That bias is largest at
low rates, so the true fall with frequency is, if anything, larger. The actions are synthetic smooth curves, so the
token counts are not comparable with the 30 to 60 tokens the FAST paper reports on robot data. FAST's training
speed-up and π0-FAST's 750 ms decoding were not measured.

### Chunk size and latency (`04`)

A toy loop at 50 Hz. The policy extrapolates the object's motion, and the object is bumped once. Mean over 20 bump
times. 80 ms inference latency:

| Mode | H | Calls per s | Time paused | Reaction (ms) | Hand step at switch |
|---|---|---|---|---|---|
| Wait for each chunk | 1 | 10.0 | 80% | 120 | 0.037 |
| Wait for each chunk | 10 | 3.75 | 30% | 206 | 0.054 |
| Wait for each chunk | 50 | 1.0 | 8% | 610 | 0.220 |
| Compute next chunk while executing | 10 | 5.0 | 2% | 170 | 0.035 |
| Compute next chunk while executing | 50 | 1.0 | 2% | 560 | 0.215 |
| Same, cross-fade over 5 overlapping steps | 50 | 1.25 | 2% | 475 | 0.038 |

The usual step elsewhere is about 0.005, so a switch between independently sampled chunks is a jump of 4 to 40
normal steps. The cross-fade is a simple stand-in for real-time chunking, not the RTC algorithm, which inpaints the
new chunk instead. Other latencies are in `results/04_chunk_latency_loop.txt`.

### Action expert and prefix cache (`05`)

Six layers. Backbone width 512 (18.9 M parameters), expert width 256 (6.4 M), 8 shared heads of 64. Sequence: 816
prefix tokens, 1 state token and 50 action tokens. Random weights.

| Measurement | Result |
|---|---|
| Max difference, cached against full pass | 4.8e-7 |
| Change in prefix outputs when actions and τ change | 0 exactly |
| Ten flow steps, full pass each step | 2449 ms |
| Prefix once + ten expert steps | 373 ms (186 prefix + 187 expert), 6.6× faster |

π0 reports 73 ms on an RTX 4090 for a 3.3 B model. That was not reproduced.

### Knowledge insulation (`07`)

A small transformer with two weight sets (the `05` expert, trained). Scene: 4 objects with one of 20 colours each.
Instruction: go to the object of colour c. Stage 1 ("VLM pretraining") teaches the backbone to point at the named
object, for all 20 colours. Stage 2 ("robot training") trains a new, randomly initialised action expert by flow
matching, on colours 0 to 4 only. 2 pretraining seeds, 2000 and 1500 steps. Errors are mean distances in [-1, 1]²:
a chunk that does not move scores 1.043, and going to a random other object scores 1.041.

| Stage-2 variant | Pointing error, all 20 colours | Action error, colours 0–4 | Action error, colours 5–19 |
|---|---|---|---|
| Before stage 2 | 0.014 | not trained | not trained |
| Joint: backbone and expert trained on the flow loss | 0.803 (0.706–0.900) | 0.037 | 0.523 |
| Joint plus pointing co-training on all colours | 0.022 | 0.033 | 0.545 |
| Insulated: stop-gradient into the backbone plus co-training | 0.019 | 0.046 | 0.574 (0.516–0.632) |
| Backbone frozen | 0.014 | 0.048 | 0.645 |

Training the new expert jointly, with nothing to protect the backbone, destroys its pointing skill: the error rises
from 0.014 to 0.80, which is close to guessing. Either co-training on the pointing data or a stop-gradient keeps it.
This is the part of the post's claim that the toy reproduces.

The toy does not reproduce the claim about following instructions. The action error on the 15 colours that only
pretraining saw stays between 0.52 and 0.65 in every variant, even when the backbone is frozen and still points
correctly. The expert learned from five colours does not read the other ten, so there is nothing for insulation to
preserve there. The differences among the four variants (0.52 to 0.65) are small against the seed ranges, so they
give no ranking. The stop-gradient here only blocks the expert's gradient through the cached keys and values; the real
method also trains the backbone on FAST tokens, which this toy leaves out.

### Trial statistics (`06`)

| Measurement | Result |
|---|---|
| 95% interval half-width, 20 trials at 50% | 21.9 points (Wilson 29.9%–70.1%) |
| Trials per policy for 80% power, 50% against 60% | 385; simulated power at that n is 0.795 |
| Power with 20 trials per policy, 50% against 60% / 70% / 80% | 0.096 / 0.248 / 0.512 (simulated 0.087 / 0.241 / 0.532) |
| Coverage of the normal interval at 10 trials | 0.89, below the nominal 0.95 (Wilson 0.978) |

## What was not measured

- No pretrained VLM was fine-tuned into a VLA. None of the papers' success rates were reproduced: not RT-2's
  ablation, OpenVLA, ACT's chunking ablation, FAST's training speed-up, π0's latency or knowledge insulation's real
  results.
- No robot, simulator or benchmark (LIBERO, CALVIN, SIMPLER) was used.
- Not covered by code: cross-embodiment action spaces, dual-system architectures, reading from middle layers,
  robot data sources, reinforcement learning for VLAs, video world models and safety.
- No GPU numbers. The post's claims checked here are about mechanisms, not speed, so CPU runs suffice. The timing in
  `05` shows the shape of the cache saving only.
