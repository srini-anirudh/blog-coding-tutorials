# Lab: Memory Is More Than Context

This lab is the code companion to the theory post
[Memory Is More Than Context](https://srini-anirudh.github.io/blogs/memory-is-more-than-context/).
It takes seven of the post's central claims and makes each one run. Five are small, exact Python models of a
mechanism. Two use a real small language model, Qwen2.5-0.5B-Instruct.

The tutorial [C11: agent memory](../../c11-agent-memory/) already builds a full memory system for the C10 agent:
a writer, fact supersession, hybrid retrieval, a context packer and a four-condition evaluation. This lab does not
rebuild those. It covers what the post adds on top: non-additive context utility, position effects, the staleness
law, the poisoning fixed point, the score as a log-odds, a store with two clocks, and the full five-condition ladder.

Part of [blog-coding-tutorials](https://github.com/srini-anirudh/blog-coding-tutorials).

## Layout

```
labs/memory-is-more-than-context/
├── memlab/
│   ├── allocation.py   # Figure 3: top-k, density and marginal-gain selection, exact optimum
│   ├── position.py     # Figure 4: the Lost in the Middle key-value task
│   ├── lm.py           # a small causal-LM wrapper: greedy check without generating, attention share
│   ├── staleness.py    # Figure 10: the 1/(lambda t) law, simulated stores, precision at fixed k
│   ├── poisoning.py    # Figure 13: the experience-following fixed point and a simulated agent loop
│   ├── scoring.py      # Figure 11: the score as naive-Bayes log-odds, fitted weights, thresholds
│   ├── bitemporal.py   # Figure 9: append, replace, supersede, invalidate, and real deletion
│   ├── ladder.py       # Figure 14: the five-condition ablation ladder
│   └── hardware.py     # the hardware line at the top of every results file
├── scripts/            # one script per mechanism, 01 to 07
├── results/            # the real outputs of those scripts
└── tests/
```

## Setup

From the repository root, after `pip install -e ".[dev]"`. Scripts 02 and 07 download Qwen2.5-0.5B-Instruct
(about 1 GB) from the Hugging Face Hub on first use. If your default cache is not writable, set `HF_HOME`.

## Run everything

```bash
python labs/memory-is-more-than-context/scripts/01_allocation.py   # seconds
python labs/memory-is-more-than-context/scripts/02_position.py     # about 35 min on a heavily shared CPU node
python labs/memory-is-more-than-context/scripts/03_staleness.py    # seconds
python labs/memory-is-more-than-context/scripts/04_poisoning.py    # about 20 s
python labs/memory-is-more-than-context/scripts/05_scoring.py      # about 10 s
python labs/memory-is-more-than-context/scripts/06_bitemporal.py   # instant
python labs/memory-is-more-than-context/scripts/07_ladder.py       # about 40 min on a heavily shared CPU node

# Optional, a larger position grid on a GPU (not yet measured):
python labs/memory-is-more-than-context/scripts/02_position.py --device cuda \
    --pairs 10 30 75 --trials 40 --name 02_position_gpu

# Tests
python -m pytest labs/memory-is-more-than-context -m "not slow and not gpu and not network"   # about 5 s
python -m pytest labs/memory-is-more-than-context -m network                                 # loads the model
```

Each script prints its table and writes it to `results/<script>.txt`, plus a `.json` with the same numbers. The
first two lines of each file give the hardware and the command.

## Post sections and scripts

| Post section | Figure | Script | What you see happen |
|---|---|---|---|
| The allocation problem | 3 | `01_allocation.py` | Marginal gain beats top-k and density, and the figure's own optimum |
| Accessible is not usable | 4 | `02_position.py` | A real model loses a fact in the middle of a long context |
| Constructing the context | 4 | `02_position.py` (`+restate` rows) | Restating the one line near the question |
| Forgetting is a feature | 10 | `03_staleness.py` | Truth density falls as 1/(λt); precision@k falls with store size |
| When memory poisons itself | 13 | `04_poisoning.py` | The post's fixed point against an agent loop with a real store |
| Which memory comes back | 11 | `05_scoring.py` | The weighted sum is a fitted log-odds, and a threshold can abstain |
| Update is not append | 9 | `06_bitemporal.py` | Three stores, seven questions, two clocks |
| Deletion has to be real | | `06_bitemporal.py` | A deleted fact survives in a summary unless deletion chases it |
| How do you know it works? | 14 | `07_ladder.py` | All five conditions, the four gaps, and the "already knew" control |

## Results

Every CPU number below is from an AMD EPYC 7R13 limited to 8 threads, torch 2.15. The machines were shared with
many other jobs, so the wall times in the results files are much longer than the same scripts take on an idle
machine. Every other number does not depend on load: the simple models are exact or seeded, and the model runs are
greedy.

### 1. Filling a budget when utility is not additive (Figure 3)

The nine items, their utilities and the three interactions are copied from the post's figure code.

| Budget | Top-k relevance | Density u/l | Marginal gain | Exact optimum |
|---|---|---|---|---|
| B=300 | U=2.87 | U=2.87 | U=3.14 | U=3.83 |
| B=2300 (figure default) | U=1.53 | U=2.72 | U=3.14 | U=3.83 |
| B=2400 | U=3.37 | U=2.72 | U=4.09 | U=4.78 |

- The post's ordering holds. Top-k relevance spends 2,200 tokens on the runbook and starves. Density keeps the
  duplicate and the stale fact. Marginal gain drops both and takes the failure and its fix together.
- **The figure's utility has a hole.** It penalises the stale Jenkins fact (#3) only next to item #1. Item #2 says
  the same thing as #1, but has no penalty with #3. So the true optimum of the figure's own model swaps #1 for #2
  and keeps the stale fact: U=3.83 against 3.14 for marginal gain.
- With the missing #2-#3 contradiction added, marginal gain equals the exact optimum at all three budgets.
- On 500 random pools of 12 items with the same three kinds of interaction, marginal gain reaches 0.980 of the
  optimum on average (worst 0.644, optimal in 364 of 500). Top-k relevance reaches 0.795 and density 0.776. Both
  can end below zero, at -0.875, because they add a contradiction or a distractor that fits.

### 2. The same fact moved through the context (Figure 4)

Qwen2.5-0.5B-Instruct on the key-value task from Lost in the Middle: a JSON object of random UUID pairs and one key.
Accuracy is the share of trials where greedy decoding returns the value. "Attention on pair" is the attention share
that the last prompt token gives to the answering pair, averaged over all layers and heads. 8 trials per cell.

| Pairs (tokens) | Position 1 | 25% | 50% | 75% | Last | 50% with the pair restated |
|---|---|---|---|---|---|---|
| 10 (about 770) | 1.000 | 1.000 | 1.000 | 0.750 | 1.000 | 1.000 |
| 30 (about 2,120) | 1.000 | 0.875 | **0.000** | 0.375 | 1.000 | 0.875 |
| attention share, 30 pairs | 0.040 | 0.011 | 0.007 | 0.012 | 0.065 | |

- The U shape is real for this model. With 30 pairs, a value in the middle was never returned, while the first
  and last positions were always right. With 10 pairs the dip is small. A longer context makes the middle worse.
- The attention share follows the same U. It is 6 to 10 times larger at the ends than in the middle.
- Restating the one pair just before the question lifts the middle from 0/8 to 7/8. That is the post's remedy
  in section 14. The original pair still sits in the middle, and gets even less attention (0.004).
- These are 8 trials per cell on one 0.5B model, so each cell is coarse: one trial is 0.125. The paper's models,
  context lengths and numbers were not reproduced.
- A larger grid (10, 30 and 75 pairs, 40 trials per cell) is not yet measured. The command is in "Run
  everything". It was queued for an NVIDIA H100 80GB HBM3 and never got a free GPU during this session.

### 3. Truth density and precision at fixed k (Figure 10)

Figure 10's settings: 6 facts per step, λ = 0.12 per step. 20 simulated stores, each fact with an exponential lifetime.

| t | λt | Append-only, closed form | Append-only, simulated | 1/(λt) | Recall 0.7, closed form | Recall 0.7, simulated |
|---|---|---|---|---|---|---|
| 10 | 1.2 | 0.582 | 0.571 | 0.833 | 0.823 | 0.812 |
| 50 | 6 | 0.166 | 0.169 | 0.167 | 0.399 | 0.406 |
| 100 | 12 | 0.083 | 0.085 | 0.083 | 0.233 | 0.240 |

The true facts saturate at a/λ = 50 (simulated mean 50.9 at t=100) while the records grow to 600.

Precision@6 with nothing going stale (6 relevant memories scored N(1, 0.4²), the rest N(0, 0.4²)): 0.600 at
n=200, 0.359 at n=2,000 and 0.155 at n=20,000. These score distributions are a choice made here, not the post's.

### 4. The poisoning fixed point (Figure 13)

The post's three worked numbers check out (ε₀ = 0.15, φ = 0.9): e* = 0.0194 at ρ = 0.25, exactly 0.15 at ρ = 1,
and 0.8873 at ρ = 4.

The simulation runs the loop with a real store: each task copies a random stored record with probability φ, or
reasons fresh; then the gate stores the episode. 40,000 tasks, error over the last half, 20 seeds.

| φ | ρ | e* (closed form) | Simulated mean | Simulated range |
|---|---|---|---|---|
| 0.9 | 0.25 | 0.019 | 0.020 | 0.017 to 0.021 |
| 0.9 | 1 | 0.150 | 0.172 | 0.062 to 0.567 |
| 0.9 | 2 | 0.833 | 0.738 | 0.544 to 0.838 |
| 0.9 | 4 | 0.887 | 0.881 | 0.845 to 0.890 |
| 0.8 | 0.25 | 0.038 | 0.038 | 0.035 to 0.041 |
| 0.8 | 4 | 0.776 | 0.775 | 0.761 to 0.781 |

- Far from ρ = 1 the closed form predicts the simulated loop well.
- **"No gate is neutral" is true on average only.** At ρ = 1 single runs end anywhere from 0.06 to 0.57. The
  store acts like an urn: the first records set an error rate that later copies reinforce, and with φ = 0.9 the
  pull back towards ε₀ is weak. A store-everything agent is not safe. It is a coin toss on its first few tasks.

### 5. The retrieval score is a log-odds (Figure 11)

The figure's eight memories, ranked by the figure's own rule, reproduce its stages: by similarity alone the stale
Jenkins fact is second (0.91), adding recency and importance drops it to fifth, and the MMR penalty moves the
near-duplicate from second to seventh.

Then a world where the features are independent Gaussians given "useful", so the true log-likelihood-ratio slopes
are known:

| | semantic | recency | importance | bias |
|---|---|---|---|---|
| True slopes | 22.22 | 4.00 | 5.00 | -20.22 |
| Logistic regression, fitted on 200,000 labels | 22.38 | 3.90 | 5.08 | -20.28 |
| Naive Bayes | 22.23 | 4.02 | 5.00 | -20.23 |

- The fitted weights recover the log-likelihood ratios, so the weighted sum is the naive-Bayes log-odds here.
- Add a near-copy of recency ("time since last access"). Logistic regression gives it 0.29 and leaves the total
  recency weight about the same. Naive Bayes counts it twice (3.97 on top of 4.02). Among pairs it scores above
  P = 0.5, it predicts 0.912 and only 0.871 are useful, against 0.910 and 0.904 without the copy.
- Thresholding at log-odds 0, with 2,000 queries and 30% that have no useful memory: top-3 returns 3 memories
  every time, so it can never abstain. The threshold returns nothing on 82.2% of the unanswerable queries. The
  price is recall on answerable ones: 0.844 against 0.968 for top-3. It returns 0.82 memories per query, not 3.

### 6. Update is not append, and deletion has to be real (Figure 9)

Six events, including a move learned two months late and a model inference that turns out to be wrong. Seven
questions:

| Store | Correct |
|---|---|
| Append-only | 0/7 |
| Replace in place | 2/7 |
| Bitemporal, with SUPERSEDE and INVALIDATE | 7/7 |

Append-only returns both cities for "where does the user live now". Replace answers "now" but cannot answer
anything about the past. Only the store with two clocks can answer "on 1 August, where did we believe the user
lived?" (Hyderabad) and "where did they actually live?" (Bangalore). The invalidated runner image is gone from
"now" and still visible in "what did we believe on 21 September".

Deleting the diet fact without chasing derived records leaves "vegetarian" in a profile summary that a later
context can quote. Deleting with the derivation list removes it.

### 7. The ablation ladder (Figure 14)

Qwen2.5-0.5B-Instruct, greedy, 3 seeded worlds of 24 invented projects, 72 questions per condition, k=3.
Half of the projects moved region. The retriever is a bag-of-words cosine top-3 with no recency.

| Condition | Accuracy | Moved projects | Unmoved | Answer in the notes | Prompt tokens |
|---|---|---|---|---|---|
| 1 no memory | 0.000 | 0.000 | 0.000 | 0.000 | 50 |
| 2 random notes | 0.069 | 0.056 | 0.083 | 0.292 | 112 |
| 3 retrieved | 0.611 | 0.722 | 0.500 | 1.000 | 114 |
| 4 oracle | 0.917 | 0.861 | 0.972 | 1.000 | 73 |
| 5 full history | 0.667 | 0.667 | 0.667 | 1.000 | 895 |

| Gap | Value |
|---|---|
| Distraction cost, A2 - A1 | +0.069 |
| Delivered lift, A3 - A1 | +0.611 |
| Retrieval headroom, A4 - A3 | +0.306 |
| Compression loss, A5 - A4 | -0.250 |

- **Curation beats capacity here.** Oracle beats the full history by 0.25 at a twelfth of the tokens.
- The retriever always put the answer in the notes (1.000), and still left 0.306 on the table. Every miss had
  the right note next to two others, so the model read the wrong line. That is a reading failure, not a recall
  failure. Unmoved projects suffer most (0.500). For them, all 72 of the extra notes are other projects' "moved and now
  deploys to" notes, pulled in because the question also says "now".
- The distraction cost came out positive, not negative. The floor is 0 because the model never named the right
  region with no notes. Random notes show region codes, and 29% of the time the right code happens to be among
  them. On this task, the random condition measures format help as well as distraction.
- The control: of the 44 questions answered correctly with retrieved notes, none is answered correctly with the
  notes removed. Every one was a memory win, as it should be for invented projects.
- One model, one prompt, 72 questions per condition. Treat the counts as a demonstration, not an estimate.

## What was not measured

- No result from the papers the post cites was reproduced at their scale: not Lost in the Middle's models or
  context lengths, not the Xiong et al. memory-management gains, not LongMemEval or MemoryAgentBench.
- The utilities in script 01 and the score distributions in scripts 03 and 05 are hand-set or synthetic. They
  show the mechanism, not the size of any real effect.
- The poisoning model is the post's toy model. No real agent's experience-following coefficient φ was measured.
- The real-model runs use one 0.5B model, greedy decoding and small trial counts. No confidence intervals.
- Token-level compression (LLMLingua) and summarisation loss were not built. C11's `demo_compress.py` covers
  compression strategies.
- The write policy (salience, novelty and outcome-gated writes) is not rebuilt here. C11's `writer.py` and its
  evaluation cover it.
