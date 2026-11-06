# Lab: How multi-agent systems actually coordinate

This is the code companion for the theory post
[How Multi-Agent Systems Actually Coordinate](https://srini-anirudh.github.io/blogs/how-multi-agent-systems-actually-coordinate/).
The post asks one question: when do N agents beat one agent that gets N times the compute? This lab rebuilds the
post's central quantities in small Python files, then tests two of its claims on a real small model.

What you can run:

- **The correlation ceiling.** Majority voting under a shared blind spot, as a closed form and as a simulation
  with one agent at a time. Best-of-N with a verifier is run next to it.
- **Topology and the token bill.** Table 1 of the post is recomputed from graph definitions. The lab adds the
  Figure 5 token bill and how far a prompt injection spreads along each graph.
- **Amdahl with a coordination term.** The interior optimum N* and the point where more agents make the system
  slower.
- **The split-decision lab.** Figure 8 of the post is ported line for line, so you can sweep it and test it.
- **Serial collapse.** A toy orchestrator learns how many subagents to spawn. It shows each of the three PARL reward
  terms at work.
- **Real model, voted with itself.** Qwen2.5-0.5B-Instruct answers arithmetic 15 times per question, either sampled
  or under 15 personas. The lab checks whether voting behaves like voting by independent agents.
- **Real model, one long context or a coordinator with workers.** The same model reads a record store of 450 to
  10,000 tokens. It reads it alone, or as workers that each see a slice. Lookup questions and two-hop questions
  are measured separately.

Every agent in the real-model runs is the same model. The agents differ only in what they are told and what they
read. The post argues this is the honest setting.

## Layout

```
labs/how-multi-agent-systems-actually-coordinate/
├── mascoord/
│   ├── voting.py      # majority, best-of-N, common-mode ceiling, Monte Carlo, mixed teams
│   ├── topology.py    # six topologies as graphs, token bill, prompt-infection spread
│   ├── scaling.py     # Amdahl, the coordination term, N*, break-even N
│   ├── splitlab.py    # the post's Figure 8 model and its eight presets
│   ├── parl.py        # toy orchestrator bandit with r_perf, r_parallel, r_finish
│   ├── haystack.py    # record stores, solo / one-round / two-round protocols, evidence check, oracle reader
│   ├── lm.py          # one Hugging Face chat model as every agent, with a token bill
│   └── report.py      # writes results/<script>.txt with a hardware line and the command
├── scripts/           # 01 to 07, listed below
├── tests/
└── results/           # real outputs, each starting with its hardware and command
```

## Setup

From the repository root, after `pip install -e ".[dev]"`. The real-model scripts download
`Qwen/Qwen2.5-0.5B-Instruct` (about 1 GB) from the Hugging Face Hub on first use.

## Run everything

```bash
# Tests (about 2 s on CPU). The one real-model test is marked network and slow.
python -m pytest labs/how-multi-agent-systems-actually-coordinate -m "not slow and not gpu and not network"
python -m pytest labs/how-multi-agent-systems-actually-coordinate -m "network"

# Analytic and toy scripts, CPU, seconds each (05 takes a few minutes).
python labs/how-multi-agent-systems-actually-coordinate/scripts/01_voting_ceiling.py
python labs/how-multi-agent-systems-actually-coordinate/scripts/02_topologies.py
python labs/how-multi-agent-systems-actually-coordinate/scripts/03_amdahl.py
python labs/how-multi-agent-systems-actually-coordinate/scripts/04_split_lab.py
python labs/how-multi-agent-systems-actually-coordinate/scripts/05_serial_collapse.py
python labs/how-multi-agent-systems-actually-coordinate/scripts/07_context_split.py --oracle   # perfect reader

# Real model. Use a GPU if you have one; on 8 CPU threads a 5,000-token prompt took about 29 s.
python labs/how-multi-agent-systems-actually-coordinate/scripts/06_persona_vote.py --device cuda
python labs/how-multi-agent-systems-actually-coordinate/scripts/07_context_split.py --device cuda
```

## Results

Every number names its hardware. Raw output is in `results/`, one file per script, each starting with its hardware and
command. Analytic and toy scripts ran on AMD EPYC 7R13, 8 threads, torch 2.15.0. The two real-model scripts ran on
NVIDIA H100 80GB HBM3, torch 2.15.0+cu132, CUDA 13.2, with Qwen2.5-0.5B-Instruct in float32 and greedy decoding
unless stated. Tests ran on AMD EPYC 7R13, 8 threads: 25 passed offline, 1 `network` test passed (20 s), and no
`slow` or `gpu` tests exist.

### Formulas and the post's own tables (AMD EPYC 7R13, 8 threads)

| Check | Result |
|---|---|
| Independent search, p = 0.4, N = 5 | 0.922 (post: 0.92) |
| Vote ceiling at p = 0.6, rho = 0.5 | 0.800. Closed form at N = 1001 is 0.800 and never exceeds it. |
| Closed form against a Monte Carlo of 200,000 trials, N = 5, rho = 0.35 | 0.6537 against 0.6538 |
| Table 1 at N = 7: channels, messages per round, depth | all six rows match the post |
| Token bill, N = 8, R = 4, L = 1500 | 336,000 tokens reading peers; 48,000 of work; 12.5% useful (post: about 340,000 and 12%) |
| Same bill with 180-token messages / on a blackboard | 40,320 / 11,520 tokens |
| Amdahl, f = 0.8, kappa = 0.01, alpha = 1.6 | best N = 5, speedup 2.04x; slower than one agent from N = 15 |
| Split lab presets | the verdict matches the caption for all eight scenarios |
| Prompt infection, 3 rounds, beta = 0.5, mean of 7 agents compromised, starting from a leaf | pipeline 1.0, hierarchy 3.0, blackboard 4.0, star 4.4, full mesh 7.0 |

### Serial collapse in the toy orchestrator (AMD EPYC 7R13, 8 threads, 20 seeds)

This is a bandit written for this lab. It is not Kimi's system and says nothing about its numbers.

| Reward | Final policy | Seeds that collapsed to serial | Subagents beyond the 4 real subtasks, per episode | True objective |
|---|---|---|---|---|
| r_perf only | always serial | 20 / 20 | 0.01 | 0.250 |
| plus r_parallel, never annealed | always 8 subagents | 0 / 20 | 3.82 | 0.558 |
| plus r_parallel, annealed | 4 subagents | 0 / 20 | 0.86 | 0.793 |
| plus r_parallel annealed and r_finish | 4 subagents | 0 / 20 | 0.07 | 0.794 |

Whether `r_perf` alone collapses depends on where the policy starts. With an initial P(serial) of 0.20 none of 20
seeds collapsed. At 0.40, 12 of 20 did. At 0.65 and above, all 20 did. The starting preference is an assumption of
the toy, set to mimic a model that was trained to work alone.

### Voting a real model with itself (NVIDIA H100 80GB HBM3, Qwen2.5-0.5B-Instruct)

80 arithmetic questions per task, 15 answers per question. "Sampled" is one prompt at temperature 0.8.
"Personas" is 15 different system prompts with greedy decoding. "Independent" is `P[Bin(N, p) > N/2]` using the
measured per-agent accuracy p. Strict majority counts a tie as wrong.

| Task and mode | p | Mean pairwise correlation of correctness | Strict majority, N = 15 | Independent, N = 15 | Best-of-15 with a perfect verifier |
|---|---|---|---|---|---|
| add two 3-digit numbers, sampled | 0.840 | 0.710 | 0.863 | 0.999 | 0.925 |
| add two 3-digit numbers, personas | 0.929 | 0.814 | 0.925 | 1.000 | 0.963 |
| multiply two 2-digit numbers, sampled | 0.378 | 0.846 | 0.388 | 0.164 | 0.537 |
| multiply two 2-digit numbers, personas | 0.422 | 0.895 | 0.425 | 0.267 | 0.525 |

Voting 15 agents lifted addition accuracy by 2.3 points when sampled and lowered it by 0.4 points under personas. If
the errors had been independent, the lift would have been 16 and 7 points. The agents agree on most of their mistakes.
On multiplication, 46% of questions (sampled) had all 15 answers wrong, so no vote could help. A perfect verifier
helps more than a vote, and still stops well short of the independent figure.

Two cautions. The persona row also changed the decoding (greedy, not sampled), so its higher correlation is not a clean
measure of what personas do. And the "implied rho" line in the output reads above 1 for personas on addition
(0.929 accuracy, 0.925 after voting). Read that as "voting added nothing", not as a correlation above 1.

### One long context against a coordinator (NVIDIA H100 80GB HBM3, Qwen2.5-0.5B-Instruct)

40 trials per cell, so the standard error is 0.05 to 0.08. "Tokens" are mean prompt tokens per question.
In every row each record is read once per round. The 4-worker rows cost 2% to 37% more prompt tokens than one agent,
because every worker repeats the question and the instruction. The 12 and 24 slice rows cost 13% to 14% more.

Lookup question, answer in one record:

| Records (about prompt tokens for one agent) | One agent | 4 workers, aggregated by vote | 4 workers + evidence check | Slices of 32 records + evidence check |
|---|---|---|---|---|
| 32 (500) | 0.75 | 0.47 | 0.88 | 0.75 (same as one agent) |
| 128 (1,800) | 0.35 | 0.20 | 0.72 | 0.72 (same as the 4-worker row) |
| 384 (5,300) | 0.20 | 0.20 | 0.57 | 0.82 (12 calls) |
| 768 (10,500) | 0.12 | 0.15 | 0.38 | 0.70 (24 calls) |

What the table shows:

- One agent loses accuracy as its context grows: 0.75 at 500 tokens, 0.12 at 10,500. This is the eta(L) term of the post
  measured on a small model.
- Splitting the store without a check made things worse at short length. When the answer is in one slice, the other
  three workers have nothing to find and often invent a code. With 4 workers the coordinator saw conflicting values in
  78% to 100% of questions, and a plain vote picked the wrong one.
- A non-model check fixes most of it. Keeping a worker's code only if its own slice has a record naming the project
  and containing the code takes 4 workers from 0.47 to 0.88 at 32 records. This is the post's "evidence pointer, verify
  before propagate".
- Slices that stay short win as the store grows (0.70 at 10,500 tokens against 0.12 for one agent). That is the
  context argument. It is also the token argument: those slices cost 14% more prompt tokens, not 4x.

Two-hop question, the room of a project leader (two facts in different records):

| Records | One agent | 4 workers, 1 round | 4 workers, 2 rounds |
|---|---|---|---|
| 32 | 0.05 | 0.03 | 0.12 |
| 128 | 0.00 | 0.03 | 0.03 |
| 384 | 0.00 | 0.00 | 0.00 |
| 768 | 0.00 | 0.00 | 0.03 |

The 0.5B model cannot do the two-hop question even at 450 tokens, so these rows sit at the floor. They show nothing
about whether splitting hurts interdependent work. The structure of that claim is shown by the rule-based reader
(`results/07_context_split_oracle.txt`, AMD EPYC 7R13, 8 threads), which reads perfectly: one agent 1.00,
four workers in one round 0.23 to 0.28, four workers with a planned second round 1.00 at about twice the tokens of the one-round split. A
single round fails because the two facts land in the same slice only about one time in four.

## Post sections and scripts

| Post section | Script or module |
|---|---|
| The baseline you have to beat; Three real reasons to add an agent (independent search) | `01_voting_ceiling.py`, `voting.py` |
| The crowd is only wise if it disagrees | `01_voting_ceiling.py`, `06_persona_vote.py` |
| Context, not parallelism | `07_context_split.py`, `haystack.py` |
| Topology is the architecture | `02_topologies.py`, `topology.py` |
| Talking has a bill | `02_topologies.py` (`comm_bill`) |
| Shared state and the reconciler (verify before propagate) | `07_context_split.py` (evidence check) |
| Amdahl, Brooks, Coase | `03_amdahl.py`, `scaling.py` |
| How these systems actually fail (the security surface) | `02_topologies.py` (`spread`) |
| Learned orchestration (serial collapse, reward shaping) | `05_serial_collapse.py`, `parl.py` |
| Lab: should you split this task? | `04_split_lab.py`, `splitlab.py` |

## What was not measured

- No paper result was reproduced. Not Tran and Kiela's matched-budget comparison, Smit et al. on debate, Mixture-of-Agents,
  Anthropic's research system, Cursor's swarm, Cognition's reviewer, or Kimi's PARL. The post quotes them. This lab
  shows mechanisms only, at a scale of one 0.5B model and toy simulations.
- The split-decision lab and the Amdahl model carry the post's made-up constants. They were ported and tested, not
  calibrated.
- The prompt-infection simulation is a graph reachability model with a per-message probability. It does not run
  injected prompts through a model.
- No debate, no Mixture-of-Agents, no learned orchestrator on a real model, and no heterogeneous models. The "Self-MoA"
  table in script 01 is a toy of the post's argument, not a test.
- The two-hop result is at the model's floor, as described above.
- Only one small model was used. A stronger reader would hold up at longer lengths. This lab does not say where
  its curve moves.
- The persona comparison confounds persona with decoding mode.
- Numbers come from one run of each real-model script. Standard errors for `07` are in the table above. Script 06 has
  no confidence intervals.
