# Lab: The Agent Evaluation Playbook

Code companion for [The Agent Evaluation Playbook](https://srini-anirudh.github.io/blogs/the-agent-evaluation-playbook/).

The post argues that an agent changes a world, so evaluation has to inspect the world and not only the agent's own
report. This lab builds that idea in small, readable Python. It has an order-management world you can snapshot,
reset and diff. A harness runs agents against it with a call budget, an optional monitor and a structured trace.
Around that sit four graders and the post's arithmetic: pass^k, horizon, grader noise, confidence thresholds,
reviewer capacity, injection monitors, memory poisoning and oracle ablation. Each piece runs and prints its result.

The agents are scripted Python functions, not language models. Each one acts out a behaviour the post describes,
for example "claims success without acting" or "obeys an injected instruction with probability q". So the lab
measures the evaluation machinery, not any model. Every model and benchmark number the post quotes, such as
τ-bench, HAL, CaMeL or TheAgentCompany, comes from a paper and was not reproduced here.

It needs only Python and numpy. No GPU, no network and no model download.

## Layout

```
labs/the-agent-evaluation-playbook/
├── agenteval/
│   ├── world.py        # state, tools, reset(s0), flatten + diff, the cancel-and-refund task
│   ├── harness.py      # Harness (budget, monitor, per-step trace), run_episode, Episode
│   ├── agents.py       # scripted behaviours, and a noisy agent with a set success probability
│   ├── graders.py      # transcript, tool-call, goal-state and state+diff graders; diff classes; attribution
│   ├── repo.py         # a tiny repo with real tests, five checkpoints, a shortcut agent
│   ├── reliability.py  # pass@k and pass^k estimators, E[p^k] approximations, hazard and H50
│   ├── noise.py        # a grader with error rates alpha and beta; runs needed to separate two agents
│   ├── oversight.py    # p*, acting policies by tier, a reviewer whose catch rate falls with load
│   ├── security.py     # provenance monitor, injection tasks, K-send tasks, memory store and poisoning
│   ├── ablation.py     # a five-seam pipeline, the oracle lattice, marginal deltas, Shapley values
│   └── report.py       # writes results/<script>.txt with a hardware line and the command
├── scripts/            # 01 to 08, listed below
├── results/            # outputs written by the scripts
└── tests/              # 18 offline tests
```

## Run everything

From the repository root, after `pip install -e ".[dev]"`:

```bash
python -m pytest labs/the-agent-evaluation-playbook -m "not slow and not gpu and not network"
python labs/the-agent-evaluation-playbook/scripts/01_grade_the_world.py
python labs/the-agent-evaluation-playbook/scripts/02_partial_credit.py
python labs/the-agent-evaluation-playbook/scripts/03_repeated_trials.py
python labs/the-agent-evaluation-playbook/scripts/04_horizon.py
python labs/the-agent-evaluation-playbook/scripts/05_grader_noise.py
python labs/the-agent-evaluation-playbook/scripts/06_thresholds_and_oversight.py
python labs/the-agent-evaluation-playbook/scripts/07_injection_and_memory.py
python labs/the-agent-evaluation-playbook/scripts/08_oracle_ablation.py
```

All eight scripts together take about 8 s. The tests take about 1 s. Each script writes `results/<script name>.txt`.

## Which script covers which section of the post

| Post section | Script | What you see happen |
|---|---|---|
| Two objects of measurement; Grade the world, not the prose | `01_grade_the_world.py` | Seven behaviours, four graders. The transcript grader passes four failing agents. |
| The other things that changed | `01_grade_the_world.py` | A full `s0 -> sT` diff catches an unrequested write and extra emails that the goal check misses. |
| Where did it actually go wrong?; What a trace must contain | `01_grade_the_world.py` | A stale read surfaces as a wrong refund argument. Argument provenance in the trace points to the read. |
| Budgets are safety primitives | `01_grade_the_world.py` | A looping agent ends as `budget_exhausted`, a separate outcome. |
| Partial credit and its price | `02_partial_credit.py` | Deleting the failing tests lights all five checkpoints and fails the goal. |
| pass^k is a variance meter | `03_repeated_trials.py` | Same mean, different spread, very different pass^k curves. Flat for bimodal, steep for constant. |
| Accuracy is half of a number | `03_repeated_trials.py` | U = Q - λC picks a different retry policy as λ grows. |
| Horizon is per-step reliability wearing a costume | `04_horizon.py` | Halving h doubles H50. So does repairing half the errors. Clustered failures fatten the tail. |
| The grader is a noisy instrument | `05_grader_noise.py` | Floor α, ceiling 1-β. The runs needed grow as (1-α-β)^-2: 1.69x and 3.31x. |
| Reversibility sets a confidence threshold | `06_thresholds_and_oversight.py` | p* by tier. Acting above p* is the only policy that stays positive on irreversible actions. |
| Oversight has a capacity | `06_thresholds_and_oversight.py` | Harm is lowest at an interior escalation rate. Escalating everything is worse than escalating nothing. |
| Data is not instruction; The monitor has a bill | `07_injection_and_memory.py` | A provenance monitor drives attack success to 0, blocks a legitimate task, and costs S(1-b)^K. |
| Attacks that outlive the session | `07_injection_and_memory.py` | One poisoned note fires r/(1-s) times. Append-only memory grows without bound. A write filter stops it. |
| Counterfactual ablation locates the bottleneck | `08_oracle_ablation.py` | Oracle deltas depend on the order they are added. Shapley values give an order-free summary. |

## Results

All numbers are from AMD EPYC 7R13, 8 threads, Python 3.12.15, numpy 2.5.4. Every run is seeded, so a rerun gives
the same numbers. Timings are not the point of this lab, and none are reported beyond total run time.

Tests: 18 passed, offline (`results/tests.txt`). There are no `network`, `slow` or `gpu` tests.

| Measurement | Result | File |
|---|---|---|
| Agents passed by the transcript grader / by the goal-state grader / by goal state plus diff, out of 7 | 5 / 3 / 1 | `01_grade_the_world.txt` |
| Agents that pass the goal check but change something unrequested | 2 (item count zeroed; three extra emails) | `01_grade_the_world.txt` |
| Honest agent fed a stale read ($32 instead of $40): tool-call grader / goal-state grader | pass / fail; root cause traced to the stale read | `01_grade_the_world.txt` |
| Shortcut agent (deletes failing tests): checkpoints / goal state | 5 of 5 / fail | `02_partial_credit.txt` |
| Partial score of the shortcut agent at goal weight 1/6, 0.5, 0.8, 1.0 | 0.83, 0.50, 0.20, 0.00 | `02_partial_credit.txt` |
| pass^8 measured, 200 tasks × 16 episodes, suite mean ~0.6: constant / Beta wide / bimodal | 0.018 / 0.157 / 0.600 | `03_repeated_trials.txt` |
| Same, against p̄^8 | 0.017 / 0.009 / 0.017 | `03_repeated_trials.txt` |
| Best config by λ (per dollar): 0, 1 / 5 / 20 | retry up to 8 / retry up to 3 / single attempt | `03_repeated_trials.txt` |
| H50 simulated for h = 0.04, 0.02, 0.01, 0.005 | 17, 35, 69, 139 steps | `04_horizon.txt` |
| H50 at h = 0.02 with ρ = 0.5 | 69 steps | `04_horizon.txt` |
| Survival at 8 × H50 with clustered failures: simulated / constant-hazard fit | 0.009 / 0.004 | `04_horizon.txt` |
| Runs per agent to separate π = 0.5 from 0.6 (80% power): perfect grader / α=0.08, β=0.15 / α=0.2, β=0.25 | 392 / 664 / 1296 | `05_grader_noise.txt` |
| Same, as ratios, against (1-α-β)^-2 | 1.69, 3.31 against 1.69, 3.31 | `05_grader_noise.txt` |
| p* for an irreversible action with L/U = 100 | 0.990 | `06_thresholds_and_oversight.txt` |
| Utility per irreversible action: always act / act if p > 0.9 / act if p > p* | -4.00 / -2.15 / 0.08 | `06_thresholds_and_oversight.txt` |
| Harm per action, least-confident-first escalation: e = 0 / best (e = 0.2) / e = 1 | 0.100 / 0.082 / 0.105 | `06_thresholds_and_oversight.txt` |
| Attack success at q = 0.8: no monitor / provenance monitor | 0.799 / 0.000 | `07_injection_and_memory.txt` |
| Legitimate "contact vendor" task under the monitor | blocked in every episode (utility 0.000) | `07_injection_and_memory.txt` |
| Defended success, b = 0.15, K = 8: measured / S(1-b)^K | 0.246 / 0.245 | `07_injection_and_memory.txt` |
| Firings of one poisoned note, r = 0.3: s = 0.9 / s = 0.99 / s = 1 over 500 sessions | 3.00 / 29.64 / 149.95 | `07_injection_and_memory.txt` |
| Retrieval-oracle delta: added first / added after the verifier oracle | +0.046 / +0.000 | `08_oracle_ablation.txt` |

Notes on reading these:

- In `03`, the second-order formula for E[p^k] tracks the narrow Beta suite. It fails for the bimodal suite, where the
  variance is too large for a second-order expansion. The exact E[p^k] still matches the measurement.
- In `06`, the post's random-escalation model gives a shallow optimum (harm 0.094 against 0.099 at e = 0). Escalating
  least-confident actions first gives a deeper one. In both, escalating everything gives the highest harm.
- In `07`, the monitor cannot give back the utility of a hijacked episode. It only stops the exfiltration.
- In `08`, the failure rates are chosen for the demo, not measured. The order dependence and the Shapley summary are
  the result. The particular deltas are not.

## What was not measured

- No language model was run. The agents are scripted, so nothing here says how often any real model claims false
  success, follows an injection, or escalates well. Running a small open model through `Harness` would add that.
- No paper result was reproduced: not the τ-bench pass^k drop, not HAL's costs, not METR's horizon trend, not
  CaMeL's or Progent's attack rates, not the Online-Mind2Web gap, not the SWE-bench Verified audit.
- No real web, container or VM environment. The world is a Python dict. `reset(s0)` is a deep copy, so the gap
  between a sandbox and a live site does not appear.
- Adaptive attacks were not tried. The monitor was tested only against the one fixed injection in `world.py`.
- Contamination (the "Reasoning or remembering" section) has no script, because it needs a pretrained model and
  a dated task set.
- Model judges and their biases are not simulated beyond the α, β model in `05`.

## Theory

- [The Agent Evaluation Playbook](https://srini-anirudh.github.io/blogs/the-agent-evaluation-playbook/)
- Related code: [C10 agent harness](../../c10-agent-harness/), [C11 agent memory](../../c11-agent-memory/),
  [C12 agent RL](../../c12-agent-rl/)
