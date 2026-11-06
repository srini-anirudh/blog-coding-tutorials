# Lab: VLMs That Act

Code companion for the post [VLMs That Act: Visual Tools, GUI Agents and Computer Use](https://srini-anirudh.github.io/blogs/vlms-that-act/).

The post asks what changes when the image is no longer a fixed input but the screen of a computer the model can
change. This lab builds a tiny app, MiniDesk, that an agent operates through pixels. It is a 96 x 96 expense form
with a hidden server state, pop-ups, dialogs and a checker that reads the saved records. On top of it we run the
post's central ideas at toy scale, on a CPU:

1. The computer-use loop, and why a screenshot is not the state.
2. Five ways to observe a screen, what each costs, and which ones survive a redesign.
3. The resolution arithmetic p = sqrt(W H / N), and how many icons share one visual token.
4. A small icon grounder trained from scratch. One pass at three token budgets against zoom search with the same
   weights.
5. One task at three action levels: pixel clicks, element ids and an API call.
6. A pixel policy trained by behavioural cloning, DAgger and RL from the task outcome. Compounding error, recovery,
   static against interactive evaluation, and the step budget.
7. The harness: coordinate formats, the cost of keeping screenshots in memory, and a gate against on-screen
   prompt injection.

Nothing here reproduces a paper's numbers. Where the post quotes results from OSWorld, ScreenSpot-Pro, DigiRL or the
pop-up study, the lab shows the mechanism small and says so.

The text-agent versions of some ideas already exist in this repo. [C12](../../c12-agent-rl/) covers behavioural
cloning, distribution shift and RL for a tool-calling agent. [C10](../../c10-agent-harness/) builds a full harness
with permissions and compaction. This lab adds what the image changes: pixels as input, pointing as the action.

## Layout

```
labs/vlms-that-act/
├── actlab/
│   ├── desk.py        # MiniDesk: hidden state, dynamics, renderer, checker, an expert that acts from any state
│   ├── observe.py     # screenshot, accessibility tree, DOM, Set-of-Mark, hybrid; token costs; locators
│   ├── resolution.py  # pixels per visual token, icons per token
│   ├── grounding.py   # synthetic icon screens, a pointer-over-tokens grounder, zoom search, click accuracy
│   ├── policy.py      # pixel policy, behavioural cloning, DAgger, REINFORCE, static and live evaluation
│   ├── harness.py     # resize and coordinate mapping, validation, reversibility gate, context cost
│   └── report.py      # results files with a hardware line and the command
├── scripts/           # 01 to 07, one per mechanism, plus run_all.sh
├── tests/             # pytest
└── results/           # real outputs of every script
```

## Setup

From the repository root, after `pip install -e ".[dev]"`. Nothing downloads. Everything runs on a CPU. Set
`OMP_NUM_THREADS` on a shared machine; the numbers below used 8 threads.

## Run everything

```bash
# Tests (about a minute).
python -m pytest labs/vlms-that-act -m "not slow and not gpu and not network"
python -m pytest labs/vlms-that-act -m slow            # one short RL smoke test

# Every script, in order (about 30 min on 8 CPU threads). Or run them one at a time.
# Scripts 01, 02, 03, 05 and 07 take seconds.
bash labs/vlms-that-act/scripts/run_all.sh
python labs/vlms-that-act/scripts/01_loop_and_state.py
python labs/vlms-that-act/scripts/02_observations.py
python labs/vlms-that-act/scripts/03_resolution.py
python labs/vlms-that-act/scripts/04_grounding_zoom.py   # trains the grounder (about 17 min); checkpoint in runs/
python labs/vlms-that-act/scripts/05_action_levels.py
python labs/vlms-that-act/scripts/06_bc_dagger_rl.py     # trains five policies (about 10 min)
python labs/vlms-that-act/scripts/07_harness.py
```

## Which script covers which section of the post

| Post section | Script |
|---|---|
| From answering to acting (Figure 1, the screenshot-is-not-the-state nuance) | `01_loop_and_state.py` |
| What a screen looks like to a model (Figure 4, Table 1) | `02_observations.py` |
| Resolution decides what can be pointed at (Figure 6) | `03_resolution.py` |
| Grounding becomes action; ScreenSpot-Pro and zoom search | `04_grounding_zoom.py` |
| Choosing the action space (Figure 7, Table 2) | `05_action_levels.py` |
| Behavioural cloning and compounding error (Figure 9) | `06_bc_dagger_rl.py`, parts 1 to 4 |
| Reinforcement learning from task outcomes | `06_bc_dagger_rl.py`, parts 2 to 4 |
| Evaluating in interactive environments (Figure 10, the step-budget nuance) | `06_bc_dagger_rl.py`, parts 3 and 5 |
| Grounding becomes action (coordinate formats) | `07_harness.py`, part 1 |
| What a computer-use agent must remember | `07_harness.py`, part 2 |
| The harness around the model; the screen is an attack surface (Figure 11) | `07_harness.py`, part 3 |

## Results

Every number below was measured on an AMD EPYC 7R13 with 8 threads, torch 2.15.0, CPU only. The machine was shared
with other jobs, so treat times as rough. Raw output is in `results/`, under the name of the script that wrote it.
Each file starts with its hardware line and command.

Tests on the same CPU: 77 offline tests pass in about 35 s. One `slow` test (an RL smoke run) passes. There are no
`network` or `gpu` tests: nothing downloads and nothing needs CUDA.

### 1. The loop, and a screenshot is not the state (`01_loop_and_state.txt`)

| Measurement | Result |
|---|---|
| Expert episode: decisions, of which need a coordinate | 5, of which 4 |
| Submit once against submit twice: records on the server | 1 against 2 |
| Final screenshots of those two episodes identical, pixel for pixel | yes |
| Checker that reads records: success once / twice | True / False |

### 2. Observations and a redesign (`02_observations.txt`)

Text tokens use 4 characters per token, not a real tokenizer. The screenshot cost is for a 1920 x 1080 screen with
14-pixel patches and a 1024-token budget.

| View | Approx. tokens | Finds Submit after the redesign? |
|---|---|---|
| Screenshot, fixed coordinates | 966 | no |
| Screenshot, visual cue (primary-button colour) | 966 | yes |
| Accessibility tree, role + name "Submit" | 46 | no (renamed) |
| DOM, id "btn-submit" | 233 | no (id changed) |
| Set-of-Mark, the old mark number | 983 | no (marks renumbered) |
| Hybrid (screenshot + compact list) | 996 | not tested |

The visual cue is a hand-written stand-in for a learned grounder. It survives only because the button keeps its
colour. This is the post's qualitative Table 1 made concrete for one page. It is not a benchmark.

### 3. Pixels per token (`03_resolution.txt`)

The arithmetic is exact. 3840 x 2160 with 1024 tokens gives p = 90.0 pixels. A strip of 16 adjacent 24-pixel icons
then puts 3 to 4 icons in each token column, and 1 of 16 icons owns a column alone. At 1920 x 1080 with 4096 tokens,
p = 22.5 and all 16 icons own a column.

### 4. Grounding and zoom search (`04_grounding_zoom.txt`)

A 384 x 384 screen with toolbars of 12-pixel icons. One target glyph among 8 kinds. A click counts if it lands inside
the icon box. One grounder (3000 steps, 1005 s to train and evaluate), the same weights in every row, 300 held-out
screens.

| Method | Visual tokens | Screen px per token | Click accuracy |
|---|---|---|---|
| Single pass, 48 px input | 36 | 64 | 0.013 |
| Single pass, 96 px input | 144 | 32 | 0.103 |
| Single pass, 192 px input | 576 | 16 | 0.517 |
| Zoom search, 9 tiles of 192 px, 48 px each | 324 | 32 | 0.077 |
| Zoom search, 25 tiles of 96 px, 48 px each | 900 | 16 | 0.577 |

Accuracy follows pixels per token, not the weights. At the 48-pixel budget the grounder almost never hits (1.3%).
Searching 25 zoomed tiles at the same per-pass budget lifts it to 57.7%, with no extra training. The search spends
more tokens in total (900) than the single 192-pixel pass (576) and lands about level with it: with 300 screens each
row has a standard error near 3 points. ScreenSeekeR's planner-guided
search, and the 18.9% to 48.1% it reports on ScreenSpot-Pro, were not reproduced. Our search is exhaustive tiling
ranked by the grounder's own "target is here" score. The grounder is small and briefly trained, so even the best row
misses 4 clicks in 10.

### 5. Action levels (`05_action_levels.txt`)

2000 single-report tasks per row. The pixel agent makes the expert's decisions but clicks with Gaussian error sigma
around each target centre on the 96-pixel screen.

| Agent | Decisions | Need coordinates | Hit rate per click | hit^4 | Task success |
|---|---|---|---|---|---|
| Pixel, sigma 2 px, 5-step budget | 5 | 4 | 0.991 | 0.964 | 0.964 |
| Pixel, sigma 3 px, 5-step budget | 5 | 4 | 0.903 | 0.663 | 0.673 |
| Pixel, sigma 3 px, 10-step budget | 5.62 | 4.62 | 0.901 | - | 0.994 |
| Element ids | 4 | 0 | - | - | 1.000 |
| Element ids, menu options missing from the tree | 3 | 0 | - | - | 0.000 |
| API call | 1 | 0 | - | - | 1.000 |

With no spare steps, success equals the per-click hit rate to the fourth power, as the post's 0.98^4 = 0.92 says.
Spare steps let the agent retry, which is why the same 3-pixel agent reaches 0.994 with ten steps. The element agent
fails completely when the tree lacks the options, and the harness says why: it rejects an id that is not on screen.

### 6. Cloning, compounding error, DAgger and RL (`06_bc_dagger_rl.txt`, 624 s in total)

Part 1 checks the no-recovery model in the app itself. A noisy expert takes a uniformly random action with
probability eps at each step, over a 30-step task (six reports), 1000 episodes per cell.

| eps | (1 - eps)^30 | Measured, no spare steps | Measured, 2x step budget |
|---|---|---|---|
| 0.02 | 0.545 | 0.551 | 1.000 |
| 0.05 | 0.215 | 0.215 | 1.000 |

The formula holds exactly when a slip cannot be undone. The same expert finishes every task when it has spare steps,
because it knows how to recover from any state.

Parts 2 to 5 train pixel policies. The 200 demonstrations are clean: the expert never errs and no pop-up appears,
like recorded human data. At test time a pop-up appears during each report with probability 0.5, and it blocks
every click until it is closed. 200 live episodes per cell, greedy actions.

| Policy | Static step match | Live, no pop-ups | Live, pop-ups |
|---|---|---|---|
| BC, 200 demonstrations | 0.981 | 0.910 | 0.480 |
| BC, 800 demonstrations | 0.998 | 0.995 | 0.525 |
| DAgger (3 rounds of expert relabelling) | 1.000 | 1.000 | 0.990 |
| BC + RL (60 x 64 episodes, KL 1.0 to BC) | 0.971 | 0.915 | 0.475 |
| BC + RL, no KL | 0.874 | 0.555 | 0.295 |

Success against task length, pop-ups on, 2x budget:

| Policy | 5 steps | 10 | 20 | 30 |
|---|---|---|---|---|
| BC, 200 demonstrations | 0.465 | 0.225 | 0.045 | 0.000 |
| BC, 800 demonstrations | 0.480 | 0.265 | 0.045 | 0.005 |
| DAgger | 0.995 | 0.995 | 0.990 | 0.970 |
| BC + RL | 0.475 | 0.215 | 0.045 | 0.005 |

Success against step budget, 2 reports (10 expert steps), pop-ups on:

| Policy | x1.0 | x1.2 | x1.5 | x2.0 | x3.0 |
|---|---|---|---|---|---|
| BC, 200 demonstrations | 0.190 | 0.190 | 0.190 | 0.190 | 0.190 |
| BC, 800 demonstrations | 0.230 | 0.230 | 0.230 | 0.230 | 0.230 |
| DAgger | 0.230 | 0.995 | 0.995 | 0.995 | 0.995 |

What these show:

- Static scores hide the failure. Cloned policies match the expert's next action 98% to 99.8% of the time, yet
  they finish only about half the live tasks with pop-ups. The pop-up is a state no demonstration contains. The
  cloned policy keeps clicking the form behind it until the budget runs out. Four times more clean demonstrations
  barely help (0.480 to 0.525), because more copies of the same corridor add no recovery data.
- DAgger fixes it with expert labels on the states the learner itself visits. Success stays near 1 as the task grows
  to 30 steps, while cloning falls to 0.
- The step budget changes the ranking. With no spare steps, DAgger ties the 800-demonstration clone (0.230), since
  closing a pop-up costs a step. With 20% spare steps it reaches 0.995.
- RL did not learn the recovery at this scale. This is a negative result, and it is not a reproduction of DigiRL.
  The mean sampled success over each 10 RL iterations stayed between 0.19 and 0.30 (`rl_sampled_success` in
  the JSON), so the 0/1 reward
  gave little signal about pop-ups. Without a KL penalty, REINFORCE made the policy worse, even on clean screens.
  Our best explanation, not tested: failed episodes are full of blocked clicks on visible form fields, and pushing
  those clicks down also pushes down the same clicks where they work. With a KL penalty to the BC policy, the policy
  stays at BC's level. A learned value function (UI-TARS-2 reports PPO beating GRPO on long GUI episodes) or
  DigiRL's longer offline-to-online training may be what is missing. We did not test either.

### 7. The harness (`07_harness.txt`)

| Measurement | Result |
|---|---|
| 2560 x 1440 screen fitted to 1024 tokens of 14 px | 597 x 336, scale 0.2333 |
| One target written as `norm01`, `bin999`, `resized_px`, `screen_px` | all four map to the same click (1830, 1210) |
| `resized_px` without the scale factor | clicks at (427, 282), 4.29x too close to the origin |
| Prompt tokens at step 30, every screenshot kept | 31,060 |
| Prompt tokens at step 30, last 3 screenshots, older steps as text | 4,222 |
| Injection banner in 30% of 1000 episodes, no gate: exfiltrations | 296 |
| Same, with the reversibility gate: exfiltrations / reports submitted | 0 / 1000 |

The gate asked for 1279 confirmations: one per real submit, which the stand-in human approves, plus one per injected
export, which it denies. The agent in part 3 is a scripted, naive one that obeys any on-screen line addressed to
it. It shows where the defence sits. It does not measure how often a real model obeys, and the pop-up study's 86% is
not reproduced.

## What was not measured

- No real VLM is run. Every policy and grounder here is a small convolutional network trained from scratch on
  synthetic screens. No result says anything about how a pretrained VLM behaves.
- No paper's number is reproduced: not OSWorld's observation comparison, ScreenSpot-Pro or ScreenSeekeR, DigiRL's
  17.7% to 67.2%, WebRL, UI-TARS's step-budget gap, or the pop-up attack rate.
- Token counts for text views use 4 characters per token, not a tokenizer.
- The observation comparison is one page and one redesign. The visual locator is hand-written, not learned.
- Zoom search is exhaustive tiling, not ScreenSeekeR's planner-guided search.
- The RL here is plain REINFORCE with a batch-mean baseline, element-level exploration and an optional KL penalty.
  No value function, PPO, GRPO clipping, replay or curriculum was tried.
- No GPU was used. The post is not about speed, and every experiment fits on a CPU.
