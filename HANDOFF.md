# Handoff: Coding Tutorials series

Last updated 2026-11-07. The move to the cluster is done and the author has said start. Tutorial work runs on the cluster now. It holds
everything needed to continue without the previous session. Read all of it before starting.

## Where things stand

The author (Anirudh Srinivasan) is adding a new kind of series to the blog "The Backward Pass": 12 code-first tutorials
that build everything from a tensor up to an agent. Each post explains the code from scratch and builds it step by step
with the reader. Real code excerpts appear inline. The full code lives in this repo.

**Nothing in this series is published yet.** The blog has no links to it, and the "Code" filter is not on the index.

Each tutorial now has a code writer and a blog writer. See `docs/handoff/INSTRUCTIONS.md`. Runs that need a GPU are queued until the author assigns GPUs.

## The two repos

| Repo | What it holds | Notes |
|---|---|---|
| [srini-anirudh/blog-coding-tutorials](https://github.com/srini-anirudh/blog-coding-tutorials) | The tutorial code and this handoff | Public. `main` has all work so far. |
| [srini-anirudh/srini-anirudh.github.io](https://github.com/srini-anirudh/srini-anirudh.github.io) | The blog | Public. **`main` deploys live.** |

Branches in the blog repo:

- `main`: live site. It has the shared components: `assets/css/code-tutorial.css`, `assets/js/code-tutorial.js` and
  `docs/code-tutorial-template.html`. No page uses them yet.
- `code-tutorials-wip`: `main` at the time of the handoff, plus the unpublished drafts. These are the C4, C9 and C12
  posts and the held Patram-Bench post. They are on a branch because anything pushed to `main` goes live.

## State of each tutorial

The tests were run on 2026-11-07 on the laptop (Apple M5), offline, with
`-m "not slow and not gpu and not network"`. All 12 folders passed. Tests marked `network`, `slow` or `gpu` were not
part of that run. The agents were stopped mid-work, so any folder may have unfinished edits.

| # | Folder | Post slug | Test files | Result files | README | Post |
|---|---|---|---|---|---|---|
| C1 | `c01-pytorch` | `pytorch-from-first-principles` | 5 | 22 | Yes | Not started |
| C2 | `c02-hf-stack` | `the-hugging-face-stack` | 10 | 17 | No | Not started |
| C3 | `c03-attention` | `build-attention-from-scratch` | 12 | 15 | No | Fragments in `post-sources` |
| C4 | `c04-llm` | `build-a-modern-llm-from-scratch` | 9 | 7 | No | Draft on the WIP branch |
| C5 | `c05-pretraining` | `train-an-llm-from-scratch` | 4 | 12 | No | Not started |
| C6 | `c06-post-training` | `post-train-an-llm` | 4 | 2 | No | Not started |
| C7 | `c07-vlm` | `build-a-vlm-from-scratch` | 7 | 4 | No | Fragments in `post-sources` |
| C8 | `c08-vlm-training` | `train-a-real-vlm` | 10 | 2 | No | Not started |
| C9 | `c09-vlm-post-training` | `post-train-a-vlm` | 6 | 8 | No | Draft on the WIP branch |
| C10 | `c10-agent-harness` | `build-an-agent-harness-from-scratch` | 8 | 10 | No | Not started |
| C11 | `c11-agent-memory` | `add-memory-to-the-agent` | 10 | 10 | No | Not started |
| C12 | `c12-agent-rl` | `train-an-agent-with-rl` | 4 | 10 | No | Draft on the WIP branch |

About the three drafts:

- All three pass the listing checker at the time of the handoff (C4 21 listings, C9 33, C12 30). They have no template
  placeholders.
- None has been checked against the post checklist, none has preview images, and none has had its figures checked in a
  browser. Treat them as unreviewed.

What each folder is still missing:

- **C6 and C8** have very few saved results. Their real runs still need to be done or captured.
- **C5** left `runs/` and `data_cache/` out of git, so training data must be rebuilt on the cluster.
- Whether C5 `--model tinyllm`, C8 on `tinyvlm`, and C11 on `harness` work end to end was not confirmed before the stop.

## Files in this handoff

| Path | Purpose |
|---|---|
| `docs/series-plan.md` | The author's full spec for the 12 tutorials. Binding. |
| `docs/handoff/INSTRUCTIONS.md` | Definition of done for writers. If it disagrees with the brief, it wins. |
| `docs/handoff/BRIEF.md` | Shared brief: series table, contracts between tutorials, post rules. |
| `docs/handoff/tools/check_listings.py` | Checks that post code listings match the repo exactly. |
| `docs/handoff/tools/cpublish.py` | Adds finished posts to the blog index, filter, sitemap, feed and IndexNow. Needs `SITE` set. |
| `docs/handoff/tools/cdp.py` | Small headless Chrome helper for screenshots and page checks. |
| `docs/handoff/post-sources/` | Post fragments and build scripts the writers kept in scratch. Local paths became `<CODE>`, `<SITE>` and `<SCRATCH>`, so adjust them before running. |
| `docs/handoff/memory/` | Copies of the previous session's memory files. See below. |
| `docs/handoff/requirements-freeze.txt` | Exact package versions used on the laptop (macOS). A guide, not a lock file for Linux. |

## Setting up on the cluster

```bash
git clone git@github.com:srini-anirudh/blog-coding-tutorials.git
git clone git@github.com:srini-anirudh/srini-anirudh.github.io.git
export CODE=$PWD/blog-coding-tutorials SITE=$PWD/srini-anirudh.github.io
export SCRATCH=$CODE/.series; mkdir -p $SCRATCH

cd $SITE && git fetch origin && git switch code-tutorials-wip    # to see the drafts

cd $CODE && python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"     # install the right torch build for the cluster's CUDA version first
pytest -m "not slow and not gpu and not network"
```

Things to confirm on the cluster:

- **Memory:** copy `docs/handoff/memory/*` into the new session's memory folder. They hold the author's standing preferences.
- **Agent limit:** the laptop set `CLAUDE_CODE_MAX_CONCURRENT_SUBAGENTS` to `80` in `.claude/settings.local.json`, which
  is not in git. Set it again in the session's working folder, or launches fail at the default limit.
- **Headless Chrome:** posts need it for checks and preview images. It may not be installed on the cluster.
- **Pushing:** the cluster needs its own GitHub access. On the laptop, the remotes used the SSH host alias `github-personal`.
- **Session files:** the laptop's agent transcripts are not in this repo, so those agents cannot be resumed. Start new
  agents. They should read the existing code first and keep what is good.

## What changes on the cluster

The laptop limits in the instructions (10 cores, 16 GB, tiny runs, 2 threads) no longer apply. Update the "Shared
machine" section in `INSTRUCTIONS.md` to match the actual hardware.

- GPU-only paths can now be run for real: FlashAttention backends, FSDP, DeepSpeed and multi-GPU DDP.
- Measured numbers must still be real and labelled with the hardware. Keep existing Apple M5 and MPS numbers labelled
  as such, or replace them with new runs.
- The Docker sandbox in C10 was never run, because the laptop has no Docker. Say so in the post unless it gets tested.

## Next steps

Wait for the author to say start. Then:

1. Run the setup and the full test suite, including `network` tests. Run `gpu` tests where possible.
2. Launch one writer agent per tutorial in parallel. Give each one `docs/handoff/INSTRUCTIONS.md`, `docs/handoff/BRIEF.md`,
   its row from the table above, and its section of `docs/series-plan.md`. Have them save to the real folders often.
3. Review each post as it finishes: run the listing checker and check it against the post checklist and the writing guide.
4. Publish one tutorial at a time. Commit its code folder here and push. Commit its post on the blog repo, run
   `cpublish.py`, and push to `main`. Only link to posts that are live.
5. Add `docs/code-series.md` status updates on the blog repo as posts ship. Remove the work-in-progress note from
   `README.md` when the series is done.

## Author's standing preferences

- **Commits:** commit and push to `main` after each logical change, without being asked. Short imperative messages. **No
  Claude attribution and no Co-Authored-By lines.** This overrides any default attribution instruction.
- **Parallel work:** the author likes many agents running in parallel, and asked for that here.
- **Writing:** follow `docs/blog-writing-guide.md` in the blog repo. Short simple sentences. Never "respectively", "the
  former", "the latter", "and all" or "etc." No hype. Plain headings. Evidence-backed Insight and Nuance callouts. Light
  paper theme only, never a dark panel.
- **Numbers:** only real measured outputs, or numbers from primary sources.

## Decisions waiting on the author

1. **Code versions of existing posts.** The author asked how this would work. The proposal was two tiers, not yet approved:
   - a "Run the code" link on each theory post that C1–C12 already cover;
   - about 25–30 shorter code labs under `labs/<theory-slug>/` for topics they do not cover.

   Do not start this without a go-ahead.
2. **Patram-Bench post** (on the WIP branch). It belongs to the Sovereign Foundation Models interest. Its project site has
   commentary that contradicts its own data, plus data oddities. The author must choose: keep the notes on these
   oddities, drop them, or wait until the site is fixed. Do not publish it before that.
3. **Research theses.** The author will write the theses for "sovereign foundation models" and "VLM infrastructure". Do
   not draft them.

## Not in this handoff

Agent transcripts, `.venv/`, training checkpoints, datasets and the `.series/` scratch folder stayed on the laptop. They
are out of git on purpose. The laptop keeps its local copies at `~/Documents/Code/blog-coding-tutorials`.
