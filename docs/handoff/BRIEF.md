# Coding Tutorials series — shared brief (binding)

SCRATCH=$CODE/.series   (git-ignored working folder; create it)
SITE=<path to your clone of srini-anirudh.github.io>   (the blog, GitHub Pages)
CODE=<path to your clone of blog-coding-tutorials>   (GitHub: https://github.com/srini-anirudh/blog-coding-tutorials)
PY=$CODE/.venv/bin/python   (Python 3.12, torch 2.15 with MPS, transformers 5.x, datasets 5.x, peft, trl, accelerate, pytest — already installed; shared by all writers)

The author (Anirudh Srinivasan) is launching a NEW KIND of series on the blog "The Backward Pass": proper coding
tutorials. The theory posts already exist. These posts are code-first: we build the thing together, step by step, in
the blog, with the real code shown inline, and the full codebase lives in the separate repo `blog-coding-tutorials`.
The full plan is `$CODE/docs/series-plan.md` (read "Series Philosophy", your tutorial's section, and
"Common Requirements Across Every Tutorial" — all binding). Tagline: "Build the thing. Inspect every tensor. Train it.
Break it. Profile it. Scale it."

## Series
| # | Title | Code folder | Python package | Blog slug |
|---|---|---|---|---|
| C1 | PyTorch From First Principles | c01-pytorch | scratchnet | pytorch-from-first-principles |
| C2 | The Hugging Face Stack | c02-hf-stack | hfstack | the-hugging-face-stack |
| C3 | Build Attention From Scratch | c03-attention | attnzoo | build-attention-from-scratch |
| C4 | Build a Modern LLM From Scratch | c04-llm | tinyllm | build-a-modern-llm-from-scratch |
| C5 | Train an LLM From Scratch | c05-pretraining | pretrain | train-an-llm-from-scratch |
| C6 | Post-Train an LLM | c06-post-training | posttrain | post-train-an-llm |
| C7 | Build a Vision-Language Model From Scratch | c07-vlm | tinyvlm | build-a-vlm-from-scratch |
| C8 | Train a Real VLM | c08-vlm-training | vlmtrain | train-a-real-vlm |
| C9 | Post-Train a VLM | c09-vlm-post-training | vlmalign | post-train-a-vlm |
| C10 | Build an Agent Harness From Scratch | c10-agent-harness | harness | build-an-agent-harness-from-scratch |
| C11 | Add Memory and Context Engineering to the Agent | c11-agent-memory | agentmem | add-memory-to-the-agent |
| C12 | Train an Agent With Reinforcement Learning | c12-agent-rl | agentrl | train-an-agent-with-rl |

## Files you own (touch nothing else)
- `$CODE/<your-folder>/` (everything inside it).
- `$SITE/blogs/<your-slug>/index.html`, `preview.png`, `preview.gif`.
- Scratch only under `$SCRATCH/code-series/<your-folder>/`.
Do NOT edit other folders, the repo root (README, pyproject, CI), the site's blogs/index.html, assets/, docs/,
sitemap/feed, or other posts. Do NOT git commit or push (the coordinator does). Do NOT install packages into the
shared venv unless essential; if you must, use
`$SCRATCH/uvdl/uv-aarch64-apple-darwin/uv pip install --python $PY <pkg>` and list it in your folder's README.

## Part A — the code (do this first; the blog shows this code)
Layout of your folder (adapt names, keep the spirit):
```
<folder>/
  README.md            what you build, how to run every script, test, benchmark; results table with hardware stated
  <package>/           the library, small readable modules (reference implementations first, fast ones separate,
                       e.g. attention_naive.py vs attention_sdpa.py)
  scripts/             runnable entry points: demos, train, eval (python -m <package>... or scripts/x.py)
  benchmarks/          benchmark scripts that print tables
  tests/               pytest: shapes, numerical equivalence vs a trusted implementation, gradients,
                       causality, serialization/resume, cache equivalence — whatever applies
  configs/             small YAML/JSON configs where useful
  conftest.py          inserts this folder on sys.path so `import <package>` works from repo root
```
- Python 3.10+ style, type hints, docstrings that state tensor shapes. Comments explain WHY, not what.
  Code must be clean enough to be quoted line for line in a tutorial.
- Must run on a laptop (this machine: Apple M5, 10 cores, 16 GB, MPS, no CUDA, no Docker). 13 writers share it:
  `torch.set_num_threads(2)` in your runs, keep models tiny, each demo/training run under ~10 minutes, tests for
  your folder under ~2 minutes with `-m "not slow and not gpu"`. Pick device via a helper (cuda > mps > cpu).
- CUDA-only paths (FlashAttention kernels, FSDP, DeepSpeed, multi-GPU) are written properly and marked
  `@pytest.mark.gpu` / documented as GPU-only. DDP can be demonstrated on CPU with gloo and `torchrun --nproc_per_node 2`.
- Tests that download from the Hub get `@pytest.mark.network`; prefer tiny models
  (`hf-internal-testing/tiny-random-*`, `HuggingFaceTB/SmolLM2-135M(-Instruct)`, `HuggingFaceTB/SmolVLM-256M-Instruct`,
  `google/siglip-base-patch16-224`, `Qwen/Qwen2.5-0.5B-Instruct`). Prefer small or synthetic datasets
  (synthetic data is encouraged when it makes rewards and labels exact, e.g. rendered shapes/text/charts for VLMs).
- RUN everything you show: tests, demos, benchmarks, short trainings. Quote only real outputs and real numbers
  from this machine, labelled with the hardware ("Apple M5 CPU" / "MPS"). Never invent GPU numbers; where GPU
  behaviour matters, explain it and give the command to measure it.
- Run tests from the repo root: `cd $CODE && $PY -m pytest <folder> -m "not slow and not gpu"`.
- Cross-tutorial reuse. Folders are written in parallel, so each folder must work on its own. Stable contracts:
  - C4 `tinyllm` exposes `ModelConfig` (dataclass: vocab_size, hidden_size, num_layers, num_heads, num_kv_heads,
    intermediate_size, max_seq_len, plus extras) and `TinyLLM(config)`; `forward(input_ids, labels=None,
    past_key_values=None)` returns an object with `.logits`, `.loss`, `.past_key_values`; `generate(...)`.
    C4's attention follows the C3 GQA+RoPE design (say so, link C3); it may be a trimmed in-package copy.
  - C5 trains the C4 architecture. Default to its own model path that works standalone (e.g. HF `LlamaForCausalLM`
    from a small `LlamaConfig`, which is the same architecture), plus `--model tinyllm` that imports `../c04-llm`
    when present.
  - C6 and C9 post-train real pretrained small models (SmolLM2-135M / SmolVLM-256M or similar), because tiny
    from-scratch models are not capable enough for preferences or RL; say this plainly.
  - C7 `tinyvlm` exposes `VLMConfig`, `TinyVLM` (submodules `.vision`, `.projector`, `.lm`;
    `forward(input_ids, pixel_values, attention_mask=None, labels=None)` → `.loss`, `.logits`; `generate`) and
    `TinyVLMProcessor`. C8 trains it: write data/mixture/eval/stage code first; when
    `$SCRATCH/code-series/status/c07.api-ready` exists, integrate with `../c07-vlm`; if it never appears, ship a
    minimal assembly under your own package that follows the same contract.
  - C10 `harness` exposes `Agent`, `ToolRegistry`/`@tool`, `EventLog`, model clients with a common interface
    including a deterministic `ScriptedModel` for tests, a local HF model client, and an OpenAI-compatible client
    (env-configured). C11 extends it the same way C8 waits for C7 (marker `c10.api-ready`). C12 builds its own
    environment and may reuse the C10 tool shapes.
  - When your public API is stable, `touch $SCRATCH/code-series/status/<cNN>.api-ready` (e.g. c07.api-ready).

## Part B — the blog post (a proper coding tutorial, a different type of post from the theory posts)
- Template: wait until `$SCRATCH/code-series/status/template.ready` exists (another agent is building it while you
  write code). Then start from `$SITE/docs/code-tutorial-template.html` and follow its instructions; it adds the
  shared `assets/css/code-tutorial.css` and `assets/js/code-tutorial.js` (code listings with file tabs, line
  numbers, highlighted lines, diff lines, copy button, GitHub link generated from `data-src`/`data-lines`, light
  terminal/output blocks, shape tables, build steps, theory links, "Break it" and "Test" blocks, build-progress rail).
  All of it is light-themed. Do not invent your own variants of these components.
- Also follow `$SITE/docs/blog-writing-guide.md` (binding): simple language, short sentences and paragraphs,
  no "respectively", "the former/latter", "and all", "etc.", no hype, no exclamation marks, no emoji, plain
  sentence-case headings, Insight/Nuance callouts plus a "Key insights and nuances" section, at least one comparison
  table along the aspects the post introduces (e.g. MHA vs MQA vs GQA vs MLA: cache size, quality, kernels).
- Build it together, step by step. Start from the simplest version, run it, look at the tensors, then add the next
  piece. Every major step: what we build and why → the code (real excerpt) → run it → the real output → a check or
  test. Explain the code from scratch, line by line where it matters. Show tensor shape tables. Start naive, verify,
  find the bottleneck, then optimise. Include intentional failures ("Break it") and how to debug them. Compare
  against a trusted implementation. End with the final repo tree, how to run everything, and 3–5 real exercises.
- Do not repeat theory derivations: link the theory post with the template's theory-link component. Theory posts
  (relative `../slug/`): what-autograd-actually-does, what-an-optimizer-actually-does, how-attention-moves-information,
  how-models-know-where-they-are, the-geometry-of-normalization, why-neural-networks-need-nonlinearity,
  why-transformers-do-not-explode, what-changed-inside-the-transformer, choosing-the-next-token,
  the-mechanics-of-llm-inference, the-serving-playbook, inside-a-training-step, the-parallelism-playbook,
  why-fast-gpus-still-wait-for-memory, why-gpus-are-built-for-deep-learning, how-to-find-what-is-slowing-your-model,
  how-llms-are-pretrained, scaling-laws-from-first-principles, mixing-data-without-losing-capabilities,
  teaching-a-language-model-to-follow-instructions, how-lora-changes-a-model-without-rewriting-it,
  teaching-a-model-what-we-prefer, reinforcement-learning-from-first-principles,
  how-reinforcement-learning-teaches-models-to-reason, one-transformer-two-modalities, inside-a-modern-vision-encoder,
  the-vlm-architecture-gallery, teaching-language-models-to-see, one-model-many-visual-skills,
  aligning-vision-language-models, rewarding-what-the-model-sees, when-language-must-point, how-vlms-read-documents,
  measuring-what-a-vlm-can-do, the-cost-of-a-picture, inside-an-ai-agent-harness, memory-is-more-than-context,
  the-making-of-an-ai-agent, how-language-models-learn-to-use-tools, reinforcement-learning-for-agents-from-first-principles,
  the-agent-evaluation-playbook. Verify each slug exists under `$SITE/blogs/` before linking.
  Link sibling coding tutorials by slug (they will exist when the series ships).
- Code shown inline must be EXACT excerpts of files in your folder, with `data-src` (repo-relative path, e.g.
  `c03-attention/attnzoo/gqa.py`) and `data-lines` (e.g. `12-40`) set so the GitHub link and the coordinator's
  drift checker work. Finalise code before you fix line numbers. Run the template's checker on your post.
- Figures: light paper theme only (never dark panels). At least 5 figures, most animated multi-stage (shared stepper
  following the template's animation contract), at least 2 interactive (e.g. stride/view visualiser, mask builder,
  KV-cache growth with sliders, loss-mask painter, agent-loop trace stepper). Tensor-level visuals are the heart of
  this series: show what is inside the tensors.
- Page contract: as in the template (head, canonical, OG, JSON-LD BlogPosting with
  `"isPartOf": [{ "@id": "https://srini-anirudh.github.io/blogs/#blog" }, { "@id": "https://srini-anirudh.github.io/blogs/#code" }]`,
  published/modified 2026-11-06, eyebrow given by the template, one h1, 3 tags, TOC, citation block, footer).
  Series navigator directly after `</header>`:
  ```html
  <nav class="ml-series-nav" aria-label="Coding Tutorials series">
    <a class="ml-series-nav__identity" href="../index.html?series=code">Coding Tutorials</a>
    <span class="ml-series-nav__position">NN of 12 &middot; PART</span>
    <div class="ml-series-nav__links">…previous/next with the table titles…</div>
  </nav>
  ```
  PART groups: C1–C3 "Raw mechanics", C4–C6 "Language models", C7–C9 "Vision-language models", C10–C12 "Agents".
  Link the repo folder prominently near the top: https://github.com/srini-anirudh/blog-coding-tutorials/tree/main/<folder>
- Preview assets: `preview.png` (960×540) and `preview.gif` (960×540, 6–10 s loop, ≤ ~3 MB) of your best figure.
  Headless Chrome via CDP: helper `$SCRATCH/cdp.py` (class CDP(port); .go(url), .ev(js), .shot(path, clip), .sleep(s),
  .logs), Pillow via `$SCRATCH/venv/bin/python`. Use your own ports (in your task):
  `python3 -m http.server <HTTP_PORT>` from $SITE in the background, and
  `"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --remote-debugging-port=<CDP_PORT> --user-data-dir=$SCRATCH/code-series/<folder>/chrome --window-size=1440,1000 --hide-scrollbars about:blank &`.
  Call Network.setCacheDisabled, navigate to about:blank before reloading, and use new screenshot filenames each
  time (the Read tool caches images). Kill both processes when done.
- Verify: zero console errors at 1440 and 390 widths, no horizontal overflow, every figure steps through all
  states, screenshots inspected (Read the PNGs), JSON-LD parses, internal links resolve, listing checker passes.

## Report back (final message, concise)
folder + slug, final title, card summary (≤160 chars), 3 tags, read time, preview alt text, the folder tree,
test command and result (pass counts), what was run and the key real numbers, the post's section headings and
figures, anything GPU-only/unverified/skipped, and any contract you exposed or depended on.

## ADDENDUM (binding, from the coordinator): repo name, stable paths, and order of work
- The GitHub repo is `srini-anirudh/blog-coding-tutorials`. Use https://github.com/srini-anirudh/blog-coding-tutorials
  in every link, README and post. Never write the old name `backward-pass-code` in anything public.
- Paths moved. CODE=$CODE and
  SCRATCH=$SCRATCH (git-ignored, survives restarts). Tools are
  at $SCRATCH/code-series/check_listings.py, $SCRATCH/cdp.py, $SCRATCH/venv/bin/python and
  $SCRATCH/uvdl/uv-aarch64-apple-darwin/uv.
- Sessions have stopped twice and no post exists yet. The post is the deliverable. Stop extending the code now. Close
  out only what the post needs: tests passing, README written, real outputs captured. Then write
  `$SITE/blogs/<slug>/index.html` straight away, saving a first complete version early and improving it in passes.
- Save real run outputs inside your folder under `results/` (small text or JSON only), so they survive a restart.
  Do not leave absolute local paths in any committed file.
- Large or generated files stay out of git (`.gitignore` covers runs/, outputs/, checkpoints/, data_cache/, *.bin,
  *.pt, *.safetensors, *.ckpt, *.pth). Keep committed files small (under ~1 MB each).
- Do not run git commands in the code repo or the site. The coordinator commits each finished tutorial.
