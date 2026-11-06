# Coding Tutorials: complete instructions for every writer

This file and BRIEF.md (same folder) are the full instructions. If they disagree, this file wins.

## Paths
- CODE=<path to your clone of blog-coding-tutorials> (GitHub: https://github.com/srini-anirudh/blog-coding-tutorials)
- SITE=<path to your clone of srini-anirudh.github.io> (the blog)
- SCRATCH=$CODE/.series (git-ignored; survives restarts)
- PY=$CODE/.venv/bin/python (torch, transformers, datasets, peft, trl, accelerate, pytest)
- Tool Python (Pillow, websocket-client): $SCRATCH/venv/bin/python. CDP helper: $SCRATCH/cdp.py
- Listing checker: `python3 $SCRATCH/code-series/check_listings.py $SITE/blogs/<slug>/index.html $CODE`
- Template and shared components, all finished: $SITE/docs/code-tutorial-template.html, $SITE/assets/css/code-tutorial.css,
  $SITE/assets/js/code-tutorial.js. Writing rules: $SITE/docs/blog-writing-guide.md
- Your tutorial's spec: its section of $CODE/docs/series-plan.md, plus that file's "Common Requirements" section.

## What the author asked for
A proper coding blog series, a different type of post from the theory posts. The code lives in the separate repo. The
post explains the code from scratch and builds it step by step together with the reader, with portions of the real code
inline and links into the repo.

## Boundaries
- You own only `$CODE/<your-folder>/`, `$SITE/blogs/<your-slug>/` and `$SCRATCH/code-series/<your-folder>/`.
- No git commands in either repo. The coordinator commits and publishes.
- Do not edit blogs/index.html, sitemap, feed, assets/, docs/, the repo root, other folders or other posts.
- Use the name `blog-coding-tutorials` everywhere. Never write `backward-pass-code`. No absolute local paths in committed files.

## Order of work (restarts happen, so always save to the real folders)
1. Check the current state of your folder and your post. Keep what is good. Do not start over.
2. Close out the code. Add no new features unless your spec section needs them. Tests pass with
   `cd $CODE && $PY -m pytest <folder> -m "not slow and not gpu"`, in under about 2 minutes.
3. Capture real outputs in `<folder>/results/` as small text or JSON. Delete trial runs you no longer need.
4. Write `<folder>/README.md`: what you build, how to run every script, test and benchmark, and a results table that states the hardware.
5. Write or finish `$SITE/blogs/<slug>/index.html` from the template. Save a complete first version early, then improve it.
6. Make the preview assets, verify, and report.
After each stage, overwrite `$SCRATCH/code-series/status/<folder>.progress` with one line saying what is done and what is next.

## Definition of done: the post
- Starts from `docs/code-tutorial-template.html`, with the noindex line removed and no placeholders or sample content left.
- Eyebrow `Coding tutorial · Build it from scratch`. One h1. Three tags. Published and modified 2026-11-06. JSON-LD parses and
  `isPartOf` lists both #blog and #code. Series navigator as in BRIEF.md. A prominent link to your repo folder near the top.
- "What you will build" box: deliverable, prerequisites, time, hardware, final repo tree.
- Every major step is a `section.build-step`: what we build and why, then the real code excerpt, then the command, then the
  real output, then a check or test. Start with the simplest version, run it, look at the tensors, and add the next piece.
- Explain the code from scratch, line by line where it matters. Assume the reader has the earlier tutorials, nothing more.
- Covers every item in your spec section. Items cut short for laptop reasons say so plainly.
- Code listings are exact excerpts with `data-src` and `data-lines`. The listing checker exits 0. Finalise the code before fixing line numbers.
- Shape tables wherever tensors appear. At least one "Break it" with the real error and how to debug it. A comparison against
  a trusted implementation. At least one comparison table along the aspects the post introduces.
- Theory is linked with `a.theory-link`, not re-derived. Every internal link resolves.
- At least 5 light figures, most of them animated multi-stage steppers and at least 2 interactive. They show what is inside the tensors.
- Insight and Nuance callouts, and a "Key insights and nuances" section.
- Ends with the final repo tree, how to run everything, 3 to 5 real exercises, and references where the post relies on papers.

## Writing rules
Simple language. Short sentences and short paragraphs. Professional tone. Plain sentence-case headings. Never use
"respectively", "the former", "the latter", "and all", "etc.", hype words, exclamation marks or emoji. Light paper theme only,
never a dark panel.

## Numbers
Quote only outputs you actually ran, labelled with the hardware (Apple M5 CPU or MPS). Never invent GPU numbers. GPU-only code
is marked as such, and the post gives the command to measure it. Report weak results honestly.

## Shared machine
10 cores and 16 GB, shared by 12 writers. Use `torch.set_num_threads(2)` and tiny models. Keep each run under about 10 minutes.
Do not run more than one training job at a time. Use your own ports only.

## Verify before reporting
Headless Chrome through CDP at 1440 and 390 widths: zero console errors, no horizontal overflow, every figure steps through all
states, and you have looked at screenshots of the figures (use new filenames each time, because image reads are cached).
Preview assets: `preview.png` (960×540) and `preview.gif` (960×540, 6–10 s loop, about 3 MB or less). Kill your server and Chrome when done.

## Final report
Folder and slug, final title, card summary (at most 160 characters), 3 tags, read time, preview alt text, folder tree, test command
and pass counts, what you ran and the key real numbers, section headings and figures, checker result, and anything GPU-only,
unverified or skipped. Then `touch $SCRATCH/code-series/status/<folder>.done`.
