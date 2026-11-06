---
name: project-code-series
description: Coding Tutorials series (C1–C12) — code in repo ~/Documents/Code/blog-coding-tutorials, posts on the blog with inline excerpts
metadata:
  type: project
---

Started 2026-11-06. The user asked for a code-first tutorial series ("a different type of blog, proper coding series"):
12 build-along posts (C1 PyTorch … C12 agent RL). The code lives in the separate repo
`/Users/anirudh-srini/Documents/Code/blog-coding-tutorials` (GitHub `srini-anirudh/blog-coding-tutorials`, chosen and
created by the user; push remote `git@github-personal:...`), one folder per tutorial. Posts quote exact excerpts using
`data-src` and `data-lines`. The plan is `docs/series-plan.md` in that repo. Shared page components are
`assets/css/code-tutorial.css`, `assets/js/code-tutorial.js` and `docs/code-tutorial-template.html` on the site. The
series id is `#code`, with `?series=code` and the nav label "Coding Tutorials series".
Coordination tools (brief, checker, publisher) are in the git-ignored `.series/` folder in that repo, because the
session scratchpad under /private/tmp gets wiped on restart.

**Why:** the user wants every layer from tensors to agents to be built and inspected, with code they can run.
**How to apply:**
- There is no `gh` CLI or token here, so the user must create GitHub repos themselves.
- The user floated a code companion for every existing theory post. Proposal: builds plus labs, with companion links;
  not yet approved.

Related: [[project-vlm-series]], [[feedback-blog-writing]], [[feedback-commits]].
