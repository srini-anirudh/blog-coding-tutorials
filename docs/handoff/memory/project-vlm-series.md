---
name: project-vlm-series
description: 20-part VLM blog series, all 20 published 2026-11-06; tracker docs/vlm-series.md; research paper posts partly held for author review
metadata:
  node_type: memory
  type: project
  originSessionId: ac2a1dcd-ee9b-4ab5-b22b-82dceadb6666
  modified: 2026-11-06T14:23:08.537Z
---

The VLM blog series is being expanded from 3 to 20 posts (plan, order, status and coverage map in `docs/vlm-series.md`).
Writers are subagents briefed via a scratchpad BRIEF; publishing is one post per commit+push, with series navs only linking
published posts and links to unpublished siblings pointed at the series list (`data-series-pending`).

**Why:** user asked to "keep documenting, committing and pushing" while many posts are written in parallel.
**How to apply:** update the status table in docs/vlm-series.md with each shipped post; never `git add -A blogs` while writers
are active (stage only the finished post's paths). Related: [[feedback-commits]], [[feedback-blog-writing]].

Status 2026-11-06: all 20 VLM posts published. Research series: thesis, M3Grounder, DoCoG published; REPLICA and DocLayout-VL drafts held (they state inconsistencies in the author's own papers; author to decide keep/soften/remove); Patram-Bench draft held (project-site data issues).
