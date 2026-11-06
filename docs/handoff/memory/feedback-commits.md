---
name: feedback-commits
description: Commit and push frequently on this site repo, and never add Claude attribution/Co-Authored-By lines to commits
metadata:
  type: feedback
---

Keep committing AND pushing to origin main as work progresses (after each logical change) without waiting to be asked. Do not add any Claude attribution to commit messages (no "Co-Authored-By: Claude", no "Generated with Claude Code").

**Why:** User stated this explicitly; it overrides the default attribution reminder and the usual "only push when asked" rule.
**How to apply:** Plain commit messages in the repo's existing short imperative style, no trailer lines; `git push origin main` after each commit.
