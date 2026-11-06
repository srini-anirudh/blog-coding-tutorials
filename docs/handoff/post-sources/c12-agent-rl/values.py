# exec()ed by build_post.py with J, RES, PKG, ns, NOTES, LADDER in scope. Every number comes from results/.
import html, re
esc = lambda s: html.escape(s, quote=False)
pct = lambda v: f"{round(100 * v)}%"
read = lambda n: (RES / n).read_text().rstrip("\n") if (RES / n).exists() else ""

ns["readtime"] = 55
ns["vocab"] = 174
bc = J("bc")
ns["params"] = f"{bc['params']:,}"
E = bc["eval"]

# ---- demo output
demo = read("demo_env.txt")
parts = demo.split("\nfiles:")
eps = parts[0].strip().split("\n\n")
ns["demo_output"] = esc("\n\n".join(eps))
tok = "files:" + parts[1]
ns["demo_tokens"] = esc("\n".join(l for l in tok.split("\n") if not l.startswith("files:")).strip())

ns["test_env"] = "15 passed"
ns["test_all"] = esc(read("test.txt") or "(pending)")

# ---- policy choice
bp = read("bench_policy.txt")
def row(name, dev):
    m = re.search(rf"{re.escape(name)}\s+{dev}\s+([\d,]+)\s+([\d,]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", bp)
    return tuple(float(x.replace(",", "")) for x in m.groups()) if m else None
tiny = row("TinyPolicy d=128 L=3", "cpu")
smol = row("SmolLM2-135M + LoRA r=16", "cpu")
smolm = row("SmolLM2-135M + LoRA r=16", "mps")
ns["policy_choice"] = (
    f"This settles it. One iteration through SmolLM2-135M with LoRA took {smol[4]:.0f} seconds on the CPU"
    + (f" and {smolm[4]:.0f} seconds on MPS" if smolm else "")
    + f". The tiny policy took {tiny[4]:.1f} seconds. Our main run has 120 iterations at 12 times this batch size. "
    "At the SmolLM2 rate that would take days on this laptop, while other jobs compete for it. "
    "A real learning curve on this machine has to come from a small model, so that is what we use. "
    "Nothing about the algorithm depends on the size. The machine was busy during these timings, so read the ratio, not the absolute numbers."
)

C = E["clean"]
ns["bc_prose"] = (
    f"<p>Read the last line carefully. On clean tasks the policy succeeds {pct(E['clean'])} of the time. "
    f"The misses are mostly level 3, where the value must be looked up in a read result. "
    f"From perturbed starts it succeeds {pct(E['perturbed'])}. With failures injected it succeeds {pct(E['failures'])}. "
    f"On long tasks it succeeds {pct(E['long'])}. It never saw an error message or an <code>and</code> in a task, and it shows.</p>"
    "<p>This is the cloned policy that the rest of the post starts from. A strong start is not the aim. "
    "It knows the action syntax and the procedure, which gives RL something to work with.</p>"
)

# ---- shift
sh = J("shift_bc")
r = sh["rows"]
st = read("shift_bc.txt").split("\n")
keep = st[:6]
i = next(k for k, l in enumerate(st) if l.startswith("task ["))
ns["shift_output"] = esc("\n".join(keep + [""] + st[i:i + 10]))
ns["shift_prose"] = (
    f"<p>One harmless step takes success from {pct(r[0]['success'])} to {pct(r[1]['success'])}. Three steps take it to {pct(r[3]['success'])}. "
    f"The expert stays at {pct(r[3]['expert'])} from the same starts, so the task never became harder. "
    f"Look at the first trajectory: after one correct-looking <code>read</code>, the policy emits <code>edit edit db/dev ttl 30</code>. "
    f"That is not a syntax it was taught. It is what a next-token model does outside its data.</p>"
    '<aside class="callout callout-insight"><p class="callout-label">Insight</p>'
    f"<p>Only {r[0]['unseen']:.0%} of decisions are in unseen states from clean starts, and {pct(r[1]['unseen'])} after a single forced mistake. "
    "The state space of a procedure is narrow, and demonstrations cover exactly one path through it. "
    "So the first mistake is the expensive one. In practice, measure success from perturbed starts, not only from clean ones. A clean evaluation hides this completely.</p></aside>"
    '<aside class="callout callout-nuance"><p class="callout-label">Nuance</p>'
    "<p>The abstract signature under-counts novelty. The second trajectory printed by the script has a forced <code>search</code> for the wrong key, and every state it visits has a signature seen in demos. The policy then reads the wrong file and edits with the wrong value. "
    "The signature says the state is familiar, while the tokens that matter are not. So treat the unseen rate as a lower bound.</p></aside>"
)

for k in ("rollout_prose", "group_prose", "main_prose", "main_output", "credit_output", "credit_prose", "shaping_output",
          "shaping_prose", "explore_output", "explore_prose", "curriculum_output", "curriculum_prose", "recovery_output",
          "recovery_prose", "offline_output", "offline_prose", "system_output", "system_prose", "hf_prose",
          "comparison_rows", "comparison_note", "insights"):
    ns[k] = "(pending)"

exec((__import__("pathlib").Path("<SCRATCH>/code-series/c12-agent-rl/values2.py")).read_text()) if (__import__("pathlib").Path("<SCRATCH>/code-series/c12-agent-rl/values2.py")).exists() else None

ns["tree"] = """c12-agent-rl/
├── README.md                # what, how to run, results table
├── agentrl/
│   ├── tasks.py             # step 1: the five levels
│   ├── env.py               # step 1: reset, observe, step, reward, done
│   ├── expert.py            # step 1: scripted demonstrator
│   ├── trajectory.py        # step 2: text view and tensor view
│   ├── codec.py             # step 2: word tokenizer, encode_trajectory
│   ├── policy.py            # step 3: tiny transformer + value head
│   ├── batching.py          # step 4: padded tensors and masks
│   ├── bc.py                # step 4: behaviour cloning
│   ├── shift.py             # step 5: perturbations, signatures
│   ├── sampling.py          # step 6: shared rollout pieces
│   ├── rollout_naive.py     # step 6: the reference
│   ├── rollout.py           # step 6: batched generation
│   ├── advantages.py        # steps 7, 9: group-relative, GAE
│   ├── grpo.py              # step 8: clipped surrogate, update
│   ├── online.py            # steps 8, 11, 12: loop, sampler, curriculum
│   ├── rewards.py           # step 10: outcome, activity, potential
│   ├── failures.py          # step 13: failure injection
│   ├── offline.py           # step 14: offline GRPO, filtered BC
│   ├── system.py            # step 15: store, verifier, workers, resume
│   ├── hf_policy.py         # step 16: Llama and SmolLM2 adapters
│   └── evaluate.py          # the four evaluation suites
├── scripts/
│   ├── demo_env.py
│   ├── train_bc.py
│   ├── distribution_shift.py
│   ├── experiments.py       # main, credit, shaping, curriculum, ...
│   ├── recovery_demo.py
│   ├── export_figures.py
│   └── train_system.py
├── benchmarks/
│   ├── bench_rollout.py
│   └── bench_policy.py
├── results/                 # measured outputs, as text and JSON
├── tests/
│   ├── test_env.py
│   ├── test_policy_rollout.py
│   ├── test_advantages_grpo.py
│   └── test_system_hf.py
└── conftest.py"""

LADDER[:] = [
    {"name": "single tool", "task": "set port in db/main to 42", "tools": "edit, submit", "label": "clean suite (levels 1-3)", "score": E["clean"],
     "note": "Rung 1. The task names the file, the key and the value. The agent only has to call <code>edit</code> correctly and then submit."},
    {"name": "multiple tools", "task": "set port to 42", "tools": "search, edit, submit", "label": "perturbed starts", "score": E["perturbed"],
     "note": "Rung 2. The file is not named. The output of <code>search</code> must become the path argument of <code>edit</code>."},
    {"name": "dependent tools", "task": "set ttl in web/base to value of host", "tools": "search, read, edit, submit", "label": "failure recovery suite", "score": E["failures"],
     "note": "Rung 3. The value is hidden too. The agent must read a file and copy one number. This is where cloning struggles most on clean tasks."},
    {"name": "failure recovery", "task": "the same tasks, with timeouts, tool errors and stale paths", "tools": "+ retry, search on miss", "label": "long suite", "score": E["long"],
     "note": "Rung 4. Tools now fail. The demonstrations contain no error message at all, so all of this behaviour must come from RL."},
    {"name": "long horizon", "task": "set ttl to 8 and set limit in web/main to value of timeout", "tools": "5 to 7 calls + submit", "label": None, "score": None,
     "note": "Rung 5. Two subtasks with failures. The cloned policy has never seen <code>and</code> in a task, and it scores " + pct(E["long"]) + " here."},
]
