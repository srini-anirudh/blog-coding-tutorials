#!/usr/bin/env python3
"""Build blogs/train-an-agent-with-rl/index.html: exact listings from the code, real numbers from results/."""
import html, json, re, sys, pathlib
HERE = pathlib.Path(__file__).parent
CODE = pathlib.Path("<CODE>")
PKG = CODE / "c12-agent-rl"
SITE = pathlib.Path("<SITE>")
RES = PKG / "results"
sys.path.insert(0, str(PKG))

def J(name, default=None):
    p = RES / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else default

def thin(run):
    keep = ("iter", "success", "return", "zero_var_groups", "steps", "submitted", "eval", "clip_frac", "stage")
    return {"name": run["name"], "start": run["start"], "seconds": run.get("seconds"),
            "history": [{k: h[k] for k in keep if k in h} for h in run["history"] if "eval" in h or h["iter"] % 5 == 0]}

def data():
    from agentrl.codec import WordCodec, encode_trajectory
    from agentrl.env import FileSystemEnv, run_episode
    from agentrl.expert import ScriptedExpert
    from agentrl.tasks import sample_task
    codec = WordCodec()
    t3 = sample_task(3, 1)
    tr3 = run_episode(FileSystemEnv(t3), ScriptedExpert(t3.text))
    t2 = sample_task(2, 1)
    tr2 = encode_trajectory(run_episode(FileSystemEnv(t2), ScriptedExpert(t2.text)), codec)
    logp = [0.0] * len(tr2.tokens)
    bc = PKG / "runs" / "bc.pt"
    if bc.exists():
        import torch
        from agentrl.batching import collate
        from agentrl.policy import TinyPolicy, token_logprobs
        pol = TinyPolicy.load(str(bc)).eval()
        b = collate([tr2], codec.pad)
        with torch.no_grad():
            logp = token_logprobs(pol(b.tokens).logits, b.tokens)[0].tolist()
    d = {
        "demo": {"task": t3.text, "source": t3.subtasks[0].source_key, "files": t3.files,
                 "steps": [{"action": s.action, "observation": s.observation} for s in tr3.steps]},
        "tape": {"tokens": [codec.vocab[i] for i in tr2.tokens], "ids": tr2.tokens, "mask": tr2.action_mask,
                 "logp": [round(x, 3) for x in logp], "states": tr2.state_positions, "adv": 2.47},
        "shift": J("shift_bc", {"rows": []}),
        "replay": J("replay", []),
    }
    main, credit, shaping, curriculum, offline, exploration, groupsize = (J(n) for n in
        ("main", "credit", "shaping", "curriculum", "offline", "exploration", "groupsize"))
    curves = {}
    if main:
        curves["main"] = {"title": "BC, then GRPO on the full mix", "suite": "failures", "runs": [thin(main)], "note": NOTES.get("main", "")}
    if credit:
        curves["credit"] = {"title": "Trajectory vs action credit", "suite": "long", "runs": [thin(r) for r in credit], "note": NOTES.get("credit", "")}
    if curriculum:
        curves["curriculum"] = {"title": "Curriculum vs hardest first", "suite": "long", "runs": [thin(r) for r in curriculum], "note": NOTES.get("curriculum", "")}
    if offline:
        fb = offline["filtered"]
        steps = fb["history"][-1]["step"] if fb["history"] else 1
        iters = len(offline["online"]["history"])
        fbrun = {"name": "filtered BC", "start": fb["start"], "history": [{"iter": h["step"] * iters / steps, "eval": h["eval"]} for h in fb["history"]]}
        curves["offline"] = {"title": "Online vs offline", "suite": "failures", "runs": [thin(offline["online"]), {"name": "offline GRPO", "start": offline["offline"]["start"], "history": [h for h in offline["offline"]["history"] if "eval" in h]}, fbrun], "note": NOTES.get("offline", "")}
    if exploration:
        curves["exploration"] = {"title": "Sampling temperature", "suite": "failures", "runs": [thin(r) for r in exploration], "note": NOTES.get("exploration", "")}
    if groupsize:
        curves["groupsize"] = {"title": "Group size at equal rollouts", "suite": "failures", "runs": [thin(r) for r in groupsize], "note": NOTES.get("groupsize", "")}
    if shaping:
        curves["shaping"] = {"title": "Outcome vs hack vs potential", "suite": "clean", "runs": [thin(r) for r in shaping], "note": NOTES.get("shaping", "")}
    d["curves"] = curves
    if shaping:
        hk = next(r for r in shaping if r["name"] == "activity")
        d["hack"] = {"iters": [h["iter"] for h in hk["history"]], "ret": [round(h["return"], 3) for h in hk["history"]],
                     "success": [round(h["success"], 3) for h in hk["history"]],
                     "traces": [hk["before"][0].split("\n"), hk["samples"][0].split("\n")]}
    d["ladder"] = LADDER
    d["system"] = J("system", [{"total": 1, "clock": {}, "stored": 0, "checked": 0, "mismatches": 0}])
    return d

NOTES, LADDER = {}, []

def listing(src, start, end=None, hl=None, cap=None, lang="python", attrs=""):
    """Exact excerpt. start and end are regexes on lines; end is inclusive. Ending on a blank line trims it."""
    path = CODE / src
    lines = path.read_text().split("\n")
    i = next(k for k, l in enumerate(lines) if re.search(start, l))
    if end is None:
        j = len(lines) - 1
        while j > i and not lines[j].strip():
            j -= 1
    elif isinstance(end, int):
        j = i + end - 1
    else:
        j = next(k for k, l in enumerate(lines) if k > i and re.search(end, l))
    while not lines[j].strip():
        j -= 1
    code = html.escape("\n".join(lines[i:j + 1]), quote=False)
    h = f' data-highlight="{hl}"' if hl else ""
    c = f"\n          <figcaption>{cap}</figcaption>" if cap else ""
    return (f'<figure class="code-listing" data-src="{src}" data-lines="{i + 1}-{j + 1}" data-lang="{lang}"{h}{attrs}>\n'
            f"          <pre><code>{code}</code></pre>{c}\n        </figure>")

def result(name):
    p = RES / name
    return html.escape(p.read_text().rstrip("\n"), quote=False) if p.exists() else "(pending)"

def build():
    src = "".join((HERE / n).read_text() for n in ("post_head.html", "post_a.html", "post_b.html", "post_c.html"))
    ns = {}
    exec((HERE / "values.py").read_text(), {"J": J, "RES": RES, "PKG": PKG, "ns": ns, "NOTES": NOTES, "LADDER": LADDER})
    def sub_l(m):
        a = [x.strip() for x in m.group(1).split("|")]
        src_, start = a[0], a[1]
        end = a[2] if len(a) > 2 and a[2] else None
        if end and end.isdigit():
            end = int(end)
        return listing("c12-agent-rl/" + src_, start, end, a[3] if len(a) > 3 and a[3] else None, a[4] if len(a) > 4 else None)
    out = re.sub(r"\{\{L (.*?)\}\}", sub_l, src, flags=re.S)
    out = re.sub(r"\{\{R (.*?)\}\}", lambda m: result(m.group(1).strip()), out)
    def sub_n(m):
        key = m.group(1).strip()
        if key not in ns:
            raise KeyError(key)
        return str(ns[key])
    out = re.sub(r"\{\{N (.*?)\}\}", sub_n, out)
    d = data()
    out = out.replace("{{DATA}}", json.dumps(d, separators=(",", ":")).replace("</", "<\\/"))
    out = out.replace("{{FIGURES}}", (HERE / "figures.js").read_text())
    dest = SITE / "blogs" / "train-an-agent-with-rl"
    dest.mkdir(exist_ok=True)
    (dest / "index.html").write_text(out)
    print("written", dest / "index.html", len(out))

build()
