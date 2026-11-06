"""Figure 7 of the post: the parameter split, measured from the real checkpoints' safetensors headers.

Network: reads only headers (no weights). For each repo prints vision / connector / adapter / language
parameters, the vision share, and the post's Figure 7 numbers where the post has that model.
Writes results/03_param_split.{txt,json}. The JSON is also what scripts/04_connectors.py compares against.
"""

import _path  # noqa: F401

from collections import defaultdict

from vlmfam import split as S
from vlmfam.report import Report

# (label, repo, post's Figure 7 [vision, connector, language] in billions or None). Gated repos via unsloth mirrors.
REPOS = [
    ("LLaVA-1.5 7B", "llava-hf/llava-1.5-7b-hf", None),
    ("LLaVA-NeXT 7B", "llava-hf/llava-v1.6-vicuna-7b-hf", (0.3035, 0.021, 6.74)),
    ("LLaVA-NeXT 34B", "llava-hf/llava-v1.6-34b-hf", (0.3035, 0.0587, 34.39)),
    ("LLaVA-OneVision 0.5B", "llava-hf/llava-onevision-qwen2-0.5b-ov-hf", None),
    ("LLaVA-OneVision 72B", "llava-hf/llava-onevision-qwen2-72b-ov-hf", None),
    ("Qwen2-VL 2B", "Qwen/Qwen2-VL-2B-Instruct", (0.675, 0, 1.5)),
    ("Qwen2-VL 72B", "Qwen/Qwen2-VL-72B-Instruct", (0.675, 0, 72)),
    ("Qwen2.5-VL 72B", "Qwen/Qwen2.5-VL-72B-Instruct", None),
    ("Qwen3-VL 2B", "Qwen/Qwen3-VL-2B-Instruct", None),
    ("Qwen3-VL 235B-A22B", "Qwen/Qwen3-VL-235B-A22B-Instruct", None),
    ("InternVL3 78B", "OpenGVLab/InternVL3-78B-hf", None),
    ("InternVL3.5 8B", "OpenGVLab/InternVL3_5-8B-HF", (0.3, 0, 8.2)),
    ("InternVL3.5 38B", "OpenGVLab/InternVL3_5-38B-HF", (5.5, 0, 32.8)),
    ("InternVL3.5 241B-A28B", "OpenGVLab/InternVL3_5-241B-A28B-HF", (5.5, 0, 235.1)),
    ("Molmo-7B-D", "allenai/Molmo-7B-D-0924", (0.29, 0.11, 7.6)),
    ("Molmo-72B", "allenai/Molmo-72B-0924", (0.29, 0.31, 72)),
    ("Idefics2 8B", "HuggingFaceM4/idefics2-8b", None),
    ("Idefics3 8B", "HuggingFaceM4/Idefics3-8B-Llama3", None),
    ("SmolVLM 256M", "HuggingFaceTB/SmolVLM-256M-Instruct", (0.093, 0, 0.135)),
    ("SmolVLM 2.2B", "HuggingFaceTB/SmolVLM-Instruct", (0.4, 0, 1.7)),
    ("Gemma 3 4B", "unsloth/gemma-3-4b-it", None),
    ("Gemma 3 27B", "unsloth/gemma-3-27b-it", (0.417, 0, 25.6)),
    ("DeepSeek-VL2-Tiny", "deepseek-ai/deepseek-vl2-tiny", None),
    ("DeepSeek-VL2", "deepseek-ai/deepseek-vl2", None),
    ("DeepSeek-OCR", "deepseek-ai/DeepSeek-OCR", (0.38, 0, 3)),
    ("MiniCPM-V 4.5", "openbmb/MiniCPM-V-4_5", None),
    ("Llama 3.2 11B Vision", "unsloth/Llama-3.2-11B-Vision-Instruct", None),
    ("Llama 3.2 90B Vision", "unsloth/Llama-3.2-90B-Vision-Instruct", None),
    ("Pixtral 12B", "mistral-community/pixtral-12b", (0.4, 0, 12)),
    ("Kimi-VL A3B", "moonshotai/Kimi-VL-A3B-Instruct", (0.4, 0, 16)),
    ("NVLM-D 72B", "nvidia/NVLM-D-72B", None),
    ("Aria", "rhymes-ai/Aria", None),
    ("GLM-4.1V 9B", "zai-org/GLM-4.1V-9B-Thinking", None),
    ("Ovis2.5 9B", "AIDC-AI/Ovis2.5-9B", None),
]


def b(n: int) -> str:
    return f"{n / 1e9:.3f}"


r = Report("03_param_split")
rows, data, failed = [], [], []
for label, repo, post in REPOS:
    try:
        sizes = S.tensor_sizes(repo)
    except Exception as e:  # not a safetensors repo, gated, or removed
        failed.append((label, repo, type(e).__name__))
        continue
    parts = S.split(sizes)
    total = sum(parts.values())
    share = S.vision_share(parts)
    post_share = post[0] / sum(post) if post else None
    rows.append([label, b(parts["vision"]), b(parts["connector"]), b(parts["adapter"]), b(parts["language"]),
                 b(total), f"{share:.1%}", "" if post is None else f"{post[0]:.3f} / {post[2]:.2f} ({post_share:.1%})"])
    # the largest tensor groups per part, so a reader can audit the name rules
    groups = defaultdict(int)
    for name, n in sizes.items():
        part = S.classify(name)
        key = ".".join(name.split(".")[:3])
        groups[(part, key)] += n
    top = {p: sorted(((k, v) for (pp, k), v in groups.items() if pp == p), key=lambda kv: -kv[1])[:3]
           for p in ("vision", "connector")}
    data.append({"label": label, "repo": repo, "parts": parts, "total": total, "vision_share": share,
                 "post_figure7_billions": post, "top_groups": {p: [[k, v] for k, v in t] for p, t in top.items()}})

r("Parameters in billions, from safetensors headers. 'adapter' = cross-attention layers inside the LLM.")
r("Last column: the post's Figure 7 (vision / language, and the vision share it implies).")
r("")
r.table(["model", "vision", "connector", "adapter", "language", "total", "vision share", "post Fig. 7"], rows)
r("")
r("Name groups counted as vision and connector (largest three), to audit the rules in vlmfam/split.py:")
for d in data:
    v = ", ".join(f"{k} {n / 1e6:.0f}M" for k, n in d["top_groups"]["vision"])
    c = ", ".join(f"{k} {n / 1e6:.1f}M" for k, n in d["top_groups"]["connector"])
    r(f"  {d['label']}: vision [{v}]  connector [{c}]")
if failed:
    r("")
    r("Not measured (no safetensors headers readable without a token):")
    for f in failed:
        r(f"  {f[0]} ({f[1]}): {f[2]}")
r.save({"models": data, "failed": failed})
