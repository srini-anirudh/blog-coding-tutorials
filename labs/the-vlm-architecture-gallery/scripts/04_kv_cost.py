"""Question 5, paying: the post's KV-cache arithmetic, then a real cache measured to check it.

The post's explainer uses 32 layers, 8 KV heads, head dim 128, bf16, and four settings:
576 (336^2 fixed), 2 880 (2x2 tiles + global), 1 300 (native 1024^2) and 10 240 (40-tile page).
"""

import _common
import torch

from gallery.cost import measure_kv_bytes, picture_cost

_common.threads()
rep = _common.Report("04_kv_cost")
settings = [("336^2 fixed", 576), ("2x2 tiles + global", 2880), ("native 1024^2", 1300), ("40-tile document page", 10240)]
rep("shape: 32 layers, 8 KV heads, head dim 128, bf16  ->  KV per token = 2*32*8*128*2 B = 128 KiB")
rep(f"{'setting':<24}{'tokens':>8}{'KV MiB (formula)':>18}{'KV MiB (measured)':>19}{'~words':>8}{'images in 128K':>16}")
rows = []
for name, tok in settings:
    c = picture_cost(tok)
    measured = measure_kv_bytes(tok, dtype=torch.bfloat16) / 2**20
    rep(f"{name:<24}{tok:>8,}{c['kv_mib']:>18.1f}{measured:>19.1f}{c['approx_words']:>8,}{c['images_in_context']:>16,}")
    rows.append({"setting": name, **c, "kv_mib_measured": measured})
rep("")
rep("measured = bytes in the KV cache of a transformers LlamaForCausalLM with that attention shape after a")
rep("forward pass over the prefix (tiny hidden size; only the attention shape sets the cache size).")
rep("~words uses the post's rule of thumb of 0.75 words per token; it is not measured.")
rep.save({"rows": rows})
