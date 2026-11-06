"""Check the gallery's token counts against real configs on the Hugging Face Hub (needs network).

Downloads only config.json (and Qwen2-VL's preprocessor_config.json): a few kB per model.
Custom-code repositories are read as JSON; their code is never run.
"""

import _common
import transformers

from gallery.hub import check_all

transformers.logging.set_verbosity_error()
rep = _common.Report("07_hub_configs")
rows = check_all()
rep(f"{'model':<14}{'repo':<36}{'post says':<30}{'from config':>12}{'match':>7}{'KV KiB/tok':>11}{'KV MiB':>8}")
for r in rows:
    if "error" in r:
        rep(f"{r['model']:<14}{r['repo']:<36}{r['post_says']:<30}  not checked: {r['error']}")
        continue
    kv = f"{r['kv_kib_per_token_bf16']:>11.0f}{r['kv_mib_for_these_tokens']:>8.1f}" if "lm_shape" in r else f"{'-':>11}{'-':>8}"
    rep(f"{r['model']:<14}{r['repo']:<36}{r['post_says']:<30}{r['tokens_from_config']:>12,}"
        f"{'yes' if r['matches_post'] else 'NO':>7}{kv}")
rep("")
for r in rows:
    if "how" in r:
        rep(f"{r['model']:<14}{r['how']}" + (f"; LM {r['lm_shape']}" if "lm_shape" in r else ""))
rep("")
rep("KV KiB/tok = 2 * layers * kv_heads * head_dim * 2 bytes (bf16), from the LM config; KV MiB = that times")
rep("the token count in the 'from config' column. Gemma 3 uses sliding-window attention in most layers, so its")
rep("real cache for long contexts is smaller than this full-attention figure.")
rep.save(rows)
