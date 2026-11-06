"""What freezing a tower saves on a real GPU: step time and peak memory, at about 1.1B parameters.

    python labs/how-vision-language-models-scale/scripts/07_frozen_step_gpu.py            # needs CUDA
    python labs/how-vision-language-models-scale/scripts/07_frozen_step_gpu.py --small    # CPU smoke test

The post's claim: a frozen LLM above a trained connector still costs about 4 of 6 FLOPs per parameter
per token, so it saves a third of the LLM's compute; its real saving is memory (no weight gradients and
no optimizer state). We time one AdamW step under four freezing settings and compare the measured
step time with the FLOP model's prediction.
"""

import argparse
import statistics
import time

import torch
from _common import Report, save_json, table

from vlmscale.flops import SETTINGS
from vlmscale.model import TwoTowerConfig, TwoTowerVLM, matmul_params, set_trainable

ap = argparse.ArgumentParser()
ap.add_argument("--small", action="store_true")
ap.add_argument("--steps", type=int, default=10)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() and not a.small else "cpu"
torch.manual_seed(0)
if a.small:
    cfg = TwoTowerConfig(image_size=64, patch_size=8, v_dim=64, v_layers=2, merge=2, l_dim=64, l_layers=2,
                         vocab=256, max_text=32)
    batch, text_len = 2, 32
else:  # a 0.30B encoder on 448 px / 14 px patches, 2x2 merge, a 0.83B LLM, 200 text tokens
    cfg = TwoTowerConfig(image_size=448, patch_size=14, v_dim=1024, v_layers=24, v_heads=16, merge=2,
                         l_dim=2048, l_layers=16, l_heads=16, vocab=32000, max_text=200)
    batch, text_len = 8, 200
r = Report("frozen_step_gpu" if not a.small else "frozen_step_small", device=dev)
model = TwoTowerVLM(cfg).to(dev)
n_v, n_c, n_l = (matmul_params(m) for m in (model.vision, model.connector, model.llm))
r(f"Encoder {n_v / 1e9:.3f}B matmul params, {cfg.patches} patches; connector {n_c / 1e6:.1f}M; LLM {n_l / 1e9:.3f}B,"
  f" {cfg.visual_tokens} visual + {text_len} text tokens. Batch {batch}, bf16 autocast, AdamW.")
pixels = torch.randn(batch, 3, cfg.image_size, cfg.image_size, device=dev)
text = torch.randint(0, cfg.vocab, (batch, text_len), device=dev)


def predicted(vision: bool, connector: bool, llm: bool) -> float:
    """Weight FLOPs per step from the post's k rule (attention left out)."""
    kv = 6 if vision else 2
    kl = 6 if llm else (4 if (connector or vision) else 2)
    return batch * (kv * n_v * cfg.patches + 6 * n_c * cfg.visual_tokens + kl * n_l * (cfg.visual_tokens + text_len))


def sync() -> None:
    if dev == "cuda":
        torch.cuda.synchronize()


rows, payload = [], {}
base_time = base_flops = None
for name, flags in SETTINGS.items():
    set_trainable(model, *flags)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-5)
    if dev == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(a.steps + 3):
        sync()
        t0 = time.perf_counter()
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            loss = model.loss(pixels, text, text)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        sync()
        if i >= 3:
            times.append(time.perf_counter() - t0)
    step = statistics.median(times)
    peak = torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else float("nan")
    flops = predicted(*flags)
    if base_time is None:
        base_time, base_flops = step, flops
    trainable = sum(p.numel() for p in params)
    payload[name] = {"step_s": step, "peak_gib": peak, "predicted_flops": flops, "trainable_params": trainable,
                     "times": times}
    rows.append([name, f"{trainable / 1e9:.3f}B", f"{1000 * step:.1f}", f"{step / base_time:.2f}",
                 f"{flops / base_flops:.2f}", f"{flops / step / 1e12:.0f}", f"{peak:.1f}"])
    del opt
r(table(rows, ["setting", "trainable", "step ms (median)", "time vs all", "FLOP model vs all",
               "weight TFLOP/s", "peak GiB"]))
r(f"Median of {a.steps} steps after 3 warm-up steps. 'FLOP model' uses k = 6 / 4 / 2 and leaves attention out.")
save_json(r.path.stem, {"config": cfg.__dict__, "batch": batch, "text_len": text_len, "settings": payload}, device=dev)
r.save()
