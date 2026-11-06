"""Train a small MoE model twice, with and without the load-balancing loss, and watch the routing.

    python c04-llm/scripts/moe_demo.py --steps 300
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tinyllm import ByteTokenizer, TinyLLM, active_parameters, count_parameters, get_config  # noqa: E402
from tinyllm.utils import pick_device  # noqa: E402


def expert_load(model: TinyLLM) -> list[list[float]]:
    """Share of routed assignments that each expert received in the last forward, per layer."""
    loads = []
    for layer in model.layers:
        counts = layer.mlp.last_tokens_per_expert.float()
        loads.append((counts / counts.sum()).tolist())
    return loads


def train(aux_coef: float, args, data: torch.Tensor, device: torch.device) -> None:
    torch.manual_seed(0)
    config = get_config(
        "tiny", hidden_size=128, num_layers=2, num_heads=4, num_kv_heads=2, intermediate_size=256,
        max_seq_len=args.seq_len, num_experts=args.experts, num_experts_per_tok=args.top_k,
        router_aux_loss_coef=aux_coef,
    )
    model = TinyLLM(config).to(device).train()
    if aux_coef == args.aux_coefs[0]:
        print(f"model: {count_parameters(config)['total']:,} params, {active_parameters(config):,} active per token "
              f"({args.experts} experts, top-{args.top_k})")
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    g = torch.Generator().manual_seed(0)
    t0 = time.perf_counter()
    print(f"\nrouter_aux_loss_coef = {aux_coef}")
    for step in range(1, args.steps + 1):
        starts = torch.randint(0, data.numel() - args.seq_len, (args.batch_size,), generator=g)
        x = torch.stack([data[s : s + args.seq_len] for s in starts.tolist()]).to(device)
        out = model(x, labels=x)
        opt.zero_grad()
        out.loss.backward()
        opt.step()
        if step == 1 or step % (args.steps // 3) == 0:
            lm_loss = out.loss.item() - aux_coef * out.aux_loss.item()
            loads = expert_load(model)
            share = " ".join(f"{s:.2f}" for s in loads[0])
            worst = max(max(layer) for layer in loads)
            print(f"  step {step:4d}  lm loss {lm_loss:.3f}  aux {out.aux_loss.item() / config.num_layers:.3f}/layer  "
                  f"layer-0 shares [{share}]  busiest expert {worst:.2f}  ({time.perf_counter() - t0:.0f}s)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--experts", type=int, default=8)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--aux-coefs", type=float, nargs="+", default=[0.0, 0.01])
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    device = pick_device(args.device)
    path = ROOT / "data" / "tinyshakespeare.txt"
    if not path.exists():
        sys.exit("run scripts/train_sanity.py once first: it downloads the corpus")
    data = torch.tensor(ByteTokenizer().encode(path.read_text()))
    print(f"device={device}; a perfectly balanced router gives every expert {1 / args.experts:.3f} of the load")
    for coef in args.aux_coefs:
        train(coef, args, data, device)


if __name__ == "__main__":
    main()
