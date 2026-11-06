"""Past softmax: the reassociation, and why a fixed state needs the delta rule.

Post section: "Past softmax: the 2026 hybrids". Part 1 checks that causal linear attention computed in parallel
equals the running-sum recurrence (C3's attnzoo.linear). Part 2 stores n key-value pairs in a 64 x 64 state and
reads them back. No training: this isolates the memory, not a trained hybrid model. The post's open question,
whether trained hybrids match full attention at scale, is not touched here.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

import organs  # noqa: F401  (puts C3's attnzoo on the path)
from attnzoo.linear import causal_linear_attention_parallel, causal_linear_attention_recurrent, elu_feature_map
from organs.memory import read_softmax, read_state, recall_cosine, recall_task, write_delta, write_linear
from organs.report import Report


def main() -> None:
    rep = Report("09_beyond_softmax")
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(2, 4, 256, 32, generator=g, dtype=torch.float64) for _ in range(3))
    q, k = elu_feature_map(q), elu_feature_map(k)  # phi(x) = elu(x) + 1, positive features
    par, (rec, _) = causal_linear_attention_parallel(q, k, v), causal_linear_attention_recurrent(q, k, v)
    rep("1. Causal linear attention, (B=2, H=4, T=256, d=32), phi = elu + 1, C3's implementations:")
    rep(f"   parallel vs recurrent, max |diff| = {(par - rec).abs().max():.1e}")
    state = 4 * 32 * 32 * 8 + 4 * 32 * 8  # S (H, d, d) and the normalizer z (H, d), float64, per sequence
    rep(f"   state per sequence: {state:,} bytes at any length. A KV cache at T = 256: {2 * 4 * 256 * 32 * 8:,} bytes,")
    rep("   and it keeps growing with T.")
    rep("")

    d = 64
    rep(f"2. Associative recall in a {d} x {d} state, random unit values, mean over 3 seeds. Each cell is the mean")
    rep("   cosine between the read-out and the stored value (1 = perfect, 0 = noise). Softmax reads a full cache.")
    rows = []
    for shared, label in ((0.0, "independent"), (1.0, "correlated")):
        for n in (16, 32, 64, 128):
            cells = {"softmax": [], "linear": [], "delta": []}
            for seed in range(3):
                keys, values = recall_task(n, d, seed, shared)
                cells["softmax"].append(recall_cosine(read_softmax(keys, values, keys, 30.0), values).mean().item())
                cells["linear"].append(recall_cosine(read_state(write_linear(keys, values), keys), values).mean().item())
                cells["delta"].append(recall_cosine(read_state(write_delta(keys, values), keys), values).mean().item())
            rows.append([label, n, *(f"{sum(c) / 3:.3f}" for c in cells.values()), f"{2 * n * d * 8:,}", f"{d * d * 8:,}"])
    rep.table(["keys", "pairs n", "softmax (scale 30)", "linear S += v k^T", "delta rule", "cache bytes", "state bytes"],
              rows)
    rep("   independent: random keys, nearly orthogonal. The two state rules are close up to n = 32; past n = d")
    rep("   the delta rule's overwrites cost more than the additive blur. Both lose to the full cache.")
    rep("   correlated: keys share a direction (mean cosine about 0.5). The additive state blurs them together;")
    rep("   the delta rule removes what the state already returns for k before writing, and does better.")
    rep("")

    rep("3. Updating a memory: write 16 pairs, then write the same 16 keys again with new values.")
    keys, old = recall_task(16, d, 0)
    _, new = recall_task(16, d, 1)
    both_k, both_v = torch.cat([keys, keys]), torch.cat([old, new])
    rows = []
    for name, s in (("linear", write_linear(both_k, both_v)), ("delta rule", write_delta(both_k, both_v))):
        out = read_state(s, keys)
        rows.append([name, f"{recall_cosine(out, new).mean():.3f}", f"{recall_cosine(out, old).mean():.3f}"])
    rep.table(["state", "cosine to new value", "cosine to old value"], rows)
    rep("   The additive state returns a blend of old and new; the delta rule returns the new value.")
    rep("")

    rep("4. The gate: decay alpha < 1 fades old memories. n = 48 pairs, delta rule, beta = 1.")
    rows = []
    for alpha in (1.0, 0.98, 0.9):
        old, new = [], []
        for seed in range(3):
            keys, values = recall_task(48, d, seed)
            cos = recall_cosine(read_state(write_delta(keys, values, alpha=alpha), keys), values)
            old.append(cos[:8].mean().item())
            new.append(cos[-8:].mean().item())
        rows.append([alpha, f"{sum(old) / 3:.3f}", f"{sum(new) / 3:.3f}"])
    rep.table(["alpha", "oldest 8 pairs", "newest 8 pairs"], rows)
    rep("   Cosine ignores scale, so the decay shows only once interference from later writes dominates.")
    print(f"saved {rep.save()}")


if __name__ == "__main__":
    main()
