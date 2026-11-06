"""Query latency of the reference and the tensor index, and the cost of embedding.

    python c11-agent-memory/benchmarks/bench_index.py

Prints tables and saves results/bench_index.json. CPU, 2 threads.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

from agentmem import ContextBudgeter, Block, HashingEmbedder, NaiveIndex, TensorIndex, TransformerEmbedder, count_tokens  # noqa: E402

torch.set_num_threads(2)
D = 384


def unit(n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.nn.functional.normalize(torch.randn(n, D, generator=g), dim=-1)


def timed(fn, repeat: int) -> float:
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples) * 1000


def main() -> None:
    out: dict = {"hardware": "Apple M5 CPU, 2 threads", "index": [], "embed": [], "tokens": {}}
    print(f"index query latency, D={D}, top-10 (Apple M5 CPU, 2 threads)\n{'N':>8}{'naive ms':>12}{'tensor ms':>12}{'speed-up':>10}{'matrix MB':>11}")
    q = unit(1, seed=1)[0]
    for n in (1_000, 10_000, 100_000, 1_000_000):
        vecs = unit(n)
        ids = [str(i) for i in range(n)]
        fast = TensorIndex(D, capacity=n)
        fast.matrix[:n] = vecs
        fast.alive[:n] = True
        fast.size = n
        fast.slot_of = dict(zip(ids, range(n)))
        fast.id_of = list(ids)
        t_fast = timed(lambda: fast.search(q, 10), 20)
        t_naive = None
        if n <= 10_000:
            naive = NaiveIndex(D)
            naive.add(ids, vecs)
            t_naive = timed(lambda: naive.search(q, 10), 3)
        speed = f"{t_naive / t_fast:>9.0f}x" if t_naive else f"{'':>10}"
        print(f"{n:>8}{(f'{t_naive:.1f}' if t_naive else 'skipped'):>12}{t_fast:>12.2f}{speed}{fast.matrix.numel() * 4 / 1e6:>11.1f}")
        out["index"].append({"n": n, "naive_ms": t_naive, "tensor_ms": t_fast, "matrix_mb": fast.matrix.numel() * 4 / 1e6})

    print(f"\nincremental add of 1,000 single memories\n  tensor index: {timed(lambda: _adds(), 3):.1f} ms total")

    texts = [f"The deploy region for project{i} is eu-west-{i % 3}." for i in range(256)]
    print(f"\nembedding {len(texts)} short memories (Apple M5 CPU, 2 threads)\n{'embedder':<24}{'dim':>5}{'ms total':>10}{'ms/memory':>11}")
    embedders = [("HashingEmbedder", HashingEmbedder())]
    try:
        embedders.append(("all-MiniLM-L6-v2", TransformerEmbedder()))
    except Exception as e:  # offline: the rest of the benchmark still runs
        print("  MiniLM unavailable:", type(e).__name__)
    for name, emb in embedders:
        ms = timed(lambda: emb.encode(texts), 3)
        print(f"{name:<24}{emb.dim:>5}{ms:>10.1f}{ms / len(texts):>11.3f}")
        out["embed"].append({"name": name, "dim": emb.dim, "ms": ms, "ms_per": ms / len(texts)})

    print("\nparaphrase test: cosine of a question with the memory that answers it vs. an unrelated memory")
    pairs = [("where does atlas run in production", "atlas deploy region: us-east-2", "lunch moved to thursday"),
             ("who is responsible for cobalt", "cobalt owner: chen", "atlas deploy region: us-east-2"),
             ("what is the command to run the checks", "atlas test command: pytest -x", "the design review moved")]
    print(f"{'embedder':<24}{'answering':>11}{'unrelated':>11}")
    for name, emb in embedders:
        a = []
        b = []
        for q_text, good, bad in pairs:
            v = emb.encode([q_text, good, bad])
            a.append(float(v[0] @ v[1]))
            b.append(float(v[0] @ v[2]))
        print(f"{name:<24}{sum(a) / 3:>11.3f}{sum(b) / 3:>11.3f}")
        out["embed"].append({"name": name, "paraphrase_answering": sum(a) / 3, "paraphrase_unrelated": sum(b) / 3})

    samples = ["The deploy region for atlas is eu-west-1. Heads up: we moved it last Tuesday after the audit.",
               "[3/7] compile atlas: done in 41.3s (cache miss)\nMigrationError: schema version 41 does not match code version 62",
               "def retrieve(self, query: str, now: float, k: int = 5) -> list[Scored]:\n    return self.retriever.search(query, now, k)"]
    try:
        from agentmem.tokens import HFTokenCounter
        hf = HFTokenCounter()
        print("\ntoken estimate vs. the SmolLM2 tokenizer\n  estimate   real   ratio")
        for s in samples:
            print(f"  {count_tokens(s):>8} {hf(s):>6} {hf(s) / count_tokens(s):>7.2f}")
            out["tokens"][s[:20]] = [count_tokens(s), hf(s)]
    except Exception as e:
        print("  tokenizer unavailable:", type(e).__name__)

    blocks = [Block("memory", f"- memory number {i} about project {i % 7}", value=1 / (i + 1), id=str(i)) for i in range(1000)]
    packer = ContextBudgeter()
    ms = timed(lambda: packer.pack(blocks, 800), 5)
    print(f"\npacking 1,000 candidate blocks under B=800: {ms:.2f} ms")
    out["pack_ms"] = ms
    (HERE / "results" / "bench_index.json").write_text(json.dumps(out, indent=1))


def _adds() -> None:
    idx = TensorIndex(D, capacity=16)
    vecs = unit(1000)
    for i in range(1000):
        idx.add([str(i)], vecs[i:i + 1])


if __name__ == "__main__":
    main()
