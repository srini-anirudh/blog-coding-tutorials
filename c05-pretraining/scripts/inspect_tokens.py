"""Look at what the data pipeline produces: token ids, EOS, boundaries, windows.

    python c05-pretraining/scripts/inspect_tokens.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pretrain.config import TrainConfig  # noqa: E402
from pretrain.corpus import load_documents  # noqa: E402
from pretrain.loader import PackedWindows  # noqa: E402
from pretrain.model import build_model, count_params  # noqa: E402
from pretrain.packing import document_ids, position_ids  # noqa: E402
from pretrain.shards import TokenShards  # noqa: E402
from pretrain.tokenizer import load_file, load_pretrained  # noqa: E402
from pretrain.utils import resolve  # noqa: E402


def show(tok, ids: list[int]) -> str:
    return " ".join(repr(tok.decode([i]))[1:-1].replace(" ", "_") for i in ids)


def main() -> None:
    data = resolve("data_cache/tinystories")
    bpe = load_file(data / "bpe4096.json")
    gpt2 = load_pretrained("openai-community/gpt2")
    text = "Once upon a time there was a little boy named Ben."
    for name, tok in (("bpe-4096", bpe), ("gpt2", gpt2)):
        ids = tok.encode(text)
        print(f"{name}: vocab {tok.vocab_size:,}, eos {tok.eos_id}, {len(ids)} tokens")
        print(f"  ids    {ids}")
        print(f"  pieces {show(tok, ids)}")

    # Two documents joined with no boundary token.
    a, b = "The dog was happy.", "Once upon a time"
    print("\nblind concatenation versus an explicit boundary")
    print(f"  joined          {show(bpe, bpe.encode(a + b))}")
    print(f"  separate + EOS  {show(bpe, bpe.encode(a) + [bpe.eos_id] + bpe.encode(b))}")
    a, b = "They played in the sun", "day was hot"  # a document cut mid-sentence
    print(f"  joined, no stop {show(bpe, bpe.encode(a + b))}")
    print(f"  separate + EOS  {show(bpe, bpe.encode(a) + [bpe.eos_id] + bpe.encode(b))}")
    docs = load_documents("val", data)[:2000]
    enc = bpe.encode_batch(docs)
    joined = bpe.encode_batch([x + y for x, y in zip(docs, docs[1:])])
    changed = sum(j != e + f for j, e, f in zip(joined, enc, enc[1:]))
    print(f"  {changed} of {len(joined)} real neighbouring document pairs tokenize differently when joined")

    # Parameter budget for the same model width with each vocabulary.
    cfg = TrainConfig.from_yaml(resolve("configs/tinystories.yaml"))
    d, blocks = cfg.model.hidden_size, count_params(build_model(cfg.model), non_embedding=True)
    for name, v in (("bpe-4096", 4096), ("gpt2", 50257)):
        print(f"embedding {name:9s} {v:6d} x {d} = {v * d / 1e6:5.2f}M params, "
              f"{100 * v * d / (v * d + blocks):.0f}% of a {blocks / 1e6:.1f}M-block model")

    shards = TokenShards(data, "train")
    ds = PackedWindows(shards, seq_len=256)
    print(f"\ntrain: {shards.num_tokens:,} tokens, {len(shards.shards)} shards, {len(ds):,} windows of L=256")
    x, y = ds[0]
    print(f"window 0: x {tuple(x.shape)} {x.dtype}  y {tuple(y.shape)} {y.dtype}")
    print(f"x[:8] {x[:8].tolist()}")
    print(f"y[:8] {y[:8].tolist()}   (x shifted by one)")
    doc, pos = document_ids(x, shards.eos_id), position_ids(x, shards.eos_id)
    k = int((x == shards.eos_id).nonzero()[0])
    print(f"first EOS at position {k}: x[{k - 2}:{k + 3}] {x[k - 2 : k + 3].tolist()}")
    print(f"  document ids {doc[k - 2 : k + 3].tolist()}")
    print(f"  positions    {pos[k - 2 : k + 3].tolist()}")
    print(f"documents touching window 0: {int(doc.max()) + 1}")


if __name__ == "__main__":
    main()
