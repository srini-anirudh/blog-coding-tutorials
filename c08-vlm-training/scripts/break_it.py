"""Break it on purpose, on the real model: failures that raise, and failures that train quietly.

    python scripts/break_it.py image_tokens    tiles and <image> slots disagree: raises, and names the mismatch
    python scripts/break_it.py loss_mask       labels supervise the prompt too: the logged loss looks better than it is
    python scripts/break_it.py block_mask      a packed row with a plain causal mask: samples leak into each other
    python scripts/break_it.py max_length      too many tiles: silent cost, then the processor refuses
    python scripts/break_it.py high_lr         a learning rate far too high: the loss does not recover
    python scripts/break_it.py frozen_loc      new <loc_k> rows that never learn: the model cannot choose between them
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vlmtrain.c07_adapter import build_c07  # noqa: E402
from vlmtrain.data import make_example, sample  # noqa: E402
from vlmtrain.device import cap_threads, describe_hardware, pick_device, seed_everything  # noqa: E402
from vlmtrain.freeze import set_trainable  # noqa: E402
from vlmtrain.grounding import loc_token  # noqa: E402
from vlmtrain.packing import block_causal_mask, pack  # noqa: E402
from vlmtrain.processor import IGNORE, TinyVLMProcessor  # noqa: E402
from vlmtrain.train import per_sample_loss  # noqa: E402


def caption_batch(proc: TinyVLMProcessor, start: int, n: int = 8) -> dict:
    return proc([make_example("caption", i) for i in range(start, start + n)])


def answer_loss(model, b: dict, device: torch.device) -> float:
    """Loss on the answer tokens only, whatever labels the run trained with."""
    with torch.no_grad():
        out = model(b["input_ids"].to(device), b["pixel_values"].to(device), attention_mask=b["attention_mask"].to(device))
    idx = torch.arange(b["input_ids"].shape[0], device=device)[:, None].expand_as(b["input_ids"])
    sums, counts = per_sample_loss(out.logits, b["labels"].to(device), idx, b["input_ids"].shape[0])
    return (sums.sum() / counts.sum()).item()


def train_projector(lr: float, steps: int, device: torch.device, wrong_labels: bool = False) -> list[float]:
    """A few Stage 1 steps. Returns the loss the training loop saw at each step."""
    seed_everything(0)
    _, model, proc = build_c07()
    set_trainable(model, "align")
    model.to(device).train()
    opt = torch.optim.AdamW(model.projector.parameters(), lr=lr, weight_decay=0.0)
    seen = []
    for step in range(steps):
        b = caption_batch(proc, step * 8)
        labels = b["input_ids"].masked_fill(b["attention_mask"] == 0, IGNORE) if wrong_labels else b["labels"]
        loss = model(b["input_ids"].to(device), b["pixel_values"].to(device),
                     attention_mask=b["attention_mask"].to(device), labels=labels.to(device)).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        seen.append(loss.item())
    held_out = caption_batch(proc, 100_000)
    print(f"   lr {lr:g}: logged loss {seen[0]:.2f} -> {seen[-1]:.2f}, answer-only loss on held-out captions "
          f"{answer_loss(model, held_out, device):.2f}")
    return seen


def image_tokens(model, proc, device) -> None:
    b = proc([make_example("doc", 0), make_example("vqa", 0)])
    print(f"input_ids {tuple(b['input_ids'].shape)}, pixel_values {tuple(b['pixel_values'].shape)}, "
          f"<image> slots {int((b['input_ids'] == proc.image_token_id).sum())}")
    print("Dropping the last tile from pixel_values:")
    try:
        model(b["input_ids"].to(device), b["pixel_values"][:-1].to(device), attention_mask=b["attention_mask"].to(device))
    except ValueError as err:
        print(f"ValueError: {err}")


def loss_mask(model, proc, device) -> None:
    b = caption_batch(proc, 0)
    print(f"supervised positions in one batch: correct mask {int((b['labels'] != IGNORE).sum())}, "
          f"everything {int(b['attention_mask'].sum())}")
    print("40 Stage 1 steps with the correct mask:")
    train_projector(1e-3, 40, device)
    print("40 Stage 1 steps supervising the prompt and the <image> slots too:")
    train_projector(1e-3, 40, device, wrong_labels=True)


def block_mask(model, proc, device) -> None:
    enc = [proc.encode(make_example("vqa", 1)), proc.encode(make_example("ocr", 1))]
    b = pack(enc, 512, proc.pad_id)
    first = len(enc[b["rows"][0][0]].input_ids)
    used = int((b["seg_ids"] >= 0).sum())
    kwargs = dict(position_ids=b["position_ids"].to(device))
    plain = block_causal_mask(torch.where(b["seg_ids"] >= 0, 0, -1)).to(device)  # one big segment
    with torch.no_grad():
        ok = model(b["input_ids"].to(device), b["pixel_values"].to(device),
                   attention_mask=b["attention_mask"].to(device), **kwargs).logits
        leaky = model(b["input_ids"].to(device), b["pixel_values"].to(device), attention_mask=plain, **kwargs).logits
    print(f"one packed row: first sample {first} tokens, second sample {used - first} tokens")
    print(f"largest logit change, first sample:  {(ok[0, :first] - leaky[0, :first]).abs().max():.2e}")
    print(f"largest logit change, second sample: {(ok[0, first:used] - leaky[0, first:used]).abs().max():.2f}")
    sums_ok, counts = per_sample_loss(ok, b["labels"].to(device), b["seg_ids"].to(device), 2)
    sums_bad, _ = per_sample_loss(leaky, b["labels"].to(device), b["seg_ids"].to(device), 2)
    print(f"loss of the second sample: block mask {(sums_ok[1] / counts[1]).item():.3f}, "
          f"plain causal mask {(sums_bad[1] / counts[1]).item():.3f}")


def max_length(model, proc, device) -> None:
    doc = make_example("doc", 0)["images"][0]
    print(f"{'max_tiles':>9}{'tiles':>7}{'tokens':>8}{'forward ms':>12}")
    for max_tiles in (1, 4, 16, 36):
        p = TinyVLMProcessor(proc.tok, proc.tokens_per_tile, proc.tile, "tile", max_tiles)
        side = 224 * int(max_tiles**0.5)
        try:
            e = p([sample("doc", [doc.resize((side, side))], "What is the total?", "1")])
        except ValueError as err:
            print(f"{max_tiles:>9}  ValueError: {err}")
            continue
        t0 = time.time()
        with torch.no_grad():
            model(e["input_ids"].to(device), e["pixel_values"].to(device), attention_mask=e["attention_mask"].to(device))
        print(f"{max_tiles:>9}{e['pixel_values'].shape[0]:>7}{e['input_ids'].shape[1]:>8}{(time.time() - t0) * 1000:>12.0f}")


def high_lr(model, proc, device) -> None:
    print("40 Stage 1 steps at the configured learning rate, then at 1000 times that:")
    train_projector(1e-3, 40, device)
    train_projector(1.0, 40, device)


def frozen_loc(model, proc, device) -> None:
    ids = proc.tok.convert_tokens_to_ids([loc_token(k) for k in range(32)])
    emb = model.lm.get_input_embeddings().weight.detach()
    print(f"embedding std per dimension: pretrained vocabulary {emb[:ids[0] - 1].std(0).mean():.2e}, "
          f"the 32 <loc_k> rows {emb[ids].std(0).mean():.2e}")
    b = proc([make_example("grounding", 0)], for_generation=True)
    with torch.no_grad():
        logits = model(b["input_ids"].to(device), b["pixel_values"].to(device),
                       attention_mask=b["attention_mask"].to(device)).logits[0, -1]
    print(f"next-token logits on a real grounding prompt: spread over the whole vocabulary "
          f"{(logits.max() - logits.min()).item():.2f}, spread over the 32 <loc_k> tokens "
          f"{(logits[ids].max() - logits[ids].min()).item():.2e}")
    print("The output head is tied to these rows. Until they move apart, the model cannot prefer one bin over another.")


BREAKS = {f.__name__: f for f in (image_tokens, loss_mask, block_mask, max_length, high_lr, frozen_loc)}


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in BREAKS:
        raise SystemExit(f"usage: break_it.py {{{'|'.join(BREAKS)}}}")
    cap_threads()
    device = pick_device()
    print(describe_hardware(device))
    _, model, proc = build_c07()
    BREAKS[sys.argv[1]](model.to(device).eval(), proc, device)


if __name__ == "__main__":
    main()
