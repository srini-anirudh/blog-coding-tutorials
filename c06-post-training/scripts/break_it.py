"""Run the classic post-training bugs on purpose and print what each one looks like.

    python scripts/break_it.py
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from posttrain.chat import encode_chat, encode_prompt, load_tokenizer  # noqa: E402
from posttrain.dpo import collate_pairs, dpo_batch_loss, dpo_loss, synthetic_pairs, tokenize_pair  # noqa: E402
from posttrain.generation import generate, score_completions  # noqa: E402
from posttrain.grpo import GRPOLossConfig, group_advantages, grpo_loss  # noqa: E402
from posttrain.logprobs import per_token_logprobs, sequence_logprobs  # noqa: E402
from posttrain.models import frozen_copy, load_policy  # noqa: E402
from posttrain.packing import collate_packed, pack_examples  # noqa: E402
from posttrain.sft import build_examples  # noqa: E402
from posttrain.tasks import exact_reward, sample_problems, to_messages  # noqa: E402
from posttrain.utils import pick_device, polite_threads, save_json, synchronize  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def grad_norm(model) -> float:
    return float(sum(p.grad.pow(2).sum() for p in model.parameters() if p.grad is not None) ** 0.5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(ROOT / "checkpoints" / "sft-full"))
    args = ap.parse_args()
    polite_threads()
    torch.manual_seed(0)
    device = pick_device()
    tok = load_tokenizer()
    policy = load_policy(args.base, device).eval()
    problems = sample_problems(8, seed=7)
    out: dict[str, dict] = {"device": {"name": str(device)}}

    # 1. Pad token = end-of-turn token. A collator that masks pad ids also masks the stop token.
    ex = encode_chat(tok, to_messages(problems[0]))
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    labels = torch.tensor(ex.labels)
    buggy = labels.masked_fill(torch.tensor(ex.input_ids) == im_end, -100)  # labels[input_ids == pad_id] = -100
    good, bad = int((labels == im_end).sum()), int((buggy == im_end).sum())
    out["pad_is_eos"] = {"stop_targets_correct": good, "stop_targets_with_bug": bad}
    print(f"[pad = <|im_end|>]  stop-token targets in one conversation: correct {good}, with the bug {bad}")

    # 2. Pack with no block mask: conversation B reads conversation A.
    a, b = build_examples(tok, problems[:2])
    packed = collate_packed(pack_examples([a, b], 256), 256, tok.pad_token_id)
    n_a = (packed["seq_ids"][0] == 0).sum().item()
    n_b = (packed["seq_ids"][0] == 1).sum().item()
    with torch.no_grad():
        ids = packed["input_ids"].to(device)
        alone = policy(input_ids=ids[:, n_a : n_a + n_b], use_cache=False).logits[0]
        safe = policy(input_ids=ids, attention_mask=packed["attention_mask"].to(device),
                      position_ids=packed["position_ids"].to(device), use_cache=False).logits[0, n_a : n_a + n_b]
        naive = policy(input_ids=ids, use_cache=False).logits[0, n_a : n_a + n_b]
    leak_safe, leak_naive = (safe - alone).abs().max().item(), (naive - alone).abs().max().item()
    out["packing"] = {"max_logit_diff_block_mask": leak_safe, "max_logit_diff_no_mask": leak_naive}
    print(f"[packing]  max |logit diff| vs running alone: block mask {leak_safe:.5f}, no mask {leak_naive:.3f}")

    # 3. DPO reference that is the policy itself: the margin is 0 by construction, and so is the gradient.
    pairs = synthetic_pairs(problems)
    batch = {k: v.to(device) for k, v in collate_pairs([tokenize_pair(tok, p) for p in pairs], tok.pad_token_id, 32).items()}
    policy.train()
    # With a real frozen copy the first step also has margin 0, so nudge the policy first to get a live gradient.
    ref = frozen_copy(policy)
    opt = torch.optim.SGD(policy.parameters(), lr=1e-4)
    dpo_batch_loss(policy, ref, batch)[0].backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    loss_ok, m_ok = dpo_batch_loss(policy, ref, batch)
    loss_ok.backward()
    gn_ok = grad_norm(policy)
    policy.zero_grad(set_to_none=True)
    loss_same, m_same = dpo_batch_loss(policy, policy, batch)
    loss_same.backward()
    gn_same = grad_norm(policy)
    policy.zero_grad(set_to_none=True)
    out["reference_is_policy"] = {"loss_frozen_copy": loss_ok.item(), "grad_norm_frozen_copy": gn_ok,
                                  "loss_bug": loss_same.item(), "grad_norm_bug": gn_same}
    print(f"[ref = policy]  frozen copy: loss {loss_ok.item():.4f}, grad norm {gn_ok:.2f}."
          f"  bug: loss {loss_same.item():.4f}, margin {m_same['margin'].abs().max().item():.1e}, grad norm {gn_same:.2f}")

    # 4. Reference with gradients enabled: the same shape of bug, but the graph doubles and the sign is wrong.
    unfrozen = copy.deepcopy(ref)
    for p in unfrozen.parameters():
        p.requires_grad_(True)
    args_t = (batch["input_ids"], batch["attention_mask"], batch["completion_mask"])
    pc, pr = sequence_logprobs(policy, *args_t).chunk(2)
    rc, rr = sequence_logprobs(unfrozen, *args_t).chunk(2)
    dpo_loss(pc, pr, rc, rr)[0].backward()
    ref_grad = grad_norm(unfrozen)
    policy.zero_grad(set_to_none=True)
    del unfrozen
    out["reference_with_grad"] = {"reference_grad_norm": ref_grad}
    print(f"[ref has grad]  gradient norm that lands on the reference: {ref_grad:.2f} (must be 0)")

    # 5. Mean instead of sum: the margin shrinks by about the reply length.
    with torch.no_grad():
        _, m_sum = dpo_batch_loss(policy, ref, batch, average=False)
        _, m_mean = dpo_batch_loss(policy, ref, batch, average=True)
    lengths = batch["completion_mask"].sum(-1).float().mean().item()
    loss_sum = dpo_batch_loss(policy, ref, batch, average=False)[0]
    loss_sum.backward()
    gn_sum = grad_norm(policy)
    policy.zero_grad(set_to_none=True)
    loss_mean = dpo_batch_loss(policy, ref, batch, average=True)[0]
    loss_mean.backward()
    gn_mean = grad_norm(policy)
    policy.zero_grad(set_to_none=True)
    out["sum_vs_mean"] = {"mean_reply_tokens": lengths, "margin_sum": m_sum["margin"].abs().mean().item(),
                          "margin_mean": m_mean["margin"].abs().mean().item(), "grad_sum": gn_sum, "grad_mean": gn_mean}
    print(f"[sum vs mean]  reply ~{lengths:.0f} tokens. |margin| sum {m_sum['margin'].abs().mean().item():.4f}, "
          f"mean {m_mean['margin'].abs().mean().item():.4f}. grad norm sum {gn_sum:.2f}, mean {gn_mean:.2f}")

    # 6. Off by one: score tokens with the wrong shift, so the header newline counts and the stop token does not.
    policy.eval()
    with torch.no_grad():
        logps = per_token_logprobs(policy, *args_t[:2])
        cm = batch["completion_mask"]
        right = (logps * cm[:, 1:]).sum(-1)
        wrong = (logps * cm[:, :-1]).sum(-1)
    out["off_by_one"] = {"right": right.mean().item(), "wrong": wrong.mean().item()}
    print(f"[off by one]  mean log pi(y|x): right {right.mean().item():.3f}, wrong shift {wrong.mean().item():.3f}")

    # 7. GRPO: recompute old log-probs inside the inner loop. The ratio is then 1 forever and clipping never fires.
    prompts = [encode_prompt(tok, to_messages(p, with_answer=False)) for p in problems[:4] for _ in range(4)]
    answers = [p.answer for p in problems[:4] for _ in range(4)]
    gen = generate(policy, tok, prompts)
    adv = group_advantages(torch.tensor([exact_reward(t, a) for t, a in zip(gen.texts, answers)], device=device), 4)
    tensors = (gen.prompt_ids, gen.prompt_mask, gen.completion_ids, gen.completion_mask)
    cfg = GRPOLossConfig(beta=0.0)
    trial = copy.deepcopy(policy).train()
    opt = torch.optim.AdamW(trial.parameters(), lr=3e-5, weight_decay=0.0)
    with torch.no_grad():
        fixed_old = score_completions(trial, *tensors)
    rows = []
    for _ in range(4):
        logps = score_completions(trial, *tensors)
        _, good_stats = grpo_loss(logps, fixed_old, adv, gen.completion_mask, cfg=cfg)
        loss, bad_stats = grpo_loss(logps, logps.detach(), adv, gen.completion_mask, cfg=cfg)
        loss_real, _ = grpo_loss(logps, fixed_old, adv, gen.completion_mask, cfg=cfg)
        loss_real.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        rows.append({"ratio_fixed": good_stats["ratio_mean"], "clip_fixed": good_stats["clip_frac"],
                     "ratio_recomputed": bad_stats["ratio_mean"], "clip_recomputed": bad_stats["clip_frac"]})
    out["old_logprobs"] = {"inner_steps": rows, "reward_mean": float(adv.abs().mean())}
    print("[old log-probs]  inner step: ratio/clip with fixed old | ratio/clip when recomputed")
    for i, r in enumerate(rows):
        print(f"    {i}: {r['ratio_fixed']:.3f} / {r['clip_fixed']:.3f}   |   "
              f"{r['ratio_recomputed']:.3f} / {r['clip_recomputed']:.3f}")
    del trial

    # 8. MPS compiles kernels per shape, so the first batch of a new length is far slower than the second.
    if device.type == "mps":
        times = {}
        for L in (61, 61, 61):
            x = torch.randint(3, 1000, (4, L), device=device)
            t = time.time()
            with torch.no_grad():
                policy(input_ids=x, use_cache=False)
            synchronize(device)
            times.setdefault(L, []).append(round(time.time() - t, 3))
        out["mps_new_shape"] = {"sec_first_second_third": times[61]}
        print(f"[MPS shapes]  forward at a new length, three times: {times[61]} sec")

    save_json(out, "break_it.json")


if __name__ == "__main__":
    main()
