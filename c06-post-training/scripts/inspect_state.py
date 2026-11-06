"""Print the tensors behind one DPO pair and one GRPO group, and export them for the figures.

    python scripts/inspect_state.py
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from posttrain.chat import encode_prompt, load_tokenizer  # noqa: E402
from posttrain.dpo import collate_pairs, synthetic_pairs, tokenize_pair  # noqa: E402
from posttrain.generation import generate, score_completions  # noqa: E402
from posttrain.grpo import group_advantages  # noqa: E402
from posttrain.logprobs import per_token_logprobs  # noqa: E402
from posttrain.models import load_policy  # noqa: E402
from posttrain.tasks import exact_reward, make_problem, sample_problems, to_messages  # noqa: E402
from posttrain.utils import pick_device, polite_threads, save_json  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(ROOT / "checkpoints" / "sft-full"))
    ap.add_argument("--group-size", type=int, default=8)
    args = ap.parse_args()
    polite_threads()
    torch.manual_seed(3)
    device = pick_device()
    tok = load_tokenizer()
    policy = load_policy(args.base, device).eval()

    # One DPO pair, token by token.
    pair = synthetic_pairs(sample_problems(1, seed=8, kinds=("add",)))[0]
    ex = tokenize_pair(tok, pair)
    batch = {k: v.to(device) for k, v in collate_pairs([ex], tok.pad_token_id).items()}
    with torch.no_grad():
        logps = per_token_logprobs(policy, batch["input_ids"], batch["attention_mask"])
    P = len(ex["prompt_ids"])
    print(f"prompt {P} tokens, chosen {len(ex['chosen_ids'])} tokens, rejected {len(ex['rejected_ids'])} tokens")
    dpo_rows = {}
    for row, key in enumerate(("chosen_ids", "rejected_ids")):
        rows = []
        for t, tid in enumerate(ex[key]):
            lp = logps[row, P + t - 1].item()  # logits at position P + t - 1 score the token at P + t
            rows.append({"token": tok.decode([tid]), "id": tid, "logp": round(lp, 3)})
        dpo_rows[key] = rows
        print(key, f"sum log-prob {sum(r['logp'] for r in rows):.3f}")
        print("  ", " ".join(f"{r['token']!r}:{r['logp']:.2f}" for r in rows))

    # One GRPO group: G samples of one prompt, exact rewards, advantages.
    problem = make_problem(random.Random(21), "chain")
    prompt = encode_prompt(tok, to_messages(problem, with_answer=False))
    print(f"\n{problem.question}  (truth {problem.answer})")
    gen = generate(policy, tok, [prompt] * args.group_size)
    rewards = torch.tensor([exact_reward(t, problem.answer) for t in gen.texts])
    adv = group_advantages(rewards, args.group_size)
    with torch.no_grad():
        token_logps = score_completions(policy, gen.prompt_ids, gen.prompt_mask, gen.completion_ids, gen.completion_mask)
    mask = gen.completion_mask.to(token_logps.dtype)
    seq = (token_logps * mask).sum(-1).tolist()
    n_tokens = gen.completion_mask.sum(-1).tolist()
    group = []
    for i in range(args.group_size):
        group.append({"text": gen.texts[i], "reward": rewards[i].item(), "advantage": round(adv[i].item(), 3),
                      "logp": round(seq[i], 2), "tokens": n_tokens[i]})
        print(f"{i}  r={rewards[i].item():.0f}  A={adv[i].item():+.3f}  logp={seq[i]:8.2f}  n={n_tokens[i]:>2}  {gen.texts[i]!r}")

    shapes = {
        "prompt_ids": list(gen.prompt_ids.shape), "completion_ids": list(gen.completion_ids.shape),
        "completion_mask": list(gen.completion_mask.shape), "token_logps": list(token_logps.shape),
    }
    print(shapes)
    save_json({"device": str(device), "dpo": {"prompt_tokens": P, **dpo_rows},
               "grpo": {"question": problem.question, "truth": problem.answer, "group": group, "shapes": shapes}},
              "inspect_state.json")


if __name__ == "__main__":
    main()
