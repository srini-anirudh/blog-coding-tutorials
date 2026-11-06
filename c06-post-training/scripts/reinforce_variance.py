"""Stage 4: measure how noisy each policy-gradient estimator is on the same model and task.

Each estimator gets the same budget (`samples` rollouts per gradient estimate) and we draw `trials` independent
estimates of the gradient of the last transformer block. Two independent estimates of the same gradient should
point the same way, so low pairwise cosine means high variance.

    python scripts/reinforce_variance.py --trials 12 --samples 32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from posttrain.chat import encode_prompt, load_tokenizer  # noqa: E402
from posttrain.generation import generate, score_completions  # noqa: E402
from posttrain.models import load_policy  # noqa: E402
from posttrain.reinforce import estimator_weights, flat_grad, gradient_noise, reinforce_loss  # noqa: E402
from posttrain.tasks import exact_reward, sample_problems, to_messages  # noqa: E402
from posttrain.utils import pick_device, polite_threads, save_json  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ESTIMATORS = {"none": 1, "batch": 1, "group": 4, "group_std": 4}  # estimator -> samples per prompt


def one_estimate(model, tok, problems, group_size, estimator, params, device):
    prompts = [encode_prompt(tok, to_messages(p, with_answer=False)) for p in problems]
    flat = [q for q in prompts for _ in range(group_size)]
    answers = [p.answer for p in problems for _ in range(group_size)]
    model.eval()
    gen = generate(model, tok, flat)
    rewards = torch.tensor([exact_reward(t, a) for t, a in zip(gen.texts, answers)], device=device)
    logps = score_completions(model, gen.prompt_ids, gen.prompt_mask, gen.completion_ids, gen.completion_mask)
    loss = reinforce_loss(logps, gen.completion_mask, estimator_weights(rewards, group_size, estimator))
    model.zero_grad(set_to_none=True)
    loss.backward()
    return flat_grad(params).detach().cpu(), rewards.cpu()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(ROOT / "checkpoints" / "sft-full"))
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--samples", type=int, default=32)
    args = ap.parse_args()

    polite_threads()
    device = pick_device()
    tok = load_tokenizer()
    model = load_policy(args.base, device)
    params = list(model.model.layers[-1].parameters())
    pool = sample_problems(args.samples * args.trials, seed=4)
    print(f"device={device}, gradient dimension {sum(p.numel() for p in params):,}, "
          f"{args.samples} rollouts per estimate, {args.trials} estimates each")
    print(f"{'estimator':<12}{'prompts x G':>13}{'mean reward':>13}{'grad norm':>11}{'pairwise cos':>14}{'SNR':>9}")
    results = {}
    for name, G in ESTIMATORS.items():
        torch.manual_seed(0)
        P = args.samples // G
        grads, rewards = [], []
        for t in range(args.trials):
            g, r = one_estimate(model, tok, pool[t * P : (t + 1) * P], G, name, params, device)
            grads.append(g)
            rewards.append(r)
        noise = gradient_noise(torch.stack(grads))
        mean_r = torch.cat(rewards).mean().item()
        results[name] = {**noise, "mean_reward": mean_r, "prompts": P, "group_size": G,
                         "per_trial_reward": [round(r.mean().item(), 4) for r in rewards]}
        print(f"{name:<12}{f'{P} x {G}':>13}{mean_r:>13.3f}{noise['mean_norm']:>11.2f}"
              f"{noise['pairwise_cos']:>14.3f}{noise['snr']:>9.3f}", flush=True)
    save_json({"device": str(device), "trials": args.trials, "samples": args.samples, "results": results},
              "reinforce_variance.json")


if __name__ == "__main__":
    main()
