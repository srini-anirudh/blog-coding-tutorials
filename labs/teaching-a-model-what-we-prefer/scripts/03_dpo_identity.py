"""Section "The DPO identity, derived slowly" and "What the DPO gradient actually does".

1. Invert the tilt: beta log(pi* / ref) equals r minus one constant, and that constant is beta log Z.
2. Z cancels: Bradley-Terry probabilities from r and from the recovered reward agree.
3. Fit a tabular policy with the DPO loss on Bradley-Terry labels over all pairs. It converges to pi*, the policy
   the reward-model-then-RL route would have produced, without ever fitting r.
4. On a tiny random Llama, autograd's DPO gradient equals -beta * w * (grad log pi_w - grad log pi_l).
5. The weight sigmoid(r_hat_l - r_hat_w) shrinks as pairs resolve.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from prefopt.dpo import SoftmaxPolicy, dpo_loss, dpo_weight, train_dpo  # noqa: E402
from prefopt.lm import pair_logprobs, tiny_llama, toy_pair  # noqa: E402
from prefopt.report import Report  # noqa: E402
from prefopt.tilt import log_partition, tilt  # noqa: E402


def gradient_check(beta: float = 0.1):
    """Compare autograd with the closed-form DPO gradient on one pair, through every parameter of a real model."""
    torch.manual_seed(0)
    policy, ref = tiny_llama(seed=0), tiny_llama(seed=1)  # ref differs from policy so the weight is not 0.5
    batch = toy_pair()
    params = [p for p in policy.parameters() if p.requires_grad]

    lw, ll = pair_logprobs(policy, batch)
    with torch.no_grad():
        rw, rl = pair_logprobs(ref, batch)
    g_auto = torch.autograd.grad(dpo_loss(lw, ll, rw, rl, beta), params)

    lw, ll = pair_logprobs(policy, batch)
    g_w = torch.autograd.grad(lw.sum(), params)
    lw2, ll2 = pair_logprobs(policy, batch)
    g_l = torch.autograd.grad(ll2.sum(), params)
    w = dpo_weight(lw.detach(), ll.detach(), rw, rl, beta)
    g_formula = [-beta * w * (a - b) for a, b in zip(g_w, g_l)]

    flat_a = torch.cat([g.flatten() for g in g_auto])
    flat_f = torch.cat([g.flatten() for g in g_formula])
    rel = ((flat_a - flat_f).norm() / flat_a.norm()).item()
    return rel, w.item(), sum(p.numel() for p in params)


def main():
    rep = Report("03_dpo_identity")
    g = torch.Generator().manual_seed(0)
    k, beta = 10, 0.5
    ref = torch.softmax(torch.randn(k, generator=g, dtype=torch.float64), -1)
    r = torch.randn(k, generator=g, dtype=torch.float64)
    p_star = tilt(ref, r, beta)

    recovered = beta * (p_star.log() - ref.log())
    offset = r - recovered
    blz = beta * log_partition(ref, r, beta).item()
    rep.line("1. Recovered reward beta log(pi*/ref) vs true r")
    rep.line(f"   r - recovered: min {offset.min().item():.12f}, max {offset.max().item():.12f}; beta log Z = {blz:.12f}")

    i, j = torch.triu_indices(k, k, 1)
    p_true = torch.sigmoid(r[i] - r[j])
    p_rec = torch.sigmoid(recovered[i] - recovered[j])
    rep.line(f"2. Over all {i.numel()} pairs, largest |BT(r) - BT(recovered)| = {(p_true - p_rec).abs().max().item():.2e}")

    rep.line()
    rep.line("3. DPO on soft Bradley-Terry labels over all pairs, tabular policy started at ref")
    winners = torch.cat([i, j])
    losers = torch.cat([j, i])
    weight = torch.cat([p_true, 1 - p_true])  # each ordered pair weighted by how often annotators pick it
    policy = SoftmaxPolicy(ref.log())
    logs = train_dpo(policy, winners, losers, beta, steps=3000, lr=0.05, weight=weight, log_every=500)
    p_dpo = policy.log_probs().exp().detach()
    tv = 0.5 * (p_dpo - p_star).abs().sum().item()
    tv_ref = 0.5 * (ref - p_star).abs().sum().item()
    rep.line(f"   total variation: ref to pi* = {tv_ref:.4f}; DPO policy to pi* = {tv:.2e}")
    rep.table(["item", "ref", "pi* (tilt of true r)", "DPO policy"],
              [[n, ref[n].item(), p_star[n].item(), p_dpo[n].item()] for n in range(k)])

    rep.line()
    rel, w, n_params = gradient_check()
    rep.line(f"4. Gradient check on a 2-layer random Llama ({n_params:,} parameters), one pair, beta = 0.1:")
    rep.line(f"   weight sigmoid(r_hat_l - r_hat_w) = {w:.4f}; ||autograd - formula|| / ||autograd|| = {rel:.2e}")

    rep.line()
    rep.line("5. The self-annealing weight during the hard-label fit below (winner = higher true reward, every pair)")
    hard_w, hard_l = torch.where(r[i] > r[j], i, j), torch.where(r[i] > r[j], j, i)
    hard = SoftmaxPolicy(ref.log())
    logs = train_dpo(hard, hard_w, hard_l, beta, steps=400, lr=0.05, log_every=50)
    rep.table(["step", "loss", "mean weight", "mean log pi(y_w)", "mean log pi(y_l)"],
              [[x["step"], x["loss"], x["weight"], x["logp_w"], x["logp_l"]] for x in logs])
    top = int(r.argmax())
    p_hard = hard.log_probs().exp()[top].item()
    rep.line("   With deterministic labels the loss keeps falling as the margins grow without bound. The mean")
    rep.line("   winner log-prob falls too, because most items win some pairs and lose others while mass drains")
    rep.line(f"   into the top item: pi(item {top}) = {p_hard:.4f} at step 400, against {p_star[top].item():.4f} under pi*.")

    rep.data = {"offset_min": offset.min().item(), "offset_max": offset.max().item(), "beta_logZ": blz,
                "tv_ref_to_pistar": tv_ref, "tv_dpo_to_pistar": tv, "grad_rel_err": rel, "grad_weight": w,
                "annealing": logs, "hard_label_top_mass": p_hard, "pistar_top_mass": p_star[top].item()}
    rep.save()


if __name__ == "__main__":
    main()
