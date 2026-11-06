"""Where the square root comes from. Section "Scaled dot-product attention". A few seconds on CPU."""

import _setup  # noqa: F401

from attnmoves.common import Report
from attnmoves.scaling import grad_to_queries, shift_invariance, softmax_stats

rep = Report("02_scaling")
rep("q, k with independent N(0, 1) entries, 128 keys, 2000 queries. ln(128) = 4.852 is the uniform entropy.")
D_KS = (16, 32, 64, 128, 256, 512, 2048, 8192)
for scaled in (False, True):
    rep(f"\n{'Scaled: logits q . k / sqrt(d_k)' if scaled else 'Unscaled: logits q . k'}")
    rows = []
    for d_k in D_KS:
        st = softmax_stats(d_k, scaled=scaled)
        rows.append([d_k, st["logit std"], st["entropy (nats)"], st["max weight"], st["rows with max weight > 0.99"],
                     st["|J_softmax|_F"], grad_to_queries(d_k, scaled)])
    rep.table(["d_k", "logit std", "entropy", "mean max weight", "rows > 0.99", "|J|", "|dL/dq|"], rows)
rep("\n|J| is the Frobenius norm of the softmax Jacobian diag(a) - a a^T, averaged over queries.")
rep("|dL/dq| is the gradient reaching the queries for a random upstream gradient on the output.")
rep("It includes a factor |k| ~ sqrt(d_k), so it can grow even while |J| shrinks.")
rep(f"\nOnly differences matter: max |softmax(s) - softmax(s + 123)| = {shift_invariance():.1e}")
rep.save()
