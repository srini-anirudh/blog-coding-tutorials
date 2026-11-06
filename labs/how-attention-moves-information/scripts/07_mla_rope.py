"""Multi-head latent attention: absorption, why RoPE breaks it, and the decoupled fix.
Section "Multi-head latent attention". A few seconds on CPU, float64.

The decoupled MLA module is C3's `attnzoo.mla.MultiHeadLatentAttention`. This script reuses it.
"""

import _setup  # noqa: F401

from attnmoves.common import Report
from attnmoves.latent import absorption_check, best_fixed_matrix_error, decoupled_check

rep = Report("07_mla_rope")
rep("Part 1. Fold W_UK into the query (4 heads, dh = 32, latent d_c = 16, 24 tokens)")
for k, v in absorption_check().items():
    rep(f"  {k}: {v:.2e}")
rep("  Without RoPE, absorption is exact. With RoPE on the full key, the naive absorbed scores are wrong")
rep("  by more than the scores' own spread.")

rep("\nPart 2. Could some other fixed matrix stand in for R_delta W_UK?")
rep(f"  best single matrix over offsets 0..127, mean relative error: {best_fixed_matrix_error():.3f}")
rep("  The rotation depends on the key's position, so no fixed matrix fits. Absorbing it exactly would")
rep("  need one matrix per position, which is the same as building the keys.")

rep("\nPart 3. Decoupled RoPE (C3's module): content key from the latent, a small shared RoPE key beside it")
for k, v in decoupled_check().items():
    rep(f"  {k}: {v if isinstance(v, int) else f'{v:.2e}'}")
rep("  With RoPE moved out of the absorbed path, the absorbed and cached forms match the explicit one.")
rep.save()
