"""Print every mask in attnzoo.masks. Rows are queries, columns are keys."""

import _setup  # noqa: F401
import torch

from attnzoo import masks as M


def show(title: str, mask: torch.Tensor) -> None:
    allowed = int(mask.sum())
    print(f"{title}  ({allowed}/{mask.numel()} pairs allowed)")
    print(M.render(mask), end="\n\n")


n = 8
show("bidirectional", M.bidirectional_mask(n))
show("causal", M.causal_mask(n))
attn = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]])
show("padding (last two tokens are pad)", M.padding_mask(attn).expand(1, 1, n, n)[0, 0])
show("causal + padding", M.combine(M.causal_mask(n), M.padding_mask(attn))[0, 0])
show("prefix-LM (prefix = 3)", M.prefix_lm_mask(n, 3))
show("packed documents [0,0,0,1,1,2,2,2]", M.document_mask(torch.tensor([[0, 0, 0, 1, 1, 2, 2, 2]]))[0, 0])
show("sliding window (W = 3)", M.sliding_window_mask(n, 3))
show("local + global (W = 2, token 0 global)", M.local_global_mask(n, 2, [0]))
show("block-sparse (4 blocks of 2)", M.expand_block_layout(M.block_sparse_layout(4), 2))
show("multimodal [t, i, i, i, t, t, t, t]", M.multimodal_mask(torch.tensor([[0, 1, 1, 1, 0, 0, 0, 0]]))[0, 0])
show("decoding: 2 queries, 8 keys (bottom-right aligned)", M.causal_mask(2, 8))
