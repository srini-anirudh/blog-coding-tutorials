"""Sections 6 and 7: the identical middle, and inside one block.

Build a text Transformer and a ViT from the same BlockConfig. Diff their parameters: only the boundaries differ.
Copy the text model's blocks into the ViT and feed both the same N x d sequence: the outputs match.
Then the block itself: the MLP's share of parameters, SwiGLU at 8/3 width, and gradient norms per layer at
initialisation for post-norm and pre-norm. Runs offline.
"""

import _setup
import torch

from twomod import Block, BlockConfig, TextTransformer, ViT
from twomod.front import count_params

_setup.header()
torch.manual_seed(0)

_setup.section("Diff the two models")
cfg = BlockConfig(dim=128, n_heads=4)
text = TextTransformer(vocab_size=5000, cfg=cfg, depth=4, max_len=256)
vit = ViT(cfg, depth=4, image_size=32, patch=4, num_classes=10)
t_shapes = {k: tuple(v.shape) for k, v in text.state_dict().items()}
v_shapes = {k: tuple(v.shape) for k, v in vit.state_dict().items()}
shared = [k for k in t_shapes if k in v_shapes and t_shapes[k] == v_shapes[k]]
middle = sum(text.state_dict()[k].numel() for k in shared)
print(f"tensors with the same name and shape in both models: {len(shared)}, {middle:,} parameters "
      f"(all under blocks.* and the final norm: {all(k.startswith(('blocks.', 'norm.')) for k in shared)})")
print("only in the text model:", {k: v for k, v in t_shapes.items() if k not in shared})
print("only in the ViT:       ", {k: v for k, v in v_shapes.items() if k not in shared})
print(f"text model: {count_params(text):,} parameters, {middle / count_params(text):.1%} in the shared middle")
print(f"ViT:        {count_params(vit):,} parameters, {middle / count_params(vit):.1%} in the shared middle")

_setup.section("Transplant the middle")
vit.blocks.load_state_dict(text.blocks.state_dict())
vit.norm.load_state_dict(text.norm.state_dict())
seq = torch.randn(2, 65, 128)  # any N x d sequence: the blocks cannot tell where it came from
with torch.no_grad():
    a, b = seq.clone(), seq.clone()
    for blk in text.blocks:
        a = blk(a, None)  # text blocks, run with the ViT's (absent) mask
    for blk in vit.blocks:
        b = blk(b, None)
print(f"text model's blocks vs ViT's blocks after the copy, same input, max |difference|: "
      f"{(a - b).abs().max().item():.1e}")

_setup.section("Where a block's parameters live")
for d in (768, 4096):
    blk = Block(BlockConfig(dim=d, n_heads=d // 64))
    mlp, attn = count_params(blk.mlp), count_params(blk.attn)
    print(f"d={d}: GELU block {count_params(blk):,} params; MLP {mlp / count_params(blk):.1%}, "
          f"attention {attn / count_params(blk):.1%}")
d = 4096
gelu = Block(BlockConfig(dim=d, n_heads=32, bias=False))
swiglu = Block(BlockConfig(dim=d, n_heads=32, bias=False, mlp="swiglu", norm="rms", mlp_hidden=int(8 * d / 3)))
print(f"d={d}, no bias: GELU MLP (4d = {4 * d}) {count_params(gelu.mlp):,} params; "
      f"SwiGLU MLP (8d/3 = {int(8 * d / 3)}) {count_params(swiglu.mlp):,} params")

_setup.section("Gradient norm per layer at initialisation: post-norm vs pre-norm")
depth, seeds = 12, 5
print(f"{depth}-layer text Transformer, d=256, untied head, random tokens, mean over {seeds} seeds; "
      "norm of the gradient of each block's MLP output matrix")


def xiong_init(model: TextTransformer) -> None:
    """Xavier-normal matrices and unit-scale embeddings, close to the setting Xiong et al. (2020) analyse."""
    for name, p in model.named_parameters():
        if p.dim() == 2 and (name.startswith("blocks.") or name.startswith("head.")):
            torch.nn.init.xavier_normal_(p)
    torch.nn.init.normal_(model.embed.weight, std=1.0)
    torch.nn.init.zeros_(model.pos)


print("layer                " + " ".join(f"{i:>6}" for i in range(1, depth + 1)))
last = {}
for init in ("PyTorch default", "Xavier, unit embed"):
    for prenorm in (False, True):
        acc = torch.zeros(depth)
        for seed in range(seeds):
            torch.manual_seed(seed)
            model = TextTransformer(2000, BlockConfig(dim=256, n_heads=8, prenorm=prenorm), depth=depth,
                                    max_len=128, tie=False)
            if init.startswith("Xavier"):
                xiong_init(model)
            model.loss(torch.randint(0, 2000, (8, 128))).backward()
            acc += torch.tensor([blk.mlp.down.weight.grad.norm().item() for blk in model.blocks])
        g = acc / seeds
        last[(init, prenorm)] = g[-1].item()
        name = f"{init.split(',')[0].split()[0]} {'pre' if prenorm else 'post'}-norm"
        print(f"{name:<20} " + " ".join(f"{v:6.3f}" for v in g.tolist()))
for init in ("PyTorch default", "Xavier, unit embed"):
    print(f"{init}: last-layer gradient, post-norm / pre-norm = {last[(init, False)] / last[(init, True)]:.2f} "
          f"(sqrt(depth) = {depth ** 0.5:.2f})")
