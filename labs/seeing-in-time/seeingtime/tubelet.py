"""From 2D patches to space-time tubes, and the 2 x 2 merge that follows.

FramePatchEmbed: an image encoder's patch embedding, applied to each frame on its own.
TubeletEmbed: a 3D convolution whose kernel spans `depth` frames as well as p x p pixels (ViViT, Qwen2-VL).
merge_2x2: fold each 2 x 2 block of neighbouring tokens into one token (Qwen2-VL's patch merger input).
"""

from __future__ import annotations

import torch
from torch import nn


class FramePatchEmbed(nn.Module):
    def __init__(self, patch: int, dim: int, channels: int = 3):
        super().__init__()
        self.proj = nn.Conv2d(channels, dim, patch, stride=patch, bias=False)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """video (T, C, H, W) -> tokens (T, H/p, W/p, dim). Frame t never sees frame t+1."""
        return self.proj(video).permute(0, 2, 3, 1)


class TubeletEmbed(nn.Module):
    def __init__(self, patch: int, depth: int, dim: int, channels: int = 3):
        super().__init__()
        self.depth = depth
        self.proj = nn.Conv3d(channels, dim, (depth, patch, patch), stride=(depth, patch, patch), bias=False)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """video (T, C, H, W) -> tokens (T/depth, H/p, W/p, dim). One token per tube."""
        if video.shape[0] % self.depth:
            raise ValueError("frame count must be a multiple of the tubelet depth")
        x = self.proj(video.permute(1, 0, 2, 3)[None])  # 1, dim, T', H', W'
        return x[0].permute(1, 2, 3, 0)

    def embed_image(self, image: torch.Tensor) -> torch.Tensor:
        """Qwen2-VL feeds a still image as `depth` identical frames, so one encoder serves both."""
        return self(image[None].expand(self.depth, *image.shape))

    def as_frame_conv(self) -> nn.Conv2d:
        """The 2D convolution this tubelet becomes on identical frames: the kernel summed over time."""
        conv = nn.Conv2d(self.proj.in_channels, self.proj.out_channels, self.proj.kernel_size[1:],
                         stride=self.proj.stride[1:], bias=False)
        conv.weight.data = self.proj.weight.data.sum(dim=2)
        return conv


def merge_2x2(tokens: torch.Tensor) -> torch.Tensor:
    """(..., h, w, d) -> (..., h/2, w/2, 4d): concatenate each 2 x 2 block of tokens into one."""
    *lead, h, w, d = tokens.shape
    x = tokens.reshape(*lead, h // 2, 2, w // 2, 2, d)
    x = x.movedim(-4, -3)  # (..., h/2, w/2, 2, 2, d)
    return x.reshape(*lead, h // 2, w // 2, 4 * d)


def token_count(tokens: torch.Tensor) -> int:
    return int(tokens[..., 0].numel())
