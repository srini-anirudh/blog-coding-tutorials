"""The undocumented half: what resizing, normalisation, EXIF and alpha do before the first attention head.

Everything here works on synthetic images, so the effects can be measured exactly.
"""

from __future__ import annotations

import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps

# The three normalisation conventions in the post.
NORMS = {
    "OpenAI CLIP": ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    "SigLIP (0.5)": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    "ImageNet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}


def zone_plate(size: int) -> np.ndarray:
    """cos(k r^2): spatial frequency rises linearly from the centre and reaches Nyquist at the edge."""
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    r2 = (x - size / 2) ** 2 + (y - size / 2) ** 2
    return 0.5 + 0.5 * np.cos(np.pi * r2 / size)


def stroke_page(size: int, stroke: int = 3) -> np.ndarray:
    """A page of thin dark lines, like small type: 'stroke' px lines with 'stroke' px gaps."""
    a = np.ones((size, size))
    for y in range(0, size, 2 * stroke):
        a[y: y + stroke] = 0.0
    return a


def to_pil(a: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(np.round(a * 255), 0, 255).astype(np.uint8), "L")


def resize_methods(a: np.ndarray, out: int) -> dict[str, np.ndarray]:
    """Shrink one grey image to out x out with every common resampler. Values in [0, 1]."""
    pil = to_pil(a)
    t = torch.from_numpy(np.asarray(pil, dtype=np.float32) / 255.0)[None, None]
    res = {
        "PIL box (area average, reference)": pil.resize((out, out), Image.Resampling.BOX),
        "PIL bicubic (antialiased)": pil.resize((out, out), Image.Resampling.BICUBIC),
        "PIL nearest": pil.resize((out, out), Image.Resampling.NEAREST),
    }
    res = {k: np.asarray(v, dtype=np.float64) / 255.0 for k, v in res.items()}
    for aa in (True, False):
        y = F.interpolate(t, size=(out, out), mode="bicubic", align_corners=False, antialias=aa)
        res[f"torch bicubic antialias={aa}"] = y[0, 0].clamp(0, 1).double().numpy()
    return res


def aliasing_report(a: np.ndarray, out: int) -> dict[str, dict[str, float]]:
    """RMSE against the area-average reference, and the spurious contrast (std) left in the output."""
    res = resize_methods(a, out)
    ref = res["PIL box (area average, reference)"]
    return {k: {"rmse_vs_box": float(np.sqrt(((v - ref) ** 2).mean())), "output_std": float(v.std())}
            for k, v in res.items()}


def smooth_photo(width: int, height: int, seed: int = 0) -> Image.Image:
    """A synthetic RGB 'photo': smooth blobs plus a few hard edges."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width].astype(np.float64)
    img = np.zeros((height, width, 3))
    for _ in range(24):
        cx, cy, s = rng.uniform(0, width), rng.uniform(0, height), rng.uniform(20, width / 4)
        img += rng.uniform(-1, 1, 3) * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s * s))[..., None]
    img = (img - img.min()) / (img.max() - img.min())
    pil = Image.fromarray((img * 255).astype(np.uint8), "RGB")
    d = ImageDraw.Draw(pil)
    for _ in range(12):
        x0, y0 = rng.integers(0, width - 40), rng.integers(0, height - 40)
        d.rectangle((x0, y0, x0 + rng.integers(4, 40), y0 + rng.integers(4, 40)), fill=tuple(rng.integers(0, 256, 3).tolist()))
    return pil


def library_disagreement(img: Image.Image, size: tuple[int, int]) -> dict[str, dict[str, float]]:
    """Same nominal 'bicubic' resize in PIL and torch. Differences in 8-bit levels against PIL."""
    ref = np.asarray(img.resize(size, Image.Resampling.BICUBIC), dtype=np.float64)
    t = torch.from_numpy(np.asarray(img, dtype=np.float32)).permute(2, 0, 1)[None]
    out = {}
    for aa in (True, False):
        y = F.interpolate(t, size=(size[1], size[0]), mode="bicubic", align_corners=False, antialias=aa)
        y = y.clamp(0, 255).round()[0].permute(1, 2, 0).double().numpy()
        d = np.abs(y - ref)
        out[f"torch antialias={aa}"] = {"mean_abs": float(d.mean()), "max_abs": float(d.max()),
                                        "pixels_off_by_more_than_2": float((d > 2).mean())}
    return out


def normalise(pixels: torch.Tensor, mean, std) -> torch.Tensor:
    """pixels: ... x 3 x H x W in [0, 1]."""
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    return (pixels - m) / s


def norm_mismatch(used: str, expected: str) -> list[tuple[float, float]]:
    """Feeding pixels normalised with `used` to a model trained with `expected` is, per channel,
    x_seen = gain * x_expected + offset. Returns (gain, offset) for R, G, B."""
    mu, su = NORMS[used]
    me, se = NORMS[expected]
    return [(se[c] / su[c], (me[c] - mu[c]) / su[c]) for c in range(3)]


def exif_rotated_jpeg(width: int = 64, height: int = 32, orientation: int = 6) -> bytes:
    """A landscape JPEG whose EXIF says 'rotate 90 degrees clockwise to display', as a phone camera writes it."""
    img = Image.new("RGB", (width, height), (255, 255, 255))
    ImageDraw.Draw(img).rectangle((0, 0, width // 4, height - 1), fill=(255, 0, 0))  # red band on the left
    exif = Image.Exif()
    exif[0x0112] = orientation
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif.tobytes())
    return buf.getvalue()


def load(data: bytes, apply_exif: bool) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    return ImageOps.exif_transpose(img) if apply_exif else img.copy()


def transparent_chart(size: int = 64) -> Image.Image:
    """Black lines on a fully transparent background, as many chart and screenshot PNGs are stored."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for i in range(4, size, 12):
        d.line((0, i, size - 1, size - 1 - i), fill=(0, 0, 0, 255), width=2)
    return img


def flatten(img: Image.Image, background: tuple[int, int, int] | None) -> Image.Image:
    """background=None is the naive img.convert('RGB'), which drops alpha and keeps the hidden RGB."""
    if background is None:
        return img.convert("RGB")
    base = Image.new("RGBA", img.size, background + (255,))
    return Image.alpha_composite(base, img).convert("RGB")


def line_contrast(img: Image.Image, chart: Image.Image) -> float:
    """Mean brightness of background pixels minus mean brightness of line pixels, in [0, 1]."""
    a = np.asarray(img.convert("L"), dtype=np.float64) / 255
    alpha = np.asarray(chart)[..., 3] > 0
    return float(a[~alpha].mean() - a[alpha].mean())
