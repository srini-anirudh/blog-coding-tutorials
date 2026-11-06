"""Order-of-magnitude training FLOPs for open VLMs, from what each paper reports (the post's Table 2).

Two methods, as in the post:
- tokens reported:        6 * N * D with N the total parameters (overstates frozen stages)
- hardware time reported: chip-hours * peak dense bf16 FLOP/s * 40% utilisation
"""

from __future__ import annotations

PEAK_BF16 = {"A100": 312e12, "H100": 989e12, "TPUv4": 275e12, "TPUv5e": 197e12}
UTILISATION = 0.40


def hardware_flops(chips: float, hours: float, chip: str, utilisation: float = UTILISATION) -> float:
    return chips * hours * 3600.0 * PEAK_BF16[chip] * utilisation


def token_flops(params: float, tokens: float) -> float:
    return 6.0 * params * tokens


# (model, what the paper reports, method, FLOPs, the post's rounded estimate)
LADDER = [
    ("Prismatic 7B", "under 9 h on 8 A100", "hardware", hardware_flops(8, 9, "A100"), 3e19),
    ("LLaVA-1.5-13B", "about 1 day on 8 A100", "hardware", hardware_flops(8, 24, "A100"), 9e19),
    ("PaliGemma 3B", "350B + 90B tokens", "6ND", token_flops(3e9, 440e9), 8e21),
    ("Molmo-72B", "4.2K + 8.3K H100-hours", "hardware", hardware_flops(1, 12.5e3, "H100"), 2e22),
    ("DeepSeek-VL-7B", "5 days on 512 A100", "hardware", hardware_flops(512, 5 * 24, "A100"), 3e22),
    ("InternVL2.5-78B", "about 120B tokens", "6ND", token_flops(78e9, 120e9), 6e22),
    ("PaLI-17B", "1,024 TPUv4 x 7 days", "hardware", hardware_flops(1024, 7 * 24, "TPUv4"), 7e22),
    ("MM1-30B", "about 400B tokens", "6ND", token_flops(30e9, 400e9), 7e22),
    ("Flamingo-80B", "1,536 TPUv4 x 15 days", "hardware", hardware_flops(1536, 15 * 24, "TPUv4"), 2e23),
    ("Qwen2-VL-72B", "about 1.4T tokens", "6ND", token_flops(72e9, 1.4e12), 6e23),
    ("Qwen2.5-VL-72B", "4.1T tokens", "6ND", token_flops(72e9, 4.1e12), 2e24),
    ("Chameleon-34B", "9.2T tokens", "6ND", token_flops(34e9, 9.2e12), 2e24),
    ("Qwen2.5-72B (text LLM)", "18T tokens", "6ND", token_flops(72e9, 18e12), 8e24),
]

# Cross-checks the post makes in its comment column.
CROSS_CHECKS = {
    "PaliGemma 3B, <3 days on 256 TPUv5e": hardware_flops(256, 3 * 24, "TPUv5e"),
    "Chameleon-34B, 4.28M A100-hours": hardware_flops(1, 4.28e6, "A100"),
}


def order_of_magnitude_match(value: float, rounded: float, tolerance: float = 0.5) -> bool:
    """True if `value` rounds to `rounded` at one significant figure, within +-tolerance of that digit."""
    return abs(value - rounded) <= tolerance * 10 ** int(f"{rounded:e}".split("e")[1])
