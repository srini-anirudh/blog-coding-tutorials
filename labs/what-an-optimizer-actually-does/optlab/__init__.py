"""optlab: the optimizers from "What an Optimizer Actually Does", small enough to read and run."""

from .newton_schulz import CUBIC_COEFFS, MUON_COEFFS, newton_schulz, ns_scalar, shampoo_direction, spectral_steepest
from .optimizers import SGD, Adafactor, Adagrad, Adam, AdamW8bit, Hybrid, Lion, Muon, RMSProp, factored_v, rms
from .quant import BlockQuantized, code_table

__all__ = [
    "CUBIC_COEFFS", "MUON_COEFFS", "SGD", "Adafactor", "Adagrad", "Adam", "AdamW8bit", "BlockQuantized", "Hybrid",
    "Lion", "Muon", "RMSProp", "code_table", "factored_v", "newton_schulz", "ns_scalar", "rms",
    "shampoo_direction", "spectral_steepest",
]
