"""memlab: small, runnable versions of the mechanisms in "Memory Is More Than Context".

Each module is one idea from the post:

- allocation   choosing context items under a token budget when utility is not additive
- position     where a fact sits in the prompt, measured on a real small model
- staleness    the 1/(lambda t) law for append-only stores, and precision at fixed k
- poisoning    the experience-following fixed point, closed form against simulation
- scoring      the weighted retrieval score as naive-Bayes log-odds, and thresholding
- bitemporal   append, replace and supersede, two clocks, and deletion that chases copies
- ladder       the five-condition ablation ladder with a real small model
"""

from .hardware import hardware_label, header

__all__ = ["hardware_label", "header"]
