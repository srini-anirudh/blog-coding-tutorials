"""thinktok: small, runnable versions of the mechanisms in "Thinking in Tokens".

Modules
  parity   a chain of thought buys serial depth: parity with and without a running-parity trace
  cost     the bill for K thinking tokens: FLOPs, KV-cache bytes and the decode-latency floor
  vote     self-consistency as an estimator of the answer marginal
  budget   thinking control as string surgery on the decode loop (Qwen3 prefill and interrupt, s1 "Wait")
  gsm8k    answer extraction and scoring for GSM8K
  hf       a thin Hugging Face adapter for the budget controller (GPU scripts only)
"""

from . import _paths  # noqa: F401  (puts the C4 tutorial's tinyllm on sys.path)
