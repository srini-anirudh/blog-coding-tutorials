"""posenc: the mechanisms of "How Models Know Where They Are", small enough to read and run.

sinusoidal  the frequency code and its shift-by-rotation property
rope        RoPE in both pairing conventions, the decay envelope, frequency-band bookkeeping
extend      Position Interpolation, NTK-aware base scaling and YaRN
tinylm      a tiny decoder with a switchable positional scheme (none, learned, sinusoidal, RoPE, ALiBi)
tasks       synthetic data: position probing and induction-style copying
mrope       Qwen2-VL style 3D position ids, chunked and interleaved frequency layouts
fourier     random Fourier features for coordinate MLPs and SAM-style prompt encoding

The half-split RoPE rotation, ALiBi and axial 2D RoPE are imported from the C3 tutorial (`attnzoo`).
"""
