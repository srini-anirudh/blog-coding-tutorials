TREE = """c04-llm/
├── README.md                   # every command, results table
├── conftest.py
├── configs/                    # tiny, smollm2-135m, llama3-8b as JSON
├── results/                    # saved outputs of every run in this post
├── tinyllm/
│   ├── config.py               # ModelConfig and presets
│   ├── tokenizer.py            # byte tokenizer
│   ├── norm.py                 # RMSNorm
│   ├── rope.py                 # RotaryEmbedding, apply_rotary
│   ├── attention_naive.py      # reference attention, masks, reshapes
│   ├── attention_sdpa.py       # the fused kernel
│   ├── attention.py            # GroupedQueryAttention
│   ├── kv_cache.py             # KVCache
│   ├── mlp.py                  # SwiGLU
│   ├── moe.py                  # optional mixture of experts
│   ├── block.py                # DecoderBlock
│   ├── model.py                # TinyLLM
│   ├── loss.py                 # shifted cross-entropy
│   ├── init.py                 # init_weights
│   ├── params.py               # closed-form counts
│   ├── sampling.py             # temperature, top-k, top-p
│   ├── generation.py           # generate, cached and uncached
│   ├── checkpoint.py           # native save and load
│   ├── hf.py                   # LlamaForCausalLM conversion
│   └── utils.py                # device, seed, timer
├── scripts/
│   ├── walkthrough.py          # one experiment per step
│   ├── inspect_shapes.py       # every tensor in one block
│   ├── param_count.py
│   ├── compare_hf.py           # the trusted-implementation check
│   ├── train_sanity.py         # a few minutes on Tiny Shakespeare
│   ├── generate.py
│   ├── break_it.py             # every intentional failure
│   ├── moe_demo.py
│   └── compare_heads.py        # MHA vs GQA vs MQA
├── benchmarks/
│   ├── bench_attention.py
│   └── bench_generation.py
└── tests/
    ├── c04_helpers.py
    ├── test_components.py      # norm, RoPE, attention, SwiGLU, loss
    ├── test_model.py           # shapes, causality, parameter counts
    ├── test_cache.py           # full forward == cached forward
    ├── test_hf_parity.py       # logits match LlamaForCausalLM
    ├── test_matches_c3.py      # matches the C3 attention library
    ├── test_sampling.py
    ├── test_checkpoint.py      # round trip and resume
    ├── test_precision.py       # bf16
    └── test_moe.py"""
FILLS = {}
