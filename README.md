# blog-coding-tutorials

Code for the coding tutorials on [The Backward Pass](https://srini-anirudh.github.io/blogs/?series=code).

> **Status: work in progress.** The code is being finished and the blog posts are not published yet, so the post links below do not work yet. See [HANDOFF.md](HANDOFF.md).

> Build the thing. Inspect every tensor. Train it. Break it. Profile it. Scale it.

Each folder is one tutorial: a complete, tested build that the matching blog post walks through step by step.
The theory posts explain why things work. These tutorials implement them, look at the tensors, and measure them.

| # | Tutorial | Folder | Blog post |
|---|---|---|---|
| C1 | PyTorch From First Principles | [c01-pytorch](c01-pytorch) | [post](https://srini-anirudh.github.io/blogs/pytorch-from-first-principles/) |
| C2 | The Hugging Face Stack | [c02-hf-stack](c02-hf-stack) | [post](https://srini-anirudh.github.io/blogs/the-hugging-face-stack/) |
| C3 | Build Attention From Scratch | [c03-attention](c03-attention) | [post](https://srini-anirudh.github.io/blogs/build-attention-from-scratch/) |
| C4 | Build a Modern LLM From Scratch | [c04-llm](c04-llm) | [post](https://srini-anirudh.github.io/blogs/build-a-modern-llm-from-scratch/) |
| C5 | Train an LLM From Scratch | [c05-pretraining](c05-pretraining) | [post](https://srini-anirudh.github.io/blogs/train-an-llm-from-scratch/) |
| C6 | Post-Train an LLM | [c06-post-training](c06-post-training) | [post](https://srini-anirudh.github.io/blogs/post-train-an-llm/) |
| C7 | Build a Vision-Language Model From Scratch | [c07-vlm](c07-vlm) | [post](https://srini-anirudh.github.io/blogs/build-a-vlm-from-scratch/) |
| C8 | Train a Real VLM | [c08-vlm-training](c08-vlm-training) | [post](https://srini-anirudh.github.io/blogs/train-a-real-vlm/) |
| C9 | Post-Train a VLM | [c09-vlm-post-training](c09-vlm-post-training) | [post](https://srini-anirudh.github.io/blogs/post-train-a-vlm/) |
| C10 | Build an Agent Harness From Scratch | [c10-agent-harness](c10-agent-harness) | [post](https://srini-anirudh.github.io/blogs/build-an-agent-harness-from-scratch/) |
| C11 | Add Memory and Context Engineering to the Agent | [c11-agent-memory](c11-agent-memory) | [post](https://srini-anirudh.github.io/blogs/add-memory-to-the-agent/) |
| C12 | Train an Agent With Reinforcement Learning | [c12-agent-rl](c12-agent-rl) | [post](https://srini-anirudh.github.io/blogs/train-an-agent-with-rl/) |

The full plan for the series is in [docs/series-plan.md](docs/series-plan.md).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Everything runs on a laptop (CPU or Apple MPS). Sections that need a CUDA GPU are marked in each folder's README
and in the tests (`@pytest.mark.gpu`).

## Tests

```bash
pytest c03-attention -m "not slow and not gpu"   # one tutorial
pytest -m "not slow and not gpu"                 # everything
```

Markers: `slow` (long runs), `gpu` (needs CUDA), `network` (downloads from the Hugging Face Hub).

## Layout of a tutorial folder

```text
cNN-name/
├── README.md      what you build, how to run it, measured results
├── <package>/     the library: readable reference code first, fast versions beside it
├── scripts/       demos, training and evaluation entry points
├── benchmarks/    measurements that print tables
├── configs/       small configs
└── tests/         shapes, equivalence against trusted implementations, gradients, resume
```

Each folder stands on its own. Later tutorials reuse earlier ones where the post says so.

## License

MIT
