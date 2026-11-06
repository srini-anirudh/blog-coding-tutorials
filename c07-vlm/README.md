# C7: Build a Vision-Language Model From Scratch

We join a pretrained vision encoder ([SigLIP-base, `google/siglip-base-patch16-224`](https://huggingface.co/google/siglip-base-patch16-224))
to a pretrained language model ([SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)).
The two have never seen each other. The package `tinyvlm` builds every piece between them by hand: the image processor,
the projector, the multimodal sequence, labels, position ids, attention masks, variable resolution, multi-image
input, generation and Hugging Face save and load. Then we train only the projector on synthetic shape captions and
check that the model really uses the image.

Part of [blog-coding-tutorials](https://github.com/srini-anirudh/blog-coding-tutorials). The post is
"Build a Vision-Language Model From Scratch" (`build-a-vlm-from-scratch`). C8 "Train a Real VLM" trains this model.

## What you build

```text
pixel_values  n x 3 x 224 x 224                        n crops in the batch
  -> SigLIP vision encoder   n x 196 x 768             N_v patch tokens, d_v
  -> pixel shuffle (s = 2)   n x 49 x 3072             2 x 2 patches folded into one token
  -> projector               n x 49 x 576              Linear, or Linear -> GELU -> Linear
input_ids     B x N  -> token embeddings B x N x 576, <image> placeholders overwritten by the image features
  -> SmolLM2                 logits B x N x 49216
```

| Module | What it does |
|---|---|
| `tinyvlm/image_processing.py` | RGB, resize, normalise. Fixed resize, aspect-ratio padding, and tiling with a thumbnail. Matches HF's SigLIP processor exactly. |
| `tinyvlm/projector.py` | `LinearProjector`, `MLPProjector`, `pixel_shuffle` (equal to the Idefics3 / SmolVLM implementation). |
| `tinyvlm/chat.py` | ChatML rendering, split into supervised and unsupervised pieces. |
| `tinyvlm/processing.py` | `TinyVLMProcessor`: expands each `<image>` into `num_crops x image_seq_len` placeholders, builds labels, pads, tracks image boundaries (`image_index`). |
| `tinyvlm/merge.py` | Scatter image features into the embeddings. `merge_naive` (a loop) and `merge_masked_scatter` (fast). |
| `tinyvlm/positions.py` | 1D position ids that respect left padding, and a teaching version of Qwen2-VL's 3D (M-RoPE) positions. |
| `tinyvlm/masks.py` | Explicit causal + padding masks, optionally bidirectional inside an image (prefix-LM). |
| `tinyvlm/configuration.py`, `modeling.py` | `VLMConfig` and `TinyVLM`, a `PreTrainedModel` with `.vision`, `.projector`, `.lm`. |
| `tinyvlm/generation.py` | Greedy decoding by hand, with and without a KV cache, checked against `lm.generate`. |
| `tinyvlm/build.py` | `build_tinyvlm()`: the two pretrained halves plus a fresh projector and a matching processor. |
| `tinyvlm/data.py` | Synthetic coloured shapes with exact captions, and a parser that scores captions. |
| `tinyvlm/testing.py` | `tiny_vlm()`: the same classes at toy size, with no downloads. Used by the tests here and in C8. |

The contract that C8 uses: `VLMConfig`, `TinyVLM(config)` with submodules `.vision`, `.projector`, `.lm`,
`forward(input_ids, pixel_values, attention_mask=None, labels=None)` returning `.loss` and `.logits`, `generate(...)`,
and `TinyVLMProcessor` with `build_example`, `collate` and `__call__`.

## Setup

From the repository root, with the repo's virtual environment (`pip install -e ".[dev]"`). No extra packages.
The first run downloads SigLIP-base and SmolLM2-135M-Instruct, about 1 GB. If your default Hugging Face cache is not
writable, set `HF_HOME` to a folder you own.

## Run it

Every command runs from the repository root. Each step script prints what the post shows, and its saved output is
in `results/steps/`.

| Command | What it shows |
|---|---|
| `python c07-vlm/scripts/step01_image_processor.py` | Our processor against HF's SigLIP processor: max difference 0. |
| `python c07-vlm/scripts/step02_vision_features.py` | Encoder output `2 x 196 x 768`, statistics, patch similarity. |
| `python c07-vlm/scripts/step03_projector.py` | Linear and MLP projectors, pixel shuffle, scale against token embeddings, parameter counts. |
| `python c07-vlm/scripts/step04_scatter.py` | Placeholders in `input_ids`, the merge, naive loop against `masked_scatter`. |
| `python c07-vlm/scripts/step05_sequence.py` | The full sequence token by token, with labels (`-100` everywhere but the reply). |
| `python c07-vlm/scripts/step07_positions_masks.py` | 1D positions under left padding, ASCII attention masks, 3D positions. |
| `python c07-vlm/scripts/step09_variable_resolution.py` | Image tokens per picture size for resize, pad and tile. |
| `python c07-vlm/scripts/step10_multi_image.py` | Two pictures in one prompt, and where each one's tokens are. |
| `python c07-vlm/scripts/step11_generate.py` | Manual greedy decoding equals `lm.generate`; the cost of no cache. |
| `python c07-vlm/scripts/step12_save_load.py` | `save_pretrained`, reload through `AutoModel`, identical logits. |
| `python c07-vlm/scripts/break_it.py` | Four failures: token-count mismatch, missing marker, wrong loss mask, excessive resolution. |
| `python c07-vlm/scripts/train_alignment.py --projector mlp --steps 300` | Stage-1 alignment: train only the projector. Writes `results/<run>.json`; weights go to `runs/` (not in git). |
| `python c07-vlm/scripts/evaluate_alignment.py --run align-mlp-s2` | Five conditions (correct, blind, wrong normalisation, right padding, reversed crops), scale, out-of-distribution pictures. |
| `python c07-vlm/benchmarks/bench_processor.py` | Preprocessing time, ours against HF. |
| `python c07-vlm/benchmarks/bench_forward.py --device cpu` | Forward latency, encoder share and KV-cache size against image tokens. `--device cuda` adds peak GPU memory. |
| `python c07-vlm/benchmarks/bench_decode.py` | Greedy decoding with and without the KV cache, against prompt length. |
| `bash c07-vlm/scripts/run_all.sh [steps\|align\|all]` | Regenerates everything in `results/`. Set `PY` to choose the Python. |

`train_alignment.py` options: `--projector linear|mlp`, `--pixel-shuffle 1|2`, `--steps`, `--batch-size`, `--lr`,
`--eval-every`, `--bad-labels` (Break it: take the loss on every token), `--device`, `--out`.

## Tests

```bash
python -m pytest c07-vlm -m "not slow and not gpu and not network"   # 53 tests, tiny random model, about 20 s
python -m pytest c07-vlm -m "network"                                # 7 tests against the real checkpoints
```

The offline tests cover shapes, pixel values against HF's SigLIP processor, `pixel_shuffle` against Idefics3's, the
naive merge against `masked_scatter`, gradients that reach only the projector, labels, padding, multi-image
boundaries, tiling with different crop counts in one batch, causality around the image, full forward against the
cached forward, manual decoding against `generate` (with left padding), explicit 4D masks against HF's default, and
save and load round trips. The `network` tests check the real model: SigLIP features equal the full SigLIP model,
text-only logits equal SmolLM2's, the chat template matches SmolLM2's, and the full model survives save and load.
That last one is also marked `slow`.

## Results

All numbers below were measured on an **AMD EPYC 7R13 CPU, 8 threads**, float32, on a shared machine with other
load. Timings are the best of several runs. The one GPU benchmark is in its own section below. The raw outputs are
in `results/`.

### Alignment: train only the projector

Vision encoder and language model are frozen. 16 pictures per step, AdamW, peak learning rate 1e-3, 20 warmup steps,
cosine decay. "All correct" means count, colour and shape all parse correctly from a greedy caption on 96 held-out
pictures (32 for the checkpoints during training).

| Run | Projector | Trainable params | Steps | Train time | All correct, before | Step 50 | Step 100 | After |
|---|---|---|---|---|---|---|---|---|
| `align-mlp-s2` | MLP, s = 2, 49 tokens | 2,102,400 | 300 | 869 s | 0.00 | 0.59 | 0.97 | **1.00** |
| `align-linear-s2` | Linear, s = 2, 49 tokens | 1,770,048 | 300 | 833 s | 0.00 | 0.94 | 0.97 | **1.00** |
| `align-mlp-s1` | MLP, s = 1, 196 tokens | 775,296 | 200 | 1253 s | 0.00 | 0.91 | 1.00 | **1.00** |
| `align-mlp-s2-badlabels` | MLP, loss on every token | 2,102,400 | 300 | 1060 s | 0.00 | 0.09 | 0.28 | **0.95** |

Before training the model ignores the picture ("The image is a photograph of a woman wearing a white dress").
After 300 steps it writes "Three orange squares." The task is easy, so the linear projector reaches the same final
score. With the loss on every token, the loss stays near 8.5 (it is dominated by the unpredictable prompt and image
tokens) and learning is much slower.

### Is the image really used? (`eval-align-mlp-s2.json`)

| Condition | All correct | Exact caption, padded samples | Exact caption, unpadded samples |
|---|---|---|---|
| Correct | 1.00 | 32 / 32 | 64 / 64 |
| Blind: all pixels zero | 0.00 | 0 / 32 | 0 / 64 |
| Wrong normalisation: [0, 1] instead of [-1, 1] | 0.82 | 23 / 32 | 56 / 64 |
| Right padding at generation | 0.70 | 1 / 32 | 64 / 64 |
| Crops reversed inside the batch (16 pictures) | 0.00, but 1.00 against the reversed targets | | |

Right padding only breaks the samples that have padding: the shortest question is one token shorter than the others.
Reversed crops raise no error at all; every caption describes another picture. The trained projector's output has a
standard deviation of 4.83, against 0.136 for the language model's token embeddings.

### What a picture costs (`bench_forward_cpu.txt`, `bench_decode_cpu.txt`)

One prompt, 14 text tokens. Seven crops is what tiling gives a 640 x 480 picture.

| Crops | Pixel shuffle | Image tokens | Sequence | Encode ms | Forward ms | KV cache MB | 20 tokens, cached s | 20 tokens, no cache s |
|---|---|---|---|---|---|---|---|---|
| 1 | 2 | 49 | 63 | 86 | 172 | 2.9 | 1.49 | 2.13 |
| 7 | 2 | 343 | 357 | 438 | 705 | 16.5 | 2.07 | 6.58 |
| 1 | 1 | 196 | 210 | 97 | 288 | 9.7 | 1.41 | 4.01 |
| 7 | 1 | 1372 | 1386 | 414 | 1369 | 63.9 | 2.79 | 21.90 |

Forward ms includes the encoder. Without a cache, every new token re-runs the whole prompt, so the cost grows with
the number of image tokens: 1.4 times slower at 63 tokens, 7.9 times at 1,386. Both decoding columns include one
encoder pass and produce the same tokens.

### Other measured numbers

| What | Result | File |
|---|---|---|
| Our pixels against HF's SigLIP processor | max difference 0.00e+00 | `steps/step01_image_processor.txt` |
| Preprocess a 1920 x 1080 picture, ours against HF | 12.75 ms against 25.70 ms | `bench_processor.txt` |
| Merge, naive loop against `masked_scatter` (8 x 63 tokens) | 3.764 ms against 0.461 ms | `steps/step04_scatter.txt` |
| 20 new tokens: `lm.generate`, manual with cache, manual without | 1.17 s, 1.24 s, 2.42 s (same tokens) | `steps/step11_generate.txt` |
| Saved checkpoint | `model.safetensors` 889.86 MB, logits identical after reload | `steps/step12_save_load.txt` |
| Parameters | vision 85,797,120, projector 2,102,400, LM 134,551,872 | `steps/step03_projector.txt` |
| Wrong loss mask on an untrained projector | 4.401 on the reply, 11.458 on every token | `steps/break_it.txt` |
| A 4032 x 3024 photo, 48 tiles, no pixel shuffle | 9,618 tokens (limit 8,192), 18.17 s forward, 443.2 MB KV cache; nothing raises | `steps/break_it.txt` |

`run_all.sh steps` took about 9 minutes and `run_all.sh align` about 74 minutes on this CPU.

### GPU

Nothing here needs a GPU. One benchmark was also run on an **NVIDIA H100 80GB HBM3** (torch 2.15.0+cu132, CUDA 13.2),
float32, best of 5, with the GPU to itself: `python c07-vlm/benchmarks/bench_forward.py --device cuda`
(`results/bench_forward_cuda.txt`).

| Crops | Pixel shuffle | Image tokens | Sequence | Encode ms | Forward ms | KV cache MB | Peak GPU memory MB |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 49 | 63 | 6 | 33 | 2.9 | 980 |
| 7 | 2 | 343 | 357 | 8 | 35 | 16.5 | 1131 |
| 1 | 1 | 196 | 210 | 6 | 33 | 9.7 | 1083 |
| 7 | 1 | 1372 | 1386 | 8 | 36 | 63.9 | 1636 |

At batch size 1 the GPU is nearly flat: 33 to 36 ms whatever the token count. That suggests these small kernels are
bound by launch overhead, not compute, though the benchmark does not measure that directly. On the CPU the same
forward pass costs 8 times more at 1,386 tokens than at 63. Peak memory includes the 0.89 GB of float32 weights. The
other GPU columns were not measured: everything else in this README is CPU only.

## Notes

- Our chat template leaves out the default system message that SmolLM2's own template adds when there is none. A
  network test pins this difference. `<|im_start|>` is the tokenizer's BOS token, so every sequence still starts with BOS.
- Aspect-ratio padding uses the mean colour, which normalises to 0.0039, not exactly 0 (127.5 rounds to 128).
- `image_index` records which picture each placeholder belongs to. The model itself only needs the order:
  features fill placeholders in row-major order over the batch.

## Exercises

1. Make the image block bidirectional during training (`masks.py` already builds the mask) and compare the alignment curve.
2. Read the encoder's second-to-last layer (`vision_feature_layer=-2`, as LLaVA does) and retrain.
3. Wire `position_ids_3d` into the language model's rotary embedding and compare text positions for a 7-crop picture.
4. Add a resampler projector (a few learned queries with cross-attention) and compare it with pixel shuffle at the same token count.
5. Extend `data.py` with mixed colours or counts above three, and measure where the aligned projector fails.
