# Lab: One Transformer, Two Modalities

The post claims that a language model and a Vision Transformer share their whole middle. They differ only at the
boundaries: how the input becomes vectors, how position gets in, which attention entries are allowed, and what the
last hidden state predicts. This lab builds both models around one `Block` class and tests that claim in code.

Post: https://srini-anirudh.github.io/blogs/one-transformer-two-modalities/

What you can see happen:

- A byte-level BPE tokenizer is fit on a corpus, then frozen. Its segmentation depends on the corpus.
- An embedding lookup is a one-hot matmul. Its gradient touches only the rows seen in the batch.
- Patchify is an exactly invertible reshape. The patch projection is a strided `Conv2d`. Its gradient is dense.
- Attention without position sees a set. A ViT with no position table gives the same logits for a patch-shuffled image.
- With the causal mask, the past ignores the future and a KV cache is exact. Without it, neither holds.
- The text model's blocks load into the ViT with a strict `load_state_dict` and give identical outputs.
- On CIFAR-10 subsets, a small CNN and a small ViT get the same step budget. The ViT's learned 1D position table is
  checked for 2D structure.

## Layout

```
labs/one-transformer-two-modalities/
├── twomod/
│   ├── bpe.py         # byte-level BPE: train, encode, decode
│   ├── front.py       # onehot_lookup, patchify / unpatchify, PatchEmbed (and its Conv2d form)
│   ├── positions.py   # sinusoids, 2D sin-cos, table interpolation, neighbour and alignment scores
│   ├── masks.py       # causal, bidirectional, prefix-LM, padding
│   ├── block.py       # Attention (MHA or GQA, KV cache), MLP (GELU or SwiGLU), Block (pre- or post-norm)
│   ├── models.py      # TextTransformer, ViT, SmallCNN
│   ├── train.py       # CIFAR-10 loading, augmentation, a fixed-step training loop
│   └── hw.py          # the hardware line at the top of every result file
├── scripts/           # 01 to 07, one per part of the post, and run_all.sh
├── results/           # the real output of every script
└── tests/
```

RoPE and 2D RoPE are not rebuilt here. Script 03 imports them from C3 (`c03-attention/attnzoo`). C3 also covers
attention variants in depth, C4 builds a full modern LLM, and C7 joins a real vision encoder to a real LLM.

## Setup

From the repository root, with the repo's environment (`pip install -e ".[dev]"`). No extra packages.
Scripts 01 and 02 download wikitext-2, the GPT-2 tokenizer and the SmolLM2-135M-Instruct tokenizer (a few MB).
Script 07 downloads CIFAR-10 (`uoft-cs/cifar10`, about 140 MB). Set `HF_HOME` if your default cache is not writable.

## Run everything

```bash
# Tests. The offline set takes about 40 s on CPU.
python -m pytest labs/one-transformer-two-modalities -m "not slow and not gpu and not network"
python -m pytest labs/one-transformer-two-modalities -m "network"      # runs scripts 01, 02 and 07 --quick

# All scripts, saving output to results/.
bash labs/one-transformer-two-modalities/scripts/run_all.sh

# Or one at a time.
python labs/one-transformer-two-modalities/scripts/01_front_doors.py       # about 2 min on CPU (BPE training)
python labs/one-transformer-two-modalities/scripts/02_sequences.py
python labs/one-transformer-two-modalities/scripts/03_position.py
python labs/one-transformer-two-modalities/scripts/04_masks.py
python labs/one-transformer-two-modalities/scripts/05_identical_middle.py
python labs/one-transformer-two-modalities/scripts/06_inductive_bias.py
python labs/one-transformer-two-modalities/scripts/07_vit_vs_cnn.py --out labs/one-transformer-two-modalities/results/vit_vs_cnn.json
python labs/one-transformer-two-modalities/scripts/07_vit_vs_cnn.py --quick   # 20 steps on CPU, a smoke test
```

Script 07 trains 8 small models for 5,000 steps each (about 8 minutes of training on one H100). It uses a GPU when one is
visible. On CPU the same setting would take hours: use `--quick`, or shorten it with `--steps`, `--batch` and
`--sizes`. A shorter setting is not comparable with the numbers below.

## Post sections and scripts

| Post section | Script | What it shows |
|---|---|---|
| 1 · Tokenizing vs patchifying | `01_front_doors.py` | BPE fit from scratch; exact round trip; segmentation of " Transformers" in our BPE and GPT-2's |
| 2 · Where the vectors come from | `01_front_doors.py` | lookup = one-hot matmul; sparse vs dense gradient; Linear = Conv2d; parameter accounting at full size |
| 3 · Assembling the sequence | `02_sequences.py` | chat-template overhead; padding waste; N = (H/P)^2; [CLS] and registers |
| 4 · Position, the only geometry | `03_position.py`, `07_vit_vs_cnn.py` | permutation equivariance; sinusoid offsets; RoPE and 2D RoPE (from C3); table interpolation; learned table structure |
| 5 · Causal vs bidirectional | `04_masks.py` | future-independence; KV cache exact only when causal; N - 1 targets per pass; prefix-LM mask |
| 6 · The identical middle | `05_identical_middle.py` | diff of the two models' parameters; the middle transplanted |
| 7 · Inside one block | `05_identical_middle.py` | MLP share of a block; SwiGLU at 8d/3; gradient norms at init, post-norm vs pre-norm |
| 8 · The exits | `04_masks.py`, `01_front_doors.py` | per-position loss vs one label; tied head; logits tensor size |
| 9 · Inductive bias and data | `06_inductive_bias.py`, `07_vit_vs_cnn.py` | conv equivariance vs ViT; receptive field; ViT vs CNN on 2.5k, 10k and 50k images |

## Results

Every number names its hardware. Raw output is in `results/`, one file per script.

Tests on AMD EPYC 7R13, 8 threads, torch 2.15.0: 34 passed offline. The 3 `network` tests (also marked `slow`)
run scripts 01, 02 and 07 `--quick`: 3 passed.

### Exact checks (AMD EPYC 7R13, 8 threads, torch 2.15.0)

| Check | Result |
|---|---|
| BPE `decode(encode(x)) == x` on 200k characters of wikitext-2 validation, and on unseen Unicode | true |
| `E[ids]` vs `onehot(ids) @ E` | max difference 0 |
| `unpatchify(patchify(x)) == x` | true, bit for bit |
| Linear on flattened 16 x 16 patches vs `Conv2d(3, 768, 16, stride=16)` | max difference 3.1e-6 |
| One block, rows shuffled in vs out | max difference 2.4e-7 |
| ViT with no position table, image vs patch-shuffled image | max logit difference 4.8e-7 |
| Causal model: logits at 0..19 after rewriting tokens 20..31 | max difference 0 |
| Same weights, no mask | max difference 0.13 |
| KV-cached decoding vs full pass, 16 steps: causal / bidirectional | 3.0e-7 / 0.43 |
| Text model's blocks loaded into the ViT, same input | max difference 0 |
| Conv with circular padding, shifted by 1, 3 or 4 pixels | max difference 0 |
| ViT, no position table, shift by 1 / 4 pixels (patch 4): relative logit change | 0.50 / 1.0e-7 |

### Measured quantities (AMD EPYC 7R13, 8 threads, torch 2.15.0)

| Measurement | Result |
|---|---|
| Our BPE, 1,000 merges on 1M characters of wikitext-2 | 2.65 bytes per token on validation text |
| Pieces for " transformers" / " Transformers": our BPE | 3 / 4 |
| Pieces for " transformers" / " Transformers": GPT-2 | 2 / 1 |
| Embedding rows with a gradient, GPT-2 vocab, one batch of 8 x 512 wikitext tokens | 1,287 of 50,257 = 2.6% |
| Patch-projection entries with a gradient, one batch of 2 images | 100% |
| Llama-2-7B shape with V = 100,000, untied: input embedding share | 409.6 M of 7.30 B = 5.6% |
| ViT-L/16 at 224: patch projection share | 787,456 of 304.3 M = 0.26% |
| bf16 logits vs one hidden state, B = 8, N = 4,096, V = 100,000, d = 4,096 | 6.55 GB vs 0.27 GB |
| SmolLM2 chat template: 10 content tokens become | 42 tokens |
| Padding, lengths 40 and 900 | short row 95.6%, batch 47.8% |
| Padding, random batches of 16 wikitext-2 paragraphs (GPT-2 tokens, cut at 1,024) | 56.1% of the batch |
| Parameters shared by name and shape, d = 128, 4 layers: text model / ViT | 54.1% / 98.0% |
| MLP share of a GELU block, d = 768 and d = 4,096 | 66.6% and 66.7% |
| MLP parameters at d = 4,096, no bias: GELU 4d / SwiGLU 8d/3 | 134.2 M / 134.2 M |
| Last-layer MLP gradient norm at init, post-norm / pre-norm, 12 layers, Xavier init | 2.68 (sqrt(12) = 3.46) |
| Same ratio with PyTorch's default init and a 0.02 embedding | 1.21 |

The post says the 7B-class embedding is about 6% of the model. With the Llama-2-7B body we measured 5.6%; the exact
share depends on the body.

The gradient-norm rows need a note. Xiong et al. (2020) predict that at the last layer the post-norm gradient is
larger than the pre-norm one by a factor that grows like sqrt(depth). With Xavier init we measured 2.68 at depth 12.
Their figure also shows post-norm gradients growing toward the output. We did not see that: our post-norm norms are
flat across layers, and our pre-norm norms shrink toward the output. The ratio depends on the init, as the second row
shows. We did not train post-norm without warmup, so the post's point about warmup is not measured here.

### ViT vs CNN on CIFAR-10 (GPU)

All runs: 5,000 steps of batch 128, AdamW (lr 1e-3, weight decay 0.05, cosine, 5% warmup), random crop and flip,
one seed (0), the same training subset for every model of a given size, test on the 10,000 CIFAR-10 test images.
The ViT has patch 4 (64 tokens), width 128, 4 heads, 6 pre-norm blocks, 1.21 M parameters. The CNN has six 3x3
convolutions with BatchNorm and 1.15 M parameters. With 2,500 images, 5,000 steps is 256 passes over the data, so
both models fit the training set; the test accuracy is what differs.

NVIDIA H100 80GB HBM3, torch 2.15.0, CUDA 13.2 (`results/vit_vs_cnn.json`, `results/vit_vs_cnn.txt`). The job held
the GPU alone, behind the lock.

| Train images | ViT test acc | CNN test acc | CNN minus ViT |
|---|---|---|---|
| 2,500 | 0.495 | 0.728 | 23.3 points |
| 10,000 | 0.659 | 0.835 | 17.6 points |
| 50,000 | 0.759 | 0.881 | 12.2 points |

The gap closes as data grows, but it has not closed at 50,000 images. We did not see the crossover the post
describes, and this lab is far too small to: the paper's crossover came with pretraining on JFT-300M.

Position embedding at 50,000 images (same GPU):

| ViT position embedding | Test acc |
|---|---|
| Learned 1D table over raster order | 0.759 |
| Fixed 2D sin-cos | 0.764 |
| None | 0.680 |

Removing position costs 8 to 9 points. A fixed 2D table is within 0.5 point of the learned 1D one, which is one seed
and inside the noise we would expect. This is consistent with the ViT paper's report that 2D-aware variants gave no
meaningful gain, but it is not a replication.

Did the learned 1D table recover the 2D grid? The neighbour score is the fraction of the 64 positions whose most
similar other row is a grid neighbour. A random table scores 0.047.

| Train images | Neighbour score | Mean cosine at grid distance 1 / 2 / 3 / 4 / 5 / 6 | Alignment after bicubic to 16 x 16 |
|---|---|---|---|
| 2,500 | 0.641 | 0.74 / 0.52 / 0.27 / 0.06 / -0.08 / -0.17 | 0.996 |
| 10,000 | 0.750 | 0.75 / 0.52 / 0.27 / 0.05 / -0.11 / -0.19 | 0.996 |
| 50,000 | 0.875 | 0.74 / 0.50 / 0.24 / 0.02 / -0.12 / -0.19 | 1.000 |

Similarity falls steadily with grid distance in every row, so the table learned a 2D layout from labels alone, and
more data recovers more of it. The post's claim holds at this scale. The alignment column checks that the standard
bicubic resize keeps each new position closest to the old position it covers (script 03 shows reusing raster rows
scores 0.007 instead).

Timing on this GPU: the CNN took 20 to 22 s per run and the ViT 83 to 93 s. The ViT here is written for clarity (plain
matmul attention, no fused kernels), so this is not a statement about the architectures.

## What was not measured

- ViT overtaking ResNets after JFT-300M pretraining (Dosovitskiy et al.), and DeiT's distillation result. Our largest
  run is 50,000 CIFAR-10 images; the paper's scale was not reproduced.
- The original ViT finding that 2D-aware position embeddings give no meaningful gain was made at ImageNet and JFT
  scale. Our one-seed comparison at 50,000 images is a small-scale check, not a reproduction.
- Post-norm training instability without warmup, QK-norm at scale, and DINOv2's high-norm tokens that motivate
  registers. Registers are built (`ViT(registers=4)`) but their effect is not measured.
- Context extension of RoPE (NTK, YaRN, position interpolation) and M-RoPE. See C3 and C7.
- Contrastive, masked-image and self-distillation objectives. Only supervised classification is trained here.
