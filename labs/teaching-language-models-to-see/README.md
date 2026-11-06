# Lab: Teaching Language Models to See

A runnable companion to the post
[Teaching Language Models to See: How Vision-Language Models Are Pretrained](https://srini-anirudh.github.io/blogs/teaching-language-models-to-see/).
The post asks what it takes to make a pretrained vision encoder and a pretrained language model work as one model.
This lab takes the post's central claims and makes each one happen in small, readable code: the two contrastive
losses, the gap between a random connector and the language model, Flamingo's zero-initialised gate, the cost of each
freezing pattern, the visual-token arithmetic, where each loss sends its learning signal, and whether an alignment
stage protects the language model.

It reuses the model from tutorial C7 ([`c07-vlm`](../../c07-vlm)): SigLIP-base joined to SmolLM2-135M-Instruct by a
2-layer MLP connector, trained on C7's synthetic shape captions ("Three orange squares."). C8
([`c08-vlm-training`](../../c08-vlm-training)) already builds staged training, data mixtures, text replay, tiling,
packing, grounding tokens and LoRA. This lab links to C8 for those, and imports C8's perplexity check.

## Setup

From the repository root, with the repo's environment (`pip install -e ".[dev]"`). No extra packages. The first run
downloads SigLIP-base (about 800 MB), SmolLM2-135M-Instruct (about 270 MB), Qwen2-VL-2B-Instruct's image-processor
config (a few kB; `s5` only) and WikiText-2 (`s7` only). Set `HF_HOME` if your default cache is not writable.

## Run everything

```bash
bash labs/teaching-language-models-to-see/scripts/run_all.sh        # or run the scripts one by one:
python labs/teaching-language-models-to-see/scripts/s1_contrastive.py
python labs/teaching-language-models-to-see/scripts/s2_connector_gap.py      # trains the connector, saves it to runs/
python labs/teaching-language-models-to-see/scripts/s3_gated_xattn.py
python labs/teaching-language-models-to-see/scripts/s4_freeze_cost.py --batch-size 2 --repeats 5
python labs/teaching-language-models-to-see/scripts/s4_freeze_cost.py --device cuda --batch-size 8 --repeats 10   # needs CUDA
python labs/teaching-language-models-to-see/scripts/s5_token_budget.py
python labs/teaching-language-models-to-see/scripts/s6_learning_signal.py    # needs s2's connector
python labs/teaching-language-models-to-see/scripts/s7_alignment_stage.py    # needs s2's connector

# Tests: 21 offline tests on C7's tiny random model, about 20 s.
python -m pytest labs/teaching-language-models-to-see -m "not slow and not gpu and not network"
# 4 network tests: smart_resize against Qwen2-VL's processor, and zero gates on the real SmolLM2 (also slow).
python -m pytest labs/teaching-language-models-to-see -m "network"
```

Measured run times on the CPU below, under load from other jobs: s1 about 22 min (almost all of it the batch-size
sweep; `--quick` takes about 1 min), s2 about 20 min, s3 about 8 min, s4 about 2 min, s5 under 1 min, s6 about 4 min,
s7 about 10 min. In a 2-step trial on a quieter moment of the same CPU a connector step took about 1.5 s, so s2 and s3
should take nearer 5 min each on an idle machine. That estimate was not measured end to end.

Each script prints its results and writes `results/<script>.txt` and `.json`. Both start with the hardware and the
command. The connector checkpoint goes to `runs/`, which is not in git.

## What is where

| Post section | Script | What it shows |
|---|---|---|
| Learning vision from language (Figure 1) | `s1_contrastive.py` | CLIP's softmax loss and SigLIP's sigmoid loss. The sigmoid loss splits into blocks with no error; the softmax does not. The modality gap at random initialisation and after training. A toy batch-size sweep. |
| Why connecting is not enough; The gap between the two models (Figure 2) | `s2_connector_gap.py` | A fresh connector between real SigLIP and real SmolLM2. Location, scale and content of the image tokens, measured at six points while only the connector trains. |
| Flamingo; Learning rates, warm-up and initialization | `s3_gated_xattn.py` | Zero-initialised tanh-gated cross-attention leaves the language model's logits exactly unchanged; a random MLP prefix does not. The gates during training. |
| Which components are trained; Compute and memory (Figure 4) | `s4_freeze_cost.py` | Step time and bytes held for the four freezing patterns, against the rule of thumb. |
| What a visual token costs; Resolution curricula; How each type becomes a training sequence (Figures 5, 7, 8) | `s5_token_budget.py` | Every visual-token count in the post, recomputed and checked against Qwen2-VL's real image processor. Figure 5's seven data types as real token sequences, with the share that carries loss. |
| Training objectives (Figure 9) | `s6_learning_signal.py` | Gradients of three objectives on SigLIP's 14 x 14 patch grid: which patches receive the learning signal. |
| Why freeze anything; the Nuance on the alignment stage | `s7_alignment_stage.py` | Unfreeze the language model with a random connector, or after s2's alignment stage, and watch its text perplexity. |
| Forgetting; Mixing the data | C8 | Text replay against no replay, temperature mixtures: [`c08-vlm-training`](../../c08-vlm-training), `scripts/train.py` with `configs/stage2_mixture.yaml`. |

| Module | Contents |
|---|---|
| `seelab/contrastive.py` | `clip_loss`, `siglip_loss`, `siglip_loss_chunked`, `modality_gap`, a toy two-tower model on synthetic pairs |
| `seelab/gap.py` | Nearest embedding rows, norms and centroid cosines for image tokens and for text embeddings |
| `seelab/gated.py` | `GatedCrossAttention` and `FlamingoLite`: gated blocks inserted into a frozen Llama-style model |
| `seelab/costs.py` | The Figure 4 cost rule, and a timer that measures forward + backward + AdamW for a freezing pattern |
| `seelab/tokens.py` | Token counts for squares, InternVL tiling and Qwen2-VL native resolution; loss share; FLOPs and memory rules |
| `seelab/signal.py` | Two-shape scenes and per-patch gradients for the contrastive, next-token and per-patch losses |
| `seelab/vlm.py` | Small wrappers around C7's `build_tinyvlm`, training batches and caption scoring |

## Results

Every number below was measured by the scripts in this folder. CPU numbers are on an **AMD EPYC 7R13, 8 threads**
(torch 2.15.0), on shared machines under heavy load from other jobs, so wall-clock times are only rough. Results
that are not times do not depend on the load.

### 1. Contrastive losses and the modality gap (`s1_contrastive.txt`)

| Check | Result |
|---|---|
| Sigmoid loss, full matrix against 3 x 3 blocks (B = 96) | 1.3506460526 against 1.3506460526, difference 6.7e-16 |
| Softmax loss, full against a softmax inside each block | 0.00423 against 0.00141: each block sees 32 of 96 captions and gets an easier task |

Toy paired data, 4-layer ReLU towers, 1,000 held-out pairs. "Gap" is the distance between the two modality centroids
on the unit sphere. "Cone" is the mean cosine between different vectors of one modality.

| Model | Gap | Cone, images | Cone, captions | Top-1 retrieval |
|---|---|---|---|---|
| Random init, seed 0 | 1.295 | 0.915 | 0.897 | 0.003 |
| Random init, seed 1 | 1.138 | 0.947 | 0.935 | 0.002 |
| Random unit vectors (reference) | 0.043 | 0.000 | 0.000 | |
| Softmax (CLIP) loss, 3,000 steps, B = 256 | 0.332 | 0.045 | 0.056 | 0.923 |
| Sigmoid (SigLIP) loss, 3,000 steps, B = 256 | 1.054 | 0.278 | 0.280 | 0.953 |

Two random towers already put images and captions in two narrow cones, far apart. This is the cone effect that Liang
and colleagues describe. Training widens the cones. In this toy the softmax loss shrinks the gap, while the sigmoid
loss keeps most of it and still retrieves best.

The batch-size sweep: 128,000 training examples per run, top-1 retrieval, mean of 3 seeds.

| Batch size | Softmax (CLIP) | Sigmoid (SigLIP) |
|---|---|---|
| 8 | 0.587 | 0.438 |
| 32 | 0.770 | 0.810 |
| 128 | 0.886 | 0.916 |
| 512 | 0.929 | 0.870 |

Neither loss wins everywhere. This 16-dimensional toy says nothing about the SigLIP paper's claim at real scale;
that was not tested.

### 2. The gap between the two models (`s2_connector_gap.txt`)

Only the 2.1M-parameter MLP connector trains: 150 steps of 8 pictures, AdamW, lr 1e-3. The reference is the language
model's own embedding table: rows have norm 3.20, and a row's nearest other row has cosine 0.780.

| Step | Image-token norm | Norm / text norm | Cosine to nearest embedding row | Captions all correct (32 held out) | Nearest words to the tracked token |
|---|---|---|---|---|---|
| 0 | 10.8 | 3.4x | 0.147 | 0.00 | notice, Earth, ter, van, orne |
| 10 | 33.7 | 10.5x | 0.160 | 0.16 | Visual, ectoral, Six, sight, Pict |
| 25 | 69.7 | 21.8x | 0.165 | 0.38 | tru, Motor, Tri, Rama, bronze |
| 50 | 90.3 | 28.2x | 0.156 | 0.66 | Tri, Flickr, tru, tri, Renewable |
| 100 | 99.0 | 30.9x | 0.160 | 0.94 | Tri, Flickr, riage, tri, bronze |
| 150 | 98.1 | 30.6x | 0.156 | **1.00** | Tri, Flickr, riage, tri, (' |

Before training the model ignores the picture: "The image is a photograph of a woman wearing a white dress". After
150 steps it writes "Three orange squares." The content gap closes completely on this task. The location and scale
gaps do not. The image tokens never come near any word: their best cosine to the embedding table stays at 0.15 to
0.17, against 0.78 between real words. Their norm grows from 3.4 to 31 times a word's. The post's Figure 2 is a
labelled sketch in which aligned patches move next to related words. In this measured run they do not: the language
model learns to read vectors that sit far outside its vocabulary, at a much larger scale. C7 saw the same scale
effect (standard deviation 4.83 against 0.136). Training time: 1,018 s on the CPU above.

### 3. Zero-initialised gates against a random prefix (`s3_gated_xattn.txt`)

Five gated cross-attention blocks (22.2M parameters) before layers 0, 6, 12, 18 and 24 of the frozen SmolLM2. They
cross-attend to the 49 pixel-shuffled SigLIP tokens. Flamingo's Perceiver Resampler is left out. Step 0, 16 held-out
captions:

| Model | Caption NLL | KL from the plain LM | Same top-1 token as the plain LM |
|---|---|---|---|
| Language model, no image | 6.091 | 0 | 1.00 |
| LLaVA-style, 49 tokens from a random MLP connector | 4.892 | 0.635 | 0.52 |
| Flamingo-style, gates at 0 | 6.091 | 0 | 1.00 |

With the gates at zero, the largest logit difference to the plain language model is exactly 0.0. The random prefix
changes the next-token choice at almost half the caption positions. At step 0 only the gates get gradient (1.15 in
total); every other new weight gets exactly 0, because its output is multiplied by tanh(0).

Training only the gated blocks, 150 steps of 8 pictures, AdamW lr 1e-3:

| Step | Loss | Largest tanh(gate) | What block 0 adds / residual stream | Captions all correct |
|---|---|---|---|---|
| 10 | 4.270 | 0.005 | 0.495 | 0.00 |
| 50 | 0.376 | 0.015 | 0.383 | 0.16 |
| 100 | 0.304 | 0.020 | 0.286 | 0.62 |
| 150 | 0.049 | 0.026 | 0.420 | **1.00** |

The model learns the task with every gate below 0.03. A small gate is not a closed gate. The block behind it can grow
its output instead: the first block, which sits on the raw embeddings (norm about 3), adds 0.3 to 0.5 times the
residual stream's norm. The deeper blocks add 0.5% to 2%. "The gate opens only as far as visual information helps"
is true of the product, not of the gate value alone. With the same data, steps and batch size, s2's prefix connector
was at 0.94 by step 100, where the gated blocks were at 0.62; both reach 1.00 at step 150. Training time: 412 s.

### 4. What each freezing pattern costs (`s4_freeze_cost_cpu.txt`, `s4_freeze_cost_cuda.txt`)

C7's model with random weights in the shape of Figure 4's example: SigLIP-base at 384 px (576 patch tokens,
86.1M parameters), a 2-layer MLP connector (0.78M) and SmolLM2-135M (134.5M). 576 image + 100 text tokens per example,
batch 2, float32, median of 5 steps. "Rule" is the post's Figure 4 cost model: 2 FLOPs per parameter per token
forward, 2 more where the backward pass must travel, 2 more for weight gradients in trained parts. Here the backward
pass travels through a part when something before it is trained, so it never enters a frozen vision encoder.

| Pattern (vision, connector, LM) | Trained parameters | Bytes held, measured | Bytes, 16 / 4 rule | Step time, CPU | Step / full, CPU | Rule: step / full | Step / full, GPU |
|---|---|---|---|---|---|---|---|
| Connector only (LLaVA stage 1) | 775,296 | 853 MB | 853 MB | 2.92 s | 0.57 | 0.62 | 0.61 |
| Connector + LM (LLaVA stage 2) | 135,290,304 | 2,393 MB | 2,393 MB | 4.03 s | 0.78 | 0.87 | 0.79 |
| Vision + connector (Qwen-VL stage 1) | 86,864,256 | 1,839 MB | 1,839 MB | 3.85 s | 0.75 | 0.76 | 0.82 |
| Everything | 221,379,264 | 3,378 MB | 3,378 MB | 5.15 s | 1.00 | 1.00 | 1.00 |

On an **NVIDIA H100 80GB HBM3** (torch 2.15.0+cu132, CUDA 13.2, float32, batch 8, median of 10) the full step takes
0.242 s and the connector-only step 0.148 s (`s4_freeze_cost_cuda.txt`).

Training 0.35% of the parameters costs 57% of a full step on the CPU and 61% on the GPU (the rule says 62%). The gradient still has to
travel back through the whole frozen language model to reach the connector. A forward pass alone is 0.30 of a full
step on the CPU and 0.32 on the GPU (the rule says 0.33). Training the language model but not the encoder costs less
than the rule predicts on both (0.78 and 0.79 against 0.87). Memory follows the rule exactly: in float32 a trained parameter holds 16 bytes (weight,
gradient, two AdamW moments) and a frozen one 4. The mixed-precision version of the rule (16 and 2 bytes) is what
gives the post's 14 GB and 112 GB for a 7B model; it was not measured here. Activations are not counted. CPU times
are from a shared machine; an earlier run under heavier load gave meaningless ratios and was discarded.

### 5. Visual tokens and loss shares (`s5_token_budget.txt`)

Every count the post quotes is reproduced: 576 tokens at 336 px with P = 14, 256 at 448 px with a 2 x 2 merge,
PaliGemma's 256 tokens at 224 px, 1,024 at 448 px and 4,096 at 896 px (16 times the 224 px count), Figure 8's 256 and 576
for the squashed page, 1,792 for InternVL tiling (2 x 3 tiles + thumbnail with at most 6 tiles), and 1,656 for
native resolution. Our re-implementation of Qwen2-VL's `smart_resize` gives the same token count as the real
Qwen2-VL image processor for all five sizes we tried, from 224 x 224 (64 tokens) to 4032 x 3024 (15,552 tokens).

Two details the post does not state. With InternVL's default of up to 12 tiles, the same page gets 3 x 4 tiles + a
thumbnail = 3,328 tokens, not 1,792. And the formula in Figure 7's caption rounds the grid up, which gives 36 x 47 =
1,692 tokens for this page. Qwen2-VL rounds each side to the nearest multiple of 28, which gives Figure 8's 36 x 46 =
1,656.

Figure 5 with real tokens. Text is tokenised with SmolLM2's tokenizer; the instruction row goes through C7's
processor, so its chat template and label mask are real. The example texts are written for this lab.

| Data type | Scored / total tokens | Share | Post's Figure 5 |
|---|---|---|---|
| Caption pair (576 image tokens) | 15 / 591 | 2.5% | 2% |
| Interleaved page (two 256-token images) | 138 / 650 | 21.2% | 26% |
| OCR page (1,656 image tokens) | 545 / 2,201 | 24.8% | 19% |
| Grounded caption, box as text | 29 / 605 | 4.8% | 3% |
| Video clip, 4 frames of 196 tokens | 21 / 805 | 2.6% | 2% |
| Instruction example, prompt masked | 24 / 620 | 3.9% | 5% |
| Text-only, packed | 2,048 / 2,048 | 100% | 100% |

The largest multimodal share is 9.8 times the smallest here, against 13 times in the post's illustrative counts.
Cost arithmetic: a 20-token caption next to 576 image tokens is 28.8 to 1, and 96.6% of a 7B model's training FLOPs
for that sample go to image tokens. A 7B language model needs 14 GB frozen and 112 GB trained under the 16-byte rule.

### 6. Where each loss puts its learning signal (`s6_learning_signal.txt`)

32 scenes, each with two different shapes, one per half. The caption names one of them, then the other. Gradients are
taken with respect to SigLIP-base's 14 x 14 patch features. The contrastive loss is SigLIP's own sigmoid loss through
its attention-pooling head; the next-token loss goes through s2's connector and the frozen language model; the patch
loss regresses each patch's pixels with a random linear decoder. A zero-shot check confirms that SigLIP works with our
tokenisation: its best of 18 colour-shape names is one of the scene's two shapes in 75% of scenes (chance 11%).

| Objective | Share on the named shape | Share on the other shape | Share on background | Strongest patch / mean patch |
|---|---|---|---|---|
| Contrastive (SigLIP) | 0.099 | 0.100 | 0.800 | 18.6 |
| Next-token (VLM caption) | 0.025 | 0.025 | 0.950 | 17.8 |
| Patch loss (own pixels) | 0.067 | 0.067 | 0.865 | 1.2 |
| Area of the patches (reference) | 0.070 | 0.070 | 0.860 | |

| Objective | Map cosine, name left vs name right | Named shape gets more per patch | Share of the change that lands on the shapes (area 0.140) |
|---|---|---|---|
| Contrastive (SigLIP) | 0.985 | 0.50 | 0.243 |
| Next-token (VLM caption) | 0.694 | 0.55 | 0.062 |
| Patch loss (own pixels) | 1.000 | 0.50 | no change |

The per-patch loss spreads its signal evenly, as Figure 9 shows. The other two do not follow Figure 9's picture. The
contrastive gradient barely depends on which shape the caption names (map cosine 0.985), and it gives the shapes about
1.4 times their area share. The next-token gradient does depend on the caption (cosine 0.69). But it concentrates on a
few background patches (the strongest patch gets 18 times the average), and only 6% of the change lands on the shapes,
less than their 14% of the area. The named shape gets more gradient than the other one in only 55% of cases. So in this
small model the next-token signal does not go to "the patches those words attended to" in any simple way. Two caveats:
the connector was trained on single-colour scenes, so two-shape scenes are out of its training distribution, and the
2 x 2 pixel shuffle mixes neighbouring patches before the language model sees them.

### 7. Does an alignment stage protect the language model? (`s7_alignment_stage.txt`)

The same joint stage twice: connector and language model both trained (lr 1e-3 and 1e-4, 5 warmup steps, gradient
clip 1.0), vision frozen, 60 steps of 8 pictures. One run starts from a random connector. The other starts from the
connector s2 aligned. Perplexity is the language model's alone, on 64 held-out WikiText-2 passages, with no image.

| Step | Perplexity, no alignment | Perplexity, aligned first | Captions, no alignment | Captions, aligned first |
|---|---|---|---|---|
| 0 | 38.37 | 38.37 | 0.00 | 1.00 |
| 5 | 38.78 | 38.43 | 0.00 | 0.56 |
| 15 | 40.31 | 38.74 | 0.00 | 0.84 |
| 30 | 42.94 | 39.14 | 0.06 | 0.84 |
| 60 | **44.20** | **39.28** | 0.47 | 1.00 |

Without the alignment stage, text perplexity rises 15% in 60 steps. After it, perplexity rises 2.4%. The first
gradient that reaches the language model is also 24 times larger (norm 34.5 against 1.43), because the random
connector's captions are wrong and the loss is high. This matches the post's reason for a connector-only first stage.
Two things the post does not mention show up as well. The aligned model loses accuracy for a few steps once the
language model starts to move (1.00, then 0.56 at step 5), and recovers by step 60. And the comparison is not equal
compute: the aligned run also had s2's 150 connector-only steps. One seed, one learning rate. Prismatic's and Molmo's
finding that the stage can be skipped with clean data and careful warm-up was not tested.

## What was not measured

- No paper result at its own scale was reproduced: not CLIP or SigLIP's batch-size findings, not LLaVA's,
  PaliGemma's, MM1's or Idefics2's ablations. Every training run here is a few hundred steps of a 220M-parameter model
  on synthetic shapes.
- Data-mixture effects (MM1's interleaved, caption and text-only ratios, and forgetting) are not rebuilt here. C8
  measures text replay against no replay on its own synthetic mixture.
- Resolution curricula are only computed, not trained. C8 trains with tiles.
- No interleaved, OCR, grounding or video data is trained on here. Figure 5's sequences are built and counted only.
- Figure 6 is a toy model in the post itself, and is not reproduced.
- Activation memory is not counted in `s4`; only weights, gradients and optimizer state are.
