# Lab: Models That See and Draw

A code companion for the post [Models That See and Draw](https://srini-anirudh.github.io/blogs/models-that-see-and-draw/).
The post explains what changes when a vision-language model must also write images. This lab builds the
writing path at toy scale, in plain PyTorch, on CPU. Images are 8 x 8 or 16 x 16, models have under a million
parameters, and every script runs here and writes its output to `results/`.

The reading half of a VLM is built in [C7](../../c07-vlm) and trained in [C8](../../c08-vlm-training). This lab does not rebuild it. It focuses on what the post adds: why regression cannot draw, how an image becomes tokens,
how those tokens are decoded, what guidance does, how one trunk can carry two losses, and what a pixel encoder and a caption encoder each keep.

The paper-scale results in the post (the Janus ablation, Transfusion's compute comparison, GenEval scores,
RealUnify) are not reproduced. See "What was not measured".

## Setup

From the repository root, after `pip install -e ".[dev]"`. No dataset or model download is needed: all images are
synthetic and generated in code. PyTorch uses every core by default, so set `OMP_NUM_THREADS` on a shared machine.

## Layout

```
labs/models-that-see-and-draw/
├── seedraw/                 # the library
│   ├── data.py              # two-mode "cat" images, 16 x 16 shapes dataset, patchify
│   ├── flow.py              # rectified flow: noisy point, velocity target, loss, Euler sampler
│   ├── tokenizers.py        # VQ (straight-through, commitment), low-dim normalised VQ, FSQ, token budget
│   ├── transformer.py       # a tiny transformer that accepts any boolean attention mask
│   ├── priors.py            # raster autoregression, MaskGIT decoding, classifier-free guidance
│   ├── hybrid.py            # Transfusion-style trunk: next-token loss on text, flow loss on image patches
│   ├── readers.py           # pixel, caption and joint encoders; linear and decoder probes; a judge CNN
│   └── common.py            # seeding, hardware line, result files
├── scripts/                 # one script per experiment, listed below
├── tests/                   # 30 pytest tests
└── results/                 # real outputs, each starting with a hardware line and the command
```

## Run everything

```bash
# Tests. 29 run offline in a few seconds; one `slow` test trains two small tokenizers.
python -m pytest labs/models-that-see-and-draw -m "not slow and not gpu and not network"
python -m pytest labs/models-that-see-and-draw -m slow

# Experiments. Each writes results/<name>.txt and .json.
python labs/models-that-see-and-draw/scripts/01_regression_blur.py
python labs/models-that-see-and-draw/scripts/02_token_budget.py
python labs/models-that-see-and-draw/scripts/03_tokenizers.py        # trains 3 tokenizers, caches them in checkpoints/
python labs/models-that-see-and-draw/scripts/04_decoding_orders.py   # trains judge and 2 priors, caches them
python labs/models-that-see-and-draw/scripts/05_guidance.py
python labs/models-that-see-and-draw/scripts/07_loss_balance.py
python labs/models-that-see-and-draw/scripts/08_readers.py

# The hybrid ablation: three independent runs, so they can run in parallel.
for r in A_bidirectional_image B_causal_image C_bidirectional_noise_cap; do
  python labs/models-that-see-and-draw/scripts/06_hybrid.py --runs $r --name 06_hybrid_${r%%_*} &
done; wait
```

Scripts 03 to 06 share cached models in `checkpoints/`, which git ignores. Delete the folder to retrain from scratch.
The machine that wrote these results was heavily loaded (load average above 50 on 192 cores), so wall-clock times
are indicative only. Losses and accuracies do not depend on load.

## Results

All numbers: AMD EPYC 7R13, 8 threads, torch 2.15.0, CPU only. Each is a single run with a fixed seed, so
differences of a few points are within noise. Raw output is in `results/`.

### 1. Regression returns the mean, samplers commit (`01_regression_blur.py`)

One caption, 8 x 8 images, a bright block on the left or the right half. Three generators trained on the same data.

| Generator | Brightest pixel | Samples committed to one side | Left share among committed | MSE to nearest valid image |
|---|---|---|---|---|
| Regression with squared error | 0.36 | 0.0% | none | 0.0550 |
| Rectified flow, 8000 steps, 100 Euler steps | 0.96 | 86.2% | 52.2% | 0.0042 |
| Raster autoregression over binary pixels | 1.00 | 99.1% | 48.8% | 0.0008 |

The regression output differs from the mean training image by at most 0.0101, as the derivation in the post predicts. It puts a half-bright block on both sides. An earlier flow run with 4000 steps and a narrower network committed only 12% of its samples, so the flow sampler needs enough training to find the modes.

### 2. Token budget (`02_token_budget.py`)

Recomputes the post's "count the bits" numbers: 17.2 bits per text token, about 516 bits for a 30-token answer, 6.3 million raw bits for a 512 x 512 RGB image, Chameleon 1,024 tokens and 13,312 bits (473 times fewer), Emu3 4,096 tokens and 61,440 bits (102 times fewer), Janus 576 tokens. All match the post.

### 3. Keeping the codebook alive (`03_tokenizers.py`)

Same conv encoder and decoder (16 x 16 image to 16 tokens), three quantizers, about 1,000 codes each, 2000 steps, 2,000 test images.

| Quantizer | Codes used | Perplexity | Test MSE |
|---|---|---|---|
| VQ, 1,024 codes, dim 64 | 4.5% | 15.1 | 0.00245 |
| VQ, 1,024 codes, dim 8, l2-normalised | 6.4% | 13.4 | 0.00151 |
| FSQ, levels [8, 5, 5, 5] | 77.4% | 81.9 | 0.00089 |

FSQ uses far more of its codebook and reconstructs best. The low-dimension, normalised VQ improved reconstruction but did not fix usage here. LlamaGen reports 97% usage with that change at a much larger scale. This lab does not reproduce that number.

### 4. Two decoding orders (`04_decoding_orders.py`)

Token priors over the 16 FSQ tokens, conditioned on shape class. A separate judge CNN reads the decoded image. The judge reads 99.0% of FSQ reconstructions correctly, which is the ceiling.

| Order | Forward passes | Shape accuracy |
|---|---|---|
| Raster autoregression | 16 | 0.949 |
| MaskGIT, 1 round | 1 | 0.373 |
| MaskGIT, 2 rounds | 2 | 0.449 |
| MaskGIT, 4 rounds | 4 | 0.629 |
| MaskGIT, 8 rounds | 8 | 0.760 |

More rounds help. With 16 tokens, MaskGIT at 8 rounds still trails the raster order, so the saving in passes does not come free here. The same script prints the pass counts of the post's Figure 2 for a 16 x 16 token grid: 256, 12, 9 and 50. That part is arithmetic.

### 5. Classifier-free guidance (`05_guidance.py`)

The identity and the Bayes form of the post's guidance box hold to 2e-6 on twelve logits, and entropy falls from 3.27 to 1.88 bits as s goes from 0 to 5. On the trained raster prior:

| Guidance scale s | Shape accuracy | Distinct token grids | Network evaluations |
|---|---|---|---|
| 0 (unconditional) | 0.232 | 97.7% | 16 |
| 1 | 0.924 | 97.5% | 16 |
| 2 | 0.906 | 91.2% | 32 |
| 3 | 0.732 | 83.2% | 32 |
| 5 | 0.467 | 80.1% | 32 |

Diversity falls and cost doubles, as the post says. Prompt following does not improve: it was already near the ceiling at s = 1, and larger scales lowered it. A plausible reason is that a 1,000-way vocabulary and only 16 tokens leave nothing to gain, while large scales push samples off the data. We did not test that explanation.

### 6. One trunk, two losses (`06_hybrid.py`)

A tiny Transfusion: 16 patches of 4 x 4 pixels, no VAE, 344,474 parameters, L = L_LM + 5 L_flow, 10,000 steps, learning rate 3e-3. Every batch holds both layouts: caption then image (drawing) and image then caption (reading). Drawing is graded by the judge on 400 samples, reading on 1,000 clean test images.

| Run | Draw: shape and side right | Read: shape and side right | Final L_LM | Final L_flow |
|---|---|---|---|---|
| A. Bidirectional attention inside the image | 0.822 | 1.000 | 0.614 | 0.0734 |
| B. Causal attention inside the image | 0.655 | 1.000 | 0.622 | 0.0892 |
| C. As A, read-side noise capped at t = 0.5 | 0.788 | 0.999 | 0.404 | 0.0618 |

Bidirectional attention inside the image helps drawing, in line with the post's claim about Transfusion's ablation. This is one seed, and the gap is 17 points. Capping noise when the image comes before its caption lowers the language loss a lot (0.614 to 0.404), because the model sees a less noisy image when it must write the caption. Reading accuracy on clean images is already 1.000 without the cap, so the cap shows up in the loss and not in the accuracy here. An earlier 3000-step trial at learning rate 1e-3 left every run near chance on drawing, so this task needs the longer schedule.

### 7. Loss weights correct for counting (`07_loss_balance.py`)

Counting loss terms for a 512 x 512 image with a 30-token caption: per-token counting puts 99.3% of the terms on the image, and Emu3's 0.5 trims that to 98.6%. One diffusion mean per image puts 3.2% on the image, and Transfusion's 5 raises it to 14.3%. In the lab's hybrid at initialisation, the trunk gradient from the flow loss is 0.81 times the text gradient at lambda 1 and 4.07 times at lambda 5 when the flow loss is a mean over patches. If the flow loss is summed over patch values instead, the same lambda gives 208 and 1,042 times. The weight only makes sense together with the normalisation.

### 8. Meaning or pixels (`08_readers.py`)

Three encoders with the same architecture. Captions name only the shape. Probes use 2,000 test images.

| Encoder | Shape probe, 64 labels | Shape probe, 256 labels | Side probe, 256 labels | Decoder MSE on frozen features |
|---|---|---|---|---|
| Pixel (reconstruction) | 0.470 | 0.603 | 0.990 | 0.00004 |
| Semantic (caption matching) | 0.819 | 0.962 | 0.979 | 0.00103 |
| Both losses | 0.726 | 0.929 | 0.990 | 0.00008 |

The caption encoder reads the shape far better from few labels, and its features rebuild pixels about 25 times worse. The joint encoder sits between them on reading and is close to the pixel encoder on rebuilding. All three keep the shape's side almost perfectly, so this toy does not show a semantic encoder discarding position. It shows the direction of the trade-off the post describes, not its size.

## Post sections to scripts

| Post section | Script or module |
|---|---|
| Why an image is hard to output: count the bits | `02_token_budget.py` |
| Why you cannot simply regress the pixels | `01_regression_blur.py`, `seedraw/flow.py` |
| Two ways to generate an image: autoregression, rectified flow | `01_regression_blur.py`, `seedraw/flow.py`, `seedraw/priors.py` |
| Masked parallel decoding, and the Figure 2 pass counts | `04_decoding_orders.py` |
| Vector quantization, token counts, keeping the codebook alive | `seedraw/tokenizers.py`, `02_token_budget.py`, `03_tokenizers.py` |
| What a tokenizer should keep: meaning or pixels | `08_readers.py` |
| One transformer, two losses: hybrid models | `06_hybrid.py`, `seedraw/hybrid.py` |
| Balancing the two losses | `07_loss_balance.py` |
| Classifier-free guidance | `05_guidance.py` |
| How unified models are evaluated (a judge model grading generations) | `seedraw/readers.py` (`Judge`), used by 04, 05 and 06 |

## What was not measured

- Paper-scale results. The Janus ablation (MMBench 69.4 against 35.0), Transfusion's compute comparison with Chameleon, MetaMorph's data-efficiency grid, BAGEL's expert layout, GenEval, FID and RealUnify are quoted in the post from the papers. None are reproduced here, and `08_readers.py` is a toy of the Janus trade-off, not a replication.
- Real image tokenizers. There is no VQGAN, no perceptual or adversarial loss, no pretrained VAE, CLIP or SigLIP. Images are synthetic 16 x 16 shapes.
- Bridge models and frozen language models, next-scale prediction (only its pass count is computed), masked discrete diffusion for text, and reinforcement learning for images.
- FID or any distribution metric. Generations are graded by a small judge CNN trained on the same synthetic data.
- GPU speed. Everything ran on CPU, and the timings in `results/` were taken on a loaded machine.
- Several seeds. Every comparison is one seed.

## Theory

- [Models That See and Draw](https://srini-anirudh.github.io/blogs/models-that-see-and-draw/)
