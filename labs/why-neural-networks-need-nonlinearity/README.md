# Lab: Why Neural Networks Need Nonlinearity

Code companion for the post [Why Neural Networks Need Nonlinearity](https://srini-anirudh.github.io/blogs/why-neural-networks-need-nonlinearity/).

The post argues that activation functions stopped being about what a network can represent long ago. What decides
the choice is how gradients flow, how the signal scales with depth, how many exact zeros come out, and how the
tensors behave under quantization. This lab builds every activation from its formula in `nonlin/activations.py`
and runs each of those arguments on a CPU, so you can see it happen and check the numbers.

What you can run:

- Ten affine layers collapse into one, and a deep linear net fails on two spirals.
- A polynomial activation, x³ − x, stops improving with width. ReLU and tanh keep improving.
- The gradient through 18 sigmoid layers falls by about 0.6 decades per layer. ReLU with He init holds level.
- Xavier init with ReLU halves the signal's second moment at every layer.
- A dead ReLU unit's weights never move again. Leaky ReLU, ELU and GELU units keep a gradient.
- GELU is the average of a random keep-or-drop gate, checked by Monte Carlo.
- Seven FFN variants (ReLU, GELU, squared ReLU, bilinear, ReGLU, GEGLU, SwiGLU) at equal parameter count, trained
  inside C4's TinyLLM.
- Only exact zeros let a kernel skip rows of W_down. GELU and SiLU give almost none.
- Swapping the activation in pretrained GPT-2 and CLIP, and what the mismatch costs.

## Setup

From the repository root, after `pip install -e ".[dev]"`. The lab imports C4's `tinyllm` package from the
neighbouring `c04-llm/` folder (script 05 only). Scripts 05 (part c) and 07 download small public models and datasets
from the Hugging Face Hub the first time they run: `gpt2`, `openai/clip-vit-base-patch32`,
`HuggingFaceTB/SmolLM2-135M`, and the datasets `Salesforce/wikitext` and `uoft-cs/cifar10`.

## Run everything

Every script writes `results/<name>.txt` and `results/<name>.json`. Each starts with a hardware line and the command.
Add `--quick` to any script except 04 for a smoke test that takes seconds.

```bash
cd labs/why-neural-networks-need-nonlinearity

python scripts/01_linear_collapse.py          # 11 min on one thread
python scripts/02_gradient_decay.py           # 20 min on one thread
python scripts/03_dying_relu.py               # 19 min on one thread
python scripts/04_smooth_gates.py             # 9 s
python scripts/05_ffn_variants.py --part toy  # 8 min on one thread
python scripts/05_ffn_variants.py --part lm --workers 8 --threads 1   # 21 trainings, see the note below
python scripts/06_sparsity.py                 # 10 s
python scripts/07_pretrained_swaps.py         # 27 min, downloads about 1.5 GB the first time
```

The language-model sweep is 7 variants x 3 seeds = 21 trainings of 2000 steps. It took 31 min with 21 workers on a
heavily shared machine. On an idle 8-core machine, `--workers 8 --threads 1` is a sensible setting. Use `--steps 500
--seeds 1` for a quick look. Several scripts were slower than they need to be because the machine had a load average
well above its core count while they ran.

Tests:

```bash
python -m pytest . -m "not slow and not gpu and not network" -o addopts="--import-mode=importlib" -q   # offline, 10 s
python -m pytest . -m "slow or network" -o addopts="--import-mode=importlib" -q                          # runs every script with --quick, downloads models
```

Test counts (AMD EPYC 7R13, torch 2.15.0): 55 offline tests pass in 10 s. The 8 `slow` and `network` tests pass in 152 s.

## Which script shows what

| Post section | Claim | Script |
|---|---|---|
| 01 Why anything at all | Stacked affine maps collapse into one | `01_linear_collapse.py` (a) |
| 01 | A deep linear net cannot separate two spirals | `01_linear_collapse.py` (b) |
| 01 | Only non-polynomial activations are universal: x^3 - x is not | `01_linear_collapse.py` (c) |
| 02 Saturating era | Sigmoid loses about 0.65 decades of gradient per layer | `02_gradient_decay.py` (a) |
| 02, 08 | He init keeps the signal for ReLU, Xavier halves it | `02_gradient_decay.py` (a), (b) |
| 08 Initialisation | The right gain depends on the activation | `02_gradient_decay.py` (c) |
| 02 | The activation was one fix among several | `02_gradient_decay.py` (d) |
| 03 Rectifier revolution | A dead ReLU is an absorbing state | `03_dying_relu.py` |
| 04 Smooth gates | GELU is the mean of a stochastic gate | `04_smooth_gates.py` (a) |
| 04 | Dips, slopes and gaps between GELU, SiLU, Mish, QuickGELU | `04_smooth_gates.py` (b), (c), (d) |
| 04, 08 | The tanh form is a different function | `04_smooth_gates.py` (c), `07_pretrained_swaps.py` (a) |
| 05 GLU | The 8d/3 rule and the width 11008 | `05_ffn_variants.py` (a) |
| 05 | A gated unit is a product of features | `05_ffn_variants.py` (b) |
| 05 | Gated FFNs beat ungated at equal parameters | `05_ffn_variants.py` (c) |
| 05 | The gated product has a heavier tail | `05_ffn_variants.py` (d), `07_pretrained_swaps.py` (c) |
| 06 Sparsity | Exact zeros can be skipped, near-zeros cannot | `06_sparsity.py` |
| 08 QuickGELU | A GELU/QuickGELU mismatch costs accuracy | `07_pretrained_swaps.py` (b) |
| 08 | Do not change the activation of a pretrained model | `07_pretrained_swaps.py` (a) |

The animated figures are not reproduced. The code gives the numbers behind them.

## Results

All numbers come from the files in `results/`. Scripts 01 to 05 ran on one thread each, in parallel, on a machine
with a load average far above its core count. Script 06 timings are affected by that load, and script 07 ran on 8 threads.
Accuracies, losses and counts do not depend on load.

### AMD EPYC 7R13, 1 thread per script, torch 2.15.0 (CPU)

**Section 01** (`results/01_linear_collapse.txt`)

| Measurement | Result |
|---|---|
| 10 affine layers vs the collapsed single map, max difference, float64 | 2.2e-16 |
| Two spirals, 6 hidden layers, train / test accuracy: identity | 0.651 / 0.652 |
| Same, ReLU | 0.999 / 0.991 |
| Same, GELU | 0.998 / 0.989 |
| Same, tanh | 0.946 / 0.941 |
| One hidden layer fit of sin(3 pi x) + 0.5\|x\|, MSE, width 256: ReLU / tanh / x^3 - x | 0.094 / 0.203 / 0.414 |
| Best cubic polynomial, MSE | 0.414 |

The x^3 - x network sits exactly at the best cubic at every width (0.4140). ReLU improves from 0.375 at width 4 to
0.094 at width 256. Tanh improves from width 4 to 16 and then does not improve consistently (0.117 at 16, 0.281 at
64, 0.203 at 256). That is an optimisation effect in a 2000-step run, not evidence about the limit of tanh networks.

**Section 02 and 08** (`results/02_gradient_decay.txt`), 18 layers of width 256, mean of 5 seeds

| Activation + init | dL/dz decades per layer | dL/dW decades per layer | E[h^2] ratio per layer |
|---|---|---|---|
| sigmoid + Xavier | -0.628 | -0.634 | not measured |
| tanh + Xavier | -0.033 | 0.001 | 0.861 |
| ReLU + Xavier | -0.151 | 0.003 | 0.487 |
| ReLU + He | 0.001 | -0.033 | 0.974 |
| GELU + He | -0.012 | 0.016 | not measured |
| SiLU + He | -0.109 | 0.011 | not measured |

The post says sigmoid with Xavier loses about 0.65 decades per layer. We measure 0.63. ReLU with Xavier halves the
forward second moment at every layer (ratio 0.487, as the post's "keeps half the variance" argument says), and its
gradient at the pre-activations shrinks by 0.15 decades per layer. The weight gradient does not show this, because
dL/dW is the product of that gradient and a forward signal that shrinks the opposite way. Look at dL/dz to see the
backward signal. The gain that keeps the signal's scale fixed is sqrt(2) for ReLU at any scale, but for GELU it is
1.73, 1.53 and 1.44 at input variance 0.25, 1 and 4, so "the" gain of GELU depends on the scale.

Training the 18-layer net on two spirals for 1500 full-batch steps (train accuracy, mean of 3 seeds): sigmoid
0.513 with SGD and 0.500 with Adam, tanh 0.611 and 0.737, ReLU with He 0.829 and 0.999. This is one small problem.
It does not isolate the activation, because there are no residuals or normalisation layers, which the post names as
the other fixes.

**Section 03** (`results/03_dying_relu.txt`). 60 hidden units, two spirals, bias pushed down by 0.002 per step for
600 steps, then 1800 steps without the push. Five seeds, 300 units per activation.

| Activation | Units inactive on every input at step 600 | At the end | Recovered | Largest weight change afterwards |
|---|---|---|---|---|
| ReLU | 116 | 117 | 0 | exactly 0 |
| Leaky ReLU (0.01) | 117 | 118 | 0 | 0.020 |
| Leaky ReLU (0.1) | 119 | 119 | 2 | 0.181 |
| ELU | 143 | 141 | 6 | 0.370 |
| GELU | 136 | 131 | 7 | 0.312 |

The mechanism is exact: the weights of every inactive ReLU unit did not change by even one float. The other
activations kept a gradient, so their weights moved, but in this budget only a few units crossed back to
the active side. Most stayed inactive. A unit that is inactive but still trains is not dead, but this run does
not show it coming back in large numbers. The post's Fig 03 is a simulation with a drift that continues, so
it is a different experiment. Part (b), with no drift, shows large learning rates alone kill units: with plain SGD at lr 2.0, 78% of ReLU units
are inactive, against 51% for Leaky ReLU (0.01), 44% for Leaky ReLU (0.1), 46% for ELU and 1% for GELU. At lr 1.0 the order is different
(ELU 51%, ReLU 23%), so this is not a clean ranking.

**Section 04** (`results/04_smooth_gates.txt`), float64, grid step 1e-5

| Function | Minimum (x, f) | Largest f' (x, f') |
|---|---|---|
| GELU | -0.752, -0.170 | 1.414, 1.129 |
| QuickGELU | -0.751, -0.164 | 1.410, 1.100 |
| SiLU | -1.279, -0.279 | 2.399, 1.100 |
| Mish | -1.192, -0.309 | 1.491, 1.089 |
| Hard-swish | -1.500, -0.375 | 3.000, 1.500 |

The Monte-Carlo mean of x * Bernoulli(Phi(x)) converges to GELU with error times sqrt(draws) staying near 0.6 from 8
to 32768 draws, so the error falls as 1/sqrt(draws). All 13 functions in `activations.py` match PyTorch's within 1.8e-15.

**Section 05** (`results/05_ffn_variants_toy.txt`, `results/05_ffn_variants_lm.txt`)

The 8d/3 rule gives 10922.7 for d = 4096, which rounds up to 11008 at a multiple of 256. At d = 4096 an ungated
FFN with h = 4d has 134,217,728 parameters and a gated one with h = 11008 has 135,266,304.

Sum of four products, y = x1x2 + x3x4 + x5x6 + x7x8, one FFN layer, held-out MSE (the variance of y is 0.444).
Gated width hg, ungated width 1.5 hg, so the parameters match:

| FFN | hg = 2 | hg = 4 | hg = 8 | hg = 16 |
|---|---|---|---|---|
| ReLU | 0.334 | 0.195 | 0.0170 | 0.0118 |
| GELU | 0.283 | 0.149 | 0.0008 | 0.0004 |
| Squared ReLU | 0.325 | 0.259 | 0.0264 | 0.0000 |
| Bilinear (no nonlinearity) | 0.126 | 0.0000 | 0.0000 | 0.0000 |
| ReGLU | 0.272 | 0.120 | 0.0003 | 0.0000 |
| SwiGLU | 0.127 | 0.0012 | 0.0000 | 0.0000 |

Bilinear and SwiGLU fit the target at hg = 4, which is the number of products. The elementwise FFNs need many more
units. This is a target built to favour products, so it shows what a degree-2 unit can express, not that real data
needs it.

Byte-level language model on wikitext-2, C4's TinyLLM with the FFN swapped (d = 128, 4 layers, 4 heads, sequence 128,
batch 32, AdamW lr 0.003, 2000 steps), validation loss in nats per byte, mean and standard deviation over 3 seeds:

| FFN | h | FFN params | Val loss | Std |
|---|---|---|---|---|
| GELU | 512 | 524,288 | 1.4606 | 0.0029 |
| ReLU | 512 | 524,288 | 1.4317 | 0.0030 |
| Squared ReLU | 512 | 524,288 | 1.4179 | 0.0030 |
| Bilinear | 344 | 528,384 | 1.4196 | 0.0039 |
| ReGLU | 344 | 528,384 | 1.4197 | 0.0073 |
| SwiGLU | 344 | 528,384 | 1.4106 | 0.0015 |
| GeGLU | 344 | 528,384 | 1.4055 | 0.0062 |

The gated variants and squared ReLU beat plain ReLU and GELU by more than the seed spread (about 0.003 to 0.007). GeGLU
and SwiGLU are 0.005 apart, which is within one seed standard deviation, so the data does not separate them,
as the post says. Bilinear matches ReGLU, as Shazeer found. Squared ReLU is as good as the gated ones here,
which agrees with the post's remark that it is "nearly free" in comparison, but it is one small model.
In this run plain ReLU beat plain GELU by 0.029, a gap
we did not expect from the post, and which may reflect the short training, the high learning rate or the byte-level task.
Shazeer's sweep used T5-base for 65k steps on a different task. His numbers were not reproduced.

The tensor that goes into W_down, first validation batch, mean over layers and seeds:

| FFN | Exact zeros | Excess kurtosis (mean / worst layer) | max / rms | INT8 per-tensor relative error |
|---|---|---|---|---|
| ReLU | 0.898 | 42 / 63 | 19 | 0.014 |
| GELU | 0.000 | 33 / 43 | 17 | 0.038 |
| Squared ReLU | 0.869 | 408 / 626 | 62 | 0.044 |
| Bilinear | 0.000 | 28 / 43 | 24 | 0.054 |
| ReGLU | 0.831 | 245 / 395 | 57 | 0.048 |
| GeGLU | 0.000 | 130 / 179 | 41 | 0.089 |
| SwiGLU | 0.000 | 108 / 142 | 40 | 0.088 |

The post says the elementwise product of two projections has a heavier tail and that per-tensor INT8 handles it
worst. Here SwiGLU and GeGLU have about 2.3 times the INT8 error of GELU and 3 to 4 times the kurtosis. Squared
ReLU has the heaviest tail, which the post does not say. ReLU and the other ReLU-based FFNs reach 83% to 90% exact
zeros after training, much more than the half you see at initialisation.

**Section 06** (`results/06_sparsity.txt`)

On 6000 Gaussian pre-activations, ReLU gives 50.6% exact zeros. GELU, GELU (tanh) and SiLU give none. Even within 0.01 of zero, GELU has only 2.0% of outputs.
Skipping rows of W_down for one token at the Llama-2-7B size (d = 4096, h = 11008, float32, 8 threads, a loaded machine):

| Exact-zero fraction | Rows read | Dense / sparse (ms) | Speed-up |
|---|---|---|---|
| 0.504 | 5457 | 13.79 / 29.62 | 0.47 |
| 0.899 | 1112 | 11.27 / 2.17 | 5.20 |
| 0.989 | 116 | 11.26 / 0.26 | 43.85 |

The sparse result equals the dense one exactly (max difference 0). At 50% sparsity the gather costs more than it
saves, so skipping is slower. The post's "free 2x" at half sparsity holds for bytes read, not for this
index-and-gather code on CPU. A real kernel does better, and we did not write one. GELU has no exact zeros, so the same
code reads all 11008 rows.

### AMD EPYC 7R13, 8 threads, torch 2.15.0, pretrained models (CPU)

`results/07_pretrained_swaps.txt`

GPT-2 small, wikitext-2 test, 40 windows of 1024 tokens, no fine-tuning. GPT-2 was trained with the tanh GELU.

| Activation in every MLP | Perplexity |
|---|---|
| GELU, tanh form (the checkpoint's own) | 31.07 |
| GELU, exact | 31.07 |
| QuickGELU | 31.11 |
| SiLU | 1321.57 |
| ReLU | 3128.02 |

The tanh and exact forms differ by 0.004 in perplexity. QuickGELU costs 0.04 here. SiLU is not a drop-in for GELU:
its perplexity is 40 times worse. The max logit change of QuickGELU on 64 tokens is large (50.9), while perplexity barely
moves, so a few logits are affected strongly. We did not investigate which. The post says SiLU and GELU differ by under 0.02 pointwise. We
measure 0.19 (above), so that statement fits QuickGELU, not SiLU.

CLIP ViT-B/32 (trained with QuickGELU), zero-shot CIFAR-10, first 2000 test images, prompt "a photo of a {label}.", activation
changed in the vision tower only:

| Vision activation | Accuracy | Mean cosine to the original image embedding |
|---|---|---|
| QuickGELU (the checkpoint's own) | 0.8985 | 1.0000 |
| GELU, exact | 0.8595 | 0.9363 |
| GELU, tanh | 0.8575 | 0.9346 |

The loss is 3.9 points on this model and task. The post's "roughly a point" is for ViT-L/14 on ImageNet, which we did not
run. Nothing here contradicts it, but the size of the effect clearly depends on the model and the task.

The tensor that goes into W_down in two real models (first 1024 tokens of wikitext-2 validation, mean over layers):

| Model (FFN) | Exact zeros | Excess kurtosis | max / rms | INT8 error |
|---|---|---|---|---|
| GPT-2 (GELU), all tokens | 0.000 | 1249 | 78 | 0.155 |
| GPT-2 (GELU), without token 0 | 0.000 | 58 | 36 | 0.079 |
| SmolLM2-135M (SwiGLU), all tokens | 0.000 | 67840 | 156 | 0.165 |
| SmolLM2-135M (SwiGLU), without token 0 | 0.000 | 2617 | 80 | 0.153 |

Most of the tail comes from the first token, a known effect (Sun et al., 2024). Once it is removed the SwiGLU model
still has the heavier tail and the larger INT8 error. The two models differ in size, data, norm and training, so this is
an observation, not an ablation.

## What was not measured

- The animated figures and the model matrix ("who ships what") are not reproduced. The matrix comes from config files
  and was not re-read here.
- Shazeer's GLU sweep (T5 span corruption, 65k steps) and the Mirzadeh et al. result (ReLU in a pretrained LLM after
  fine-tuning) were not reproduced. Our language model is a 4-layer byte model trained for 2000 steps.
- Real inference speed-ups from activation sparsity need a kernel that skips memory reads. We timed an index-and-gather
  in PyTorch on CPU.
- ViT-L/14 on ImageNet (the post's CLIP example), SigLIP, Qwen2.5-VL, OpenCLIP's `-quickgelu` variants, SmoothQuant
  and LLM.int8() were not run.
- SELU's self-normalising property, PReLU's learned slopes, KANs and xIELU are not covered.
- No GPU was used. Everything here runs on CPU.

## Where the post and the measurements differ

- The post says GELU has its dip near x = -1.28 with value about -0.17. The value -0.17 is right. The location is
  -0.75. The x = -1.28 location is SiLU's (whose minimum is -0.278).
- The post says SiLU and GELU differ by under 0.02 everywhere. The largest gap is 0.19. QuickGELU, not SiLU, is
  within 0.02 of GELU. This also affects the "coin flip" argument for GeGLU against SwiGLU: the functions are similar but
  not that close. In our language-model runs the two were within seed noise.
- The post says GELU and SiLU "both overshoot to about 1.1 near x of about 1.5". GELU's peak slope is 1.129 at 1.41.
  SiLU's is 1.100 at 2.40.
- The post says the GELU/SiLU effective gain is "about 1.7". We measure 1.73, 1.53 and 1.44 for GELU at three input
  scales, and 1.86, 1.68 and 1.51 for SiLU. About 1.7 fits at unit input variance for SiLU, not as a single constant.

