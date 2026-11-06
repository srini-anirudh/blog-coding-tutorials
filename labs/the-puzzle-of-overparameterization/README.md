# Lab: The Puzzle of Overparameterization

Code companion to the theory post
[The Puzzle of Overparameterization](https://srini-anirudh.github.io/blogs/the-puzzle-of-overparameterization/).
The post asks why networks that can memorize their training set mostly do not. This lab rebuilds its central
experiments in small numpy and PyTorch scripts, so you can watch each idea happen on a CPU.

You can:

- turn the label-corruption dial on a CIFAR-10 subset and see the same MLP fit real and random labels;
- compute bias, variance and noise for polynomial fits, and the vacuous VC bound;
- find the double-descent peak along model size, sample size and training time, then remove it with ridge;
- split a minimum-norm interpolant into its signal and noise pieces, and see which spectra hide the noise;
- watch gradient descent walk towards the max-margin direction, and see other optimizers pick other answers;
- permute and rescale a trained network: same function, very different sharpness;
- push one network between lazy and rich training with a single scale;
- watch a small MLP grok modular addition, with two progress measures;
- count bits: the capacity budget, and a compression bound that stops being vacuous.

## Layout

```
labs/the-puzzle-of-overparameterization/
├── overparam/
│   ├── linear.py         # one exact least-squares solver; bias-variance; double descent (Figs 1, 4-6)
│   ├── benign.py         # min-norm interpolant split into signal and noise pieces; spectra (Fig 7)
│   ├── implicit_bias.py  # hard-margin SVM, GD on logistic loss, steepest descent in l2 / l1 / l-inf
│   ├── memorize.py       # CIFAR-10 subset, label corruption, MLP fit to 100% train accuracy
│   ├── symmetry.py       # permutation and rescaling, Hessian top eigenvalue by power iteration
│   ├── lazy_rich.py      # two-layer net with output scale alpha, tangent kernel, lazy limit
│   ├── grokking.py       # (a + b) mod 97, MLP, weight norm and Fourier progress measures
│   ├── bits.py           # VC arithmetic, capacity budget, quantize + prune + lzma compression bound
│   └── common.py         # hardware line, result files, tables
├── scripts/              # one script per post section, listed below
├── results/              # what the scripts printed (.txt) and their numbers (.json)
└── tests/
```

## Setup

From the repository root, after `pip install -e ".[dev]"` (numpy, torch, and `datasets` for CIFAR-10).
PyTorch uses every core by default, so set `OMP_NUM_THREADS` on a shared machine. The numbers below used 8 threads.

## Run everything

```bash
L=labs/the-puzzle-of-overparameterization
python -m pytest $L -m "not slow and not gpu and not network"   # offline tests, about 10 s
python -m pytest $L -m network                                    # downloads CIFAR-10 once

python $L/scripts/01_bias_variance.py        # 20 s
python $L/scripts/02_random_labels.py        # CIFAR-10 subset, 11 min
python $L/scripts/02_random_labels.py --synthetic   # offline version, seconds
python $L/scripts/03_double_descent.py       # 15 s
python $L/scripts/04_kill_the_peak.py        # 3 min
python $L/scripts/05_benign_overfitting.py   # 1.5 min
python $L/scripts/06_implicit_bias.py        # 1 min
python $L/scripts/07_symmetries.py           # 10 s
python $L/scripts/08_lazy_rich.py            # 10 to 17 min
python $L/scripts/09_grokking.py             # 16 min
python $L/scripts/10_bits_and_bounds.py      # 35 s
```

Every script writes `results/<script>.txt` (what it printed) and `results/<script>.json`. Each file starts with
the hardware line and the command.

## Which script covers which section

| Post section | Script | What it reproduces |
|---|---|---|
| 1. The classical picture | `01_bias_variance.py` | Figure 1 decomposition; the VC bound near 300 |
| 2. Networks memorize anything | `02_random_labels.py` | Figure 2 label-corruption dial, random pixels |
| 3. The interpolation threshold, 4. Double descent | `03_double_descent.py` | Figures 4, 5 and 6 |
| 5. What double descent does not explain | `04_kill_the_peak.py` | Lab 1: ridge and label noise vs the peak |
| 6. Benign overfitting | `05_benign_overfitting.py` | Figure 7; the spectrum conditions |
| 7. The algorithm picks the solution | `06_implicit_bias.py` | Figure 8; Lab 2; the optimizer-geometry claim |
| 8. Parameter space is not function space | `07_symmetries.py` | Figure 9; Dinh et al.'s sharpness argument |
| 9. Lazy and rich | `08_lazy_rich.py` | Figure 10; the NTK limit; width scaling |
| 10. Grokking | `09_grokking.py` | Figure 11 at small scale |
| 11. The regime LLMs train in, 12. What can be certified | `10_bits_and_bounds.py` | Lab 3 arithmetic; Figure 13 ladder |

C1 to C12 in this repository do not build any of these mechanisms, so the lab imports nothing from them.

## Results

Every number comes from the files in `results/`, run on AMD EPYC 7R13, 8 threads, torch 2.15.0+cu132, numpy 2.5.4.
The machine was shared and heavily loaded, so the run times above are upper bounds, and the `seconds` column in
`results/02_random_labels.txt` is not comparable between rows. Nothing was run on a GPU. Offline tests: 21 passed.
The `network` test (CIFAR-10 download): 1 passed. There are no `slow` tests.

| Experiment | Result |
|---|---|
| Bias-variance, n = 14, sigma = 0.24, 2,000 redraws | Lowest expected error at degree 3 (0.0748). Degree 12: 45.4, almost all variance (45.3) |
| VC bound, 11M weights, 20 layers, 50,000 images | VC term sqrt(VC / n) = 321 |
| MLP (1.84M parameters) on 5,000 CIFAR-10 images, 0% / 100% random labels | Reaches 100% train accuracy in 37 / 55 epochs (1.5x the steps). Test accuracy 0.420 / 0.098 |
| Same MLP, 20 / 40 / 60 / 80% random labels | 52 / 38 / 39 / 43 epochs to 100% train accuracy. Test accuracy 0.363 / 0.311 / 0.221 / 0.166 |
| Same MLP, Gaussian-noise inputs, random labels | 100% train accuracy in 3 epochs. Test accuracy 0.096 |
| Model-wise double descent, n = 40 | Median excess risk 0.103 at p = 10, **15.5 at p = 40**, 0.122 at p = 420 |
| Sample-wise, p = 60 | Median excess risk 0.249 at n = 24, **28.0 at n = 60**, 0.253 at n = 108 |
| Epoch-wise, n = 40, p = 44 | Best 0.089 at 10 steps; 1.98 at the interpolating end point. Training MSE falls to 1e-19 |
| Kill the peak, p = n = 40 | Ridgeless 15.5; ridge 0.1: 0.43; oracle-tuned ridge 0.105 (curve stays within 0.094 to 0.112 for all p) |
| Same, label noise removed | sigma = 0.1: 1.81; sigma = 0: 0.73 at p = 40. The peak shrinks 21x but does not vanish |
| Benign overfitting, n = 30, p = 100 | Zero training residual, total risk 0.118 against noise variance 0.16 |
| Same, p = 31 / 10,000 | Noise piece of the risk 3.03 / 0.0006. Signal piece 0.0001 / 0.903 |
| Noise risk vs n, spike + flat tail of n^2 dims | 0.051, 0.025, 0.013, 0.0063, 0.0031 for n = 25 ... 400 |
| Noise risk vs n, isotropic and k^-2 spectra | Stay at 1.0 (1.04 to 1.00, and 0.98 to 1.02) |
| Noise risk vs n, k^-1 log^-2(k+1) | 0.49 to 0.42. Does not fall to 0 at this size |
| GD on separable logistic regression, 2-D, n = 40, dataset seed 17 | After 10^6 steps: l2 margin 0.1438 vs SVM 0.1499. Norm grows 2.4, 4.8, 8.7, 13.2, 15.1 per decade (theory: 15.4) |
| Same, dataset seed 1 | Margin 0.1178 vs SVM 0.1225. Angle to the SVM direction 0.18 at step 10, 0.024 at 10^3, 0.049 at 10^6 |
| Seven weight-decay settings, 1,000 steps, zero training error in all | Angle to the SVM direction 0.019 to 0.201 rad; test accuracy 0.9677 to 0.9425 |
| Steepest descent in l2 / l1 / l-inf, d = 50 | Each has the best margin in its own norm. Share of the l1 norm in the top 5 coordinates: 0.29 / 0.82 / 0.13 |
| Permuting hidden units of a trained 1-64-1 net | Largest output change 3.6e-7 |
| Rescaling layer 1 by c, layer 2 by 1/c, c from 0.1 to 10 | Output change at most 4.8e-7. Top Hessian eigenvalue 234 (c = 0.1), 22.5 (c = 1), 1,850 (c = 10): 82x range |
| Lazy limit (initial tangent kernel), width 1024 | Test MSE 0.0287 |
| Output scale alpha = 100 / 1 / 0.1 | Relative weight change 0.0008 / 0.082 / 0.70. Test MSE 0.0287 / 0.0267 / 0.0196 |
| Width 256 / 1024 / 4096, alpha = 1 | Relative weight change 0.174 / 0.082 / 0.040. Test MSE 0.0273 / 0.0267 / 0.0266 |
| Same widths, alpha = 8 / sqrt(width) | Relative weight change 0.327 / 0.318 / 0.309. Test MSE 0.0215 / 0.0206 / 0.0199 |
| Grokking, (a+b) mod 97, 30% train, weight decay 1.0 | 99% train accuracy by step 500; validation 0.046 at 6,500, 0.385 at 8,500, 0.910 at 10,500, 99% at 12,000 |
| Same run, progress measures | Weight norm 98.3 (step 500) to 64.3 (step 18,500). Embedding power in top 8 frequencies 0.187 to 0.705 (random table: about 0.17) |
| Same, weight decay 0 | Validation accuracy stays at chance for 20,000 steps. Weight norm grows from 133 (step 500) to 287 (step 20,000) |
| Capacity budget, 7B parameters, 10T tokens | 3.15 GB vs 2.5 TB: the corpus holds 794x more bits |
| Compression ladder, real labels, 50,000 examples, 1,474-parameter MLP | Bound 0.975 (VC), 0.582 (float32), 0.299 (8-bit + lzma), **0.258 (4-bit + lzma)**; measured test error 0.042 |
| Same, random labels | Best bound 0.656; measured test error 0.504 |

### Things in the results that differ from the post

- **The second descent does not beat the first minimum here.** The post says test error often comes back below the
  classical sweet spot. In this model, p = 420 reaches 0.122 against 0.103 at p = 10. Past p = 420 the curve
  would need a larger model to cross it, and we did not look.
- **Removing label noise does not remove the peak here.** The post says the peak nearly disappears at zero noise. In this
  model it falls from 15.5 to 0.73, which is still 17x the p = 10 value. A likely reason is that the true
  coefficients on coordinates the model cannot use act like noise. We did not test this.
- **Tuned ridge flattens the curve, but the monotonicity theorem does not strictly apply.** The theorem is for
  isotropic data. This data has a 1/j spectrum. The oracle picks lambda using the true risk.
- **The k^-1 log^-2 spectrum did not look benign at n up to 400.** The benign condition is asymptotic in n, and p was cut
  at 20,000. The spike-plus-flat-tail spectrum, built to satisfy the condition, did go to 0.
- **Growing the tail forever is not benign for total risk.** The noise piece vanishes, but at p = 10,000 the signal piece is
  0.903 and the total risk is 0.903. The tail variance is fixed, so more tail dimensions soak up signal too.
- **The angle to the SVM direction is not monotone.** It fell from 0.18 to 0.024 and then rose to 0.049 on seed 1.
  The margin did approach the SVM margin (96% at 10^6 steps on both seeds). The log-t rates were not verified.
- **Rich training here is mild.** The tangent kernel changes by at most 0.047 (cosine distance), and test MSE improves by
  about 25 to 32%. Each of the lazy-rich runs uses one seed.
- **Grokking here uses an MLP, not a transformer.** One seed per weight-decay setting; accuracy is logged every 500 steps.
- **The compression bound covers the compressed model only.** Its bit count is the lzma size of the integer codes plus one
  32-bit scale per tensor; it does not count the architecture or the decoder. The five rungs are paid for with a
  union bound (delta / 5 each). The random-label bound is above 0.5, so it is vacuous on a two-class task.

## What was not measured

- Paper-scale results: ImageNet random labels (95.2% top-1), the Inception and CNN runs of Zhang et al., deep double
  descent on ResNets, and the Morris et al. 3.6 bits per parameter. We only did the arithmetic with the quoted figure.
- Lotfi et al.'s non-vacuous bounds for pretrained LLMs. We build the same idea on a 1,474-parameter classifier.
- Grokking in a transformer, Nanda et al.'s Fourier circuit, and the ungrokking and semi-grokking predictions.
- Implicit bias of AdamW on a transformer, which the post says is open.
- Muennighoff's repeated-data scaling and Carlini's extraction results.
- The log t and 1 / log t rates for gradient descent on separable data, and many seeds for any experiment above.
- GPU runs. Everything is CPU, so no GPU numbers appear.
