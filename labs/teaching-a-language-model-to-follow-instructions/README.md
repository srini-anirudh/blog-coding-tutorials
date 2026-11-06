# Lab: teaching a model what we prefer, from RLHF to DPO

This lab is the code companion to the post
[Teaching a Model What We Prefer: From RLHF to DPO](https://srini-anirudh.github.io/blogs/teaching-a-language-model-to-follow-instructions/).
The post argues that preference tuning is a chain of manufactured signals. A comparison becomes a probability through
Bradley-Terry. A probability becomes a reward. A reward becomes a policy through the KL-regularised objective. DPO
then folds the last two steps into one loss. The lab builds each link in small PyTorch and runs it, so you can see
each claim hold, or fail, with real numbers.

Most experiments use spaces small enough to enumerate. There you can compute the exact optimal policy and check
the derivations to machine precision. One experiment trains a real 135M-parameter language model with DPO.

Tutorial [C6, post-training](../../c06-post-training/) already builds SFT, a reward model, DPO and GRPO on
SmolLM2-135M. This lab reuses C6's pair data, tokenisation and log-prob code for the real-model run. It does not
rebuild them. It covers what the post adds: the derivations, the loss family, and the two failure modes.

## Layout

```
labs/teaching-a-language-model-to-follow-instructions/
├── prefopt/
│   ├── bt.py        # Gumbel and Gaussian annotators, Bradley-Terry fit, the self-annealing gradient
│   ├── tilt.py      # the KL-regularised optimum, beta log Z, implicit reward, tabular DPO
│   ├── losses.py    # DPO, IPO, SLiC, RRHF, SimPO, KTO, ORPO on one margin axis; DPO-Positive and NLL patches
│   ├── bandit.py    # a one-prompt policy with shared features, for likelihood displacement
│   ├── overopt.py   # gold reward, proxy reward models, best-of-n and the exact KL tilt
│   ├── lm.py        # DPO on SmolLM2-135M-Instruct, tracking chosen and rejected log-probs (uses C6)
│   └── common.py    # hardware labels and result files
├── scripts/         # 01 to 07, one per claim, listed below
├── tests/
└── results/         # the real outputs; every file starts with its hardware line and command
```

## Setup

From the repository root, after `pip install -e ".[dev]"`. Only script 07 and the `network` test download
anything (SmolLM2-135M-Instruct, about 270 MB). Everything runs on CPU. No GPU is needed or used.

## Run everything

```bash
export OMP_NUM_THREADS=8
L=labs/teaching-a-language-model-to-follow-instructions
python $L/scripts/01_bradley_terry.py           # ~20 s
python $L/scripts/02_kl_tilt.py                 # ~20 s
python $L/scripts/03_dpo_identity.py            # ~1.5 min
python $L/scripts/04_loss_family.py             # ~4 s; try --beta 0.5 --target 2
python $L/scripts/05_likelihood_displacement.py # ~10 s
python $L/scripts/06_overoptimisation.py        # ~4.5 min
python $L/scripts/07_dpo_real_model.py          # downloads SmolLM2-135M-Instruct; 4 to 10 min per method on CPU

# Tests
python -m pytest $L -m "not slow and not gpu and not network"   # 21 tests, ~10 s
python -m pytest $L -m network                                  # 1 test, real model, ~45 s
```

## Which script covers which section of the post

| Post section | Script | What it shows |
|---|---|---|
| From comparisons to a scalar field | `01_bradley_terry.py` | Gumbel noise gives the sigmoid, Gaussian gives the probit; a reward recovered up to shift and scale; the gradient is sigmoid(-delta); a preference cycle fits as a tie |
| The KL-regularised objective | `02_kl_tilt.py` | The closed form matches gradient ascent and equals beta log Z; the support never grows; beta as temperature |
| The DPO identity, derived slowly | `03_dpo_identity.py` | r = beta log(pi*/ref) + beta log Z holds exactly; Z cancels in Bradley-Terry; the reward-model route and the DPO route give the same policy |
| The offline preference family (Figure 9) | `04_loss_family.py` | Loss and gradient of seven methods on one margin axis |
| What the DPO gradient actually does; the failure mode (Figure 8) | `05_likelihood_displacement.py` | The loss falls while the chosen log-prob falls; mass leaves for completions outside the data; DPO-Positive and an NLL term |
| The same failure on a real model | `07_dpo_real_model.py` | DPO on SmolLM2-135M-Instruct with near-identical pairs |
| Pathology 2, Goodhart's law, measured (Figure 5) | `06_overoptimisation.py` | Gold reward rises, peaks and falls under best-of-n; better proxies peak later and higher; Gao et al.'s forms fit the curves |

## Results

Every number names its hardware. Raw output is in `results/`, under the name of the script.

Every script and test ran on AMD EPYC 7R13, 8 threads, torch 2.15.0. The machine was shared with other jobs,
so the run times above are rough. None of the numbers below depend on speed.

### Bradley-Terry, the tilt and the identity (AMD EPYC 7R13, 8 threads, torch 2.15.0)

| Measurement | Result |
|---|---|
| Gumbel annotator vs sigmoid, Gaussian annotator vs probit, 400k draws per point | largest gap 0.0010 (binomial std 0.0008) |
| Best rescaled logistic vs probit | sigmoid(1.203 delta), largest gap 0.0095 in probability |
| Reward fitted from 20,000 comparisons, noise scale 0.5 / 1 / 2 | fitted scale 2.11 / 1.00 / 0.51 times the truth |
| Loss change when every score is shifted by 3.7 | 0.0 |
| A 90% preference cycle A > B > C > A | best scores tie at 0, NLL = log 2 |
| KL objective at the closed form vs Adam, beta = 0.1 | 1.676771 vs 1.676732 (closed form = beta log Z) |
| Completion with reward 10 but reference probability 0 | probability 0 at every beta |
| Same completion at reference probability 1e-12 | overtakes the runner-up only below beta = 0.336 |
| r_hat - r within one prompt, at pi* | constant to 4.4e-16 (equals -beta log Z) |
| Bradley-Terry from implicit rewards vs from true rewards | equal to 2.2e-16 |
| Reward-model route vs DPO route, same 300 to 300,000 pairs | total variation 0.00000 to 0.00020 |
| DPO policy vs pi*, 300 / 3,000 / 30,000 / 300,000 pairs | total variation 0.266 / 0.136 / 0.031 / 0.008 |

### Loss family and likelihood displacement (AMD EPYC 7R13, 8 threads, torch 2.15.0)

| Measurement | Result |
|---|---|
| DPO gradient by autograd vs -beta sigmoid(-beta h) | equal to 1.1e-16 |
| Hinges (SLiC, RRHF) past their margin | gradient exactly 0 |
| IPO past h = 1/(2 beta) | gradient changes sign and pulls the margin back |
| Toy DPO, similarity 0: chosen log-prob change at step 50 / step 400 | +1.50 / -9.64 nats, while the loss falls 0.693 to 0.049 |
| Toy DPO, similarity 0.9: chosen log-prob change at step 400 | -35.8 nats, loss 0.693 to 0.210 |
| Toy DPO, similarity 0 / 0.5 / 0.9 / 0.99: chosen log-prob change at step 400 | -9.6 / -18.4 / -35.8 / -47.2 nats |
| Mass on completions outside the data, start / end of plain DPO | 0.984 / 0.995 to 1.000 |
| Same runs with DPO-Positive (lambda 5) | chosen +0.17 to +1.61 nats |
| Same runs with an extra 0.1 NLL term | chosen +4.3 to +4.7 nats |

### Goodhart's law in miniature (AMD EPYC 7R13, 8 threads, torch 2.15.0)

Gold is measured from its mean under the reference. The best single completion scores 4.25.

| Proxy reward model | Best-of-n peak (d, gold) | Gold at n = 16384 (d = 2.95) | Fit R^2 of d(alpha - beta d) |
|---|---|---|---|
| 300 pairs, MLP width 32 | 1.78, 3.00 | 2.60 | 0.924 |
| 1000 pairs, linear | 1.96, 3.07 | 1.81 | 0.995 |
| 1000 pairs, width 4 | 2.13, 3.46 | 2.29 | 0.988 |
| 1000 pairs, width 32 | 2.29, 3.72 | 3.48 | 0.992 |
| 3000 pairs, width 32 | 2.57, 3.86 | 3.69 | 0.992 |
| 10000 pairs, width 32 | 2.83, 4.00 | 3.98 | 0.992 |

The proxy's own score rises at every n. The gold score peaks and falls. Better proxies, with more labels or more
capacity, peak later and higher, which is the direction Gao et al. report for larger reward models. The exact KL tilt
against the same proxies peaks too (for example d = 1.57, gold 2.69 for the linear proxy), and d(alpha - beta log d)
fits it with R^2 0.96 to 0.99. For the best proxies the tilt curve is cut off by the importance-sampling limit close
to its peak.

### DPO on SmolLM2-135M-Instruct (AMD EPYC 7R13, 8 threads, torch 2.15.0)

400 of C6's synthetic pairs. In each pair the chosen and rejected responses are 22 tokens long, and 20 of those
tokens are the same. Batch 8, lr 5e-6, beta 0.1, 60 steps, reference log-probs cached once. The table measures the
first 32 training pairs, and log-prob changes are in nats against the reference.

| Method, after 60 steps | DPO loss | Pair accuracy | Change in log pi(y_w) | Change in log pi(y_l) | Chosen pairs below reference |
|---|---|---|---|---|---|
| DPO | 0.465 | 0.78 | -4.25 | -10.37 | 28 of 32 |
| DPO-Positive, lambda 5 | 0.494 | 0.84 | +8.91 | +3.91 | 0 of 32 |
| DPO + 0.1 NLL on y_w | 0.549 | 0.91 | +14.42 | +11.06 | 0 of 32 |

Plain DPO shows likelihood displacement on a real model. The loss fell from 0.693 to 0.465, and the pair accuracy
rose to 0.78. Over the same steps the chosen response lost 4.25 nats on average. By step 30 every one of the 32
measured chosen responses was below its reference probability. Both patches keep the chosen above the reference. They
also lift the rejected response, which shares 20 of its 22 tokens with the chosen one. This is one run per method,
with no repeats over seeds. It took 590 s, 360 s and 263 s per method on a heavily shared machine, so the timings are
rough.

## What was not measured

- Nothing here runs at the post's scale. The paper numbers the post quotes (Tülu 3's table, RLAIF vs RLHF win rates,
  Gao et al.'s coefficients) were not reproduced. The Goodhart toy copies the curve shapes, not the paper's numbers.
- The KL tilt in script 06 is the exact target of KL-regularised RL. It is not a PPO run. The post's claim that RL
  optimises and over-optimises more slowly than best-of-n is about the optimiser, so it is not tested.
- PPO itself, the reward-model architecture on a real model, and SFT are in [C6](../../c06-post-training/), not here.
- Online versus offline preference data is not measured. A first toy attempt gave no clean, stable effect, so it was
  dropped rather than reported.
- Length bias (Pathology 3, Nuance 04), sycophancy, Constitutional AI and LLM judges are not built.
- No GPU numbers. The post is not about speed, and every experiment fits on a CPU.
