# Lab: Teaching a Model What We Prefer

Code companion for the theory post
[Teaching a Model What We Prefer: From RLHF to DPO](https://srini-anirudh.github.io/blogs/teaching-a-model-what-we-prefer/).

The post makes a chain of claims. Annotator noise gives the Bradley–Terry sigmoid. The KL-regularised objective has a
closed-form optimum. Inverting that optimum gives DPO. DPO's loss sees only a difference, so the winner can lose
probability. A proxy reward fails under optimisation pressure. Offline data leaves most of the space unconstrained.
This lab checks each claim on an example small enough to print, then repeats the displacement claim on a real
135M-parameter model.

Most of it uses finite sets of "responses" instead of a language model. On a finite set, the partition function
`Z(x)` can be computed exactly. You can then compare DPO's answer with the true optimum, which no real model allows.

The full SFT, reward-model, DPO and GRPO pipeline on a language model is built in
[C6: post-training](../../c06-post-training/). This lab does not rebuild it. It imports C6's sequence log-probs,
batch layout, DPO loss and tiny test Llama.

## Layout

```
labs/teaching-a-model-what-we-prefer/
├── prefopt/
│   ├── bt.py          # annotator simulation, Bradley-Terry loss, fitting scores
│   ├── tilt.py        # the KL-regularised objective, its closed-form optimum, exact best-of-n
│   ├── dpo.py         # implicit reward, DPO loss and weight, a softmax policy, DPO-Positive, RPO, the loss family
│   ├── goodhart.py    # gold reward, proxy reward model, over-optimisation curves, Gao et al.'s functional forms
│   ├── lm.py          # imports from C6: sequence log-probs, pair batches, tiny Llama
│   └── report.py      # writes results/ with a hardware and command header
├── scripts/           # one script per claim, listed below
├── tests/
└── results/           # the real outputs of every script
```

## Setup

From the repository root, after `pip install -e ".[dev]"`. Only `08_displacement_lm.py` and the `network` test
download anything: SmolLM2-135M-Instruct, about 270 MB.

## Run everything

```bash
python -m pytest labs/teaching-a-model-what-we-prefer -m "not slow and not gpu and not network"   # about 20 s
python -m pytest labs/teaching-a-model-what-we-prefer -m "network or slow"

python labs/teaching-a-model-what-we-prefer/scripts/01_bradley_terry.py     # about 30 s
python labs/teaching-a-model-what-we-prefer/scripts/02_kl_tilt.py           # about 15 s
python labs/teaching-a-model-what-we-prefer/scripts/03_dpo_identity.py      # about 20 s
python labs/teaching-a-model-what-we-prefer/scripts/04_overoptimization.py  # about 10 minutes on a loaded machine
python labs/teaching-a-model-what-we-prefer/scripts/05_displacement_toy.py  # about 40 s
python labs/teaching-a-model-what-we-prefer/scripts/06_loss_family.py       # a second
python labs/teaching-a-model-what-we-prefer/scripts/07_online_vs_offline.py # a few minutes
python labs/teaching-a-model-what-we-prefer/scripts/08_displacement_lm.py   # downloads SmolLM2-135M-Instruct; about 15 minutes on a loaded machine
```

Times are on AMD EPYC 7R13 with 8 threads, on a shared machine. Everything runs on CPU.

## Which script covers which section

| Post section | Script |
|---|---|
| From comparisons to a scalar field: Gumbel to sigmoid, probit, scale and noise, shift invariance, the self-annealing gradient | `01_bradley_terry.py` |
| What Bradley–Terry quietly assumes: a single scalar orders everything | `01_bradley_terry.py` (part 6) |
| The KL-regularised objective: the closed form, `beta log Z`, same support, Figure 06's beta sweep | `02_kl_tilt.py` |
| The DPO identity, derived slowly; reading the result correctly | `03_dpo_identity.py` |
| What the DPO gradient actually does: the weight and the gradient formula | `03_dpo_identity.py` (parts 4 and 5) |
| Pathology 2 · Goodhart's law, measured; Pathology 3 · Length; Figure 05 | `04_overoptimization.py` |
| The failure mode hiding in plain sight (likelihood displacement); Figure 08; DPO-Positive and RPO | `05_displacement_toy.py`, `08_displacement_lm.py` |
| Nuance 04 · the objective is length-confounded by construction | `08_displacement_lm.py` (part 1) |
| The offline preference family; Figure 09 | `06_loss_family.py` |
| Online versus offline; Nuance 02 (ii) | `07_online_vs_offline.py` |

## Results

All numbers below are from AMD EPYC 7R13, 8 threads, torch 2.15.0, CPU only. The machine was shared and heavily loaded,
so treat timings as orders of magnitude. The results do not depend on load. Every file in `results/` starts with its
hardware and command. Seeds are fixed.

Tests (same hardware): 28 passed offline (`not slow and not gpu and not network`), 4 passed with `slow or network`
(three scripts re-run end to end, and one download of the SmolLM2 tokenizer).

### Derivations checked on small examples

| Claim in the post | Measurement | Result |
|---|---|---|
| Gumbel noise gives `sigmoid(delta)` (`01`) | 200,000 simulated annotators at each of 8 reward gaps; largest gap between the choice rate and the sigmoid | 0.0016 (one standard error at p = 0.5 is 0.0011) |
| Gaussian noise gives the probit (`01`) | Same, against `Phi(delta / sqrt 2)` | 0.0010 |
| Logistic and probit are close after rescaling (`01`) | Best `sigmoid(c d)` against the probit over [-6, 6] | c = 1.203, largest gap 0.0095 |
| Noise scale and reward scale are one knob (`01`) | Fit 20 scores from 40,000 comparisons at noise 1 and noise 2 | Fitted scores are 0.990 and 0.501 times the true ones |
| Shift invariance (`01`) | Add 7.25 to every score | Loss 0.480969 before and after (differs by 6e-8, float32 rounding) |
| Self-annealing gradient (`01`) | Autograd against `-sigmoid(-delta)` | Equal to the four printed digits at 7 values of delta |
| A preference cycle cannot be fit (`01`) | A beats B, B beats C, C beats A, 80% each | The best fit predicts 0.500, 0.500, 0.500 |
| Closed-form optimum of the KL objective (`02`) | Adam on 12 completions, 3 values of beta, against `ref * exp(r / beta) / Z` | Largest probability difference 1.7e-5 at beta = 0.2 and below 1e-12 at beta = 0.6 and 2; optimal value equals `beta log Z` |
| Nothing beats the tilt (`02`) | 20,000 random distributions at beta = 0.6 | Smallest shortfall 0.084 |
| Same support (`02`) | Completion with reference probability 0 and reward 10 | Probability exactly 0 at beta = 0.01. With reference probability 1e-9 it gets 0.0089 at beta = 0.6 |
| `beta log Z` is the missing constant (`03`) | `r` minus `beta log(pi* / ref)` over 10 completions | Constant to 12 digits, equal to `beta log Z` = 0.318655 |
| Z cancels in Bradley–Terry (`03`) | All 45 pairs, true reward against recovered reward | Largest probability difference 1.1e-16 |
| DPO finds the same policy as reward model plus tilt (`03`) | Tabular policy, DPO on Bradley–Terry-weighted pairs, 3000 steps | Total variation to `pi*` is 1.1e-16, from 0.54 at the start |
| The gradient formula (`03`) | Autograd against `-beta w (grad log pi_w - grad log pi_l)` on a 22,688-parameter random Llama | Relative difference 2.8e-7 |

### Failure modes, at toy scale

| Claim in the post | Measurement | Result |
|---|---|---|
| Over-optimisation, best-of-n (`04`) | Pool of 4000 completions, proxy MLP trained on 1000 labels (held-out pairwise accuracy 0.718). Gold gain against `d = sqrt(KL)` | Rises to +0.723 at d = 1.78, then falls to -0.334 at d = 3.55 while the proxy gain keeps rising to 19.6. Fit `d (a - b d)`: R^2 = 0.990 |
| Over-optimisation, KL-regularised optimum (`04`) | Same proxy, exact tilt swept over beta | Gold peaks at +0.466 at d = 1.23, ends at -5.150 at d = 5.50. Fit `d (a - b log d)`: R^2 = 0.964 |
| Length rides along (`04`) | Same runs | Length gain grows from 0 to 2.8 (best-of-n) and 4.7 (tilt). Gold's length term peaks at +1 |
| Sweep: width x labels x 2 seeds (`04`) | 18 proxies | Under the tilt, 8 of 18 proxies end below their gold peak. Accuracy rises with labels (0.74, 0.80, 0.88) and falls with width (0.88, 0.79, 0.75) |
| Likelihood displacement (`05`) | One pair in a softmax over 50 responses, 20 embedding draws, beta = 0.1. Fraction of draws where `log pi(y_w)` ends below the reference | cos(E_w, E_l) = 0: 5%. 0.5: 50%. 0.8: 85%. 0.95: 95%. 0.99: 95%. The DPO loss falls in every draw |
| Median change in `log pi(y_w)` (`05`) | Same | +4.0 at cos 0 and -21.6 at cos 0.99 |
| DPO-Positive and RPO fix it (`05`) | Same, lambda = 50 for DPO-Positive, 1.0 for RPO | 0 of 20 draws fall below the reference at every cos. Median change at cos 0.99 is +0.95 and +3.9 |
| Likelihood displacement in a real model (`08`) | SmolLM2-135M-Instruct, 16 pairs, full fine-tuning, 40 steps, AdamW 5e-6, beta 0.1. Rejected answer differs from the chosen one by one fact | `log pi(y_w)` change -1.26 and `log pi(y_l)` change -31.7. 11 of 16 winners end below the reference. Loss falls from 0.693 to 0.065 |
| Same, unrelated rejected answer (`08`) | Rejected answer is a refusal | `log pi(y_w)` change +2.05, `log pi(y_l)` change -119.5. No winner falls |
| Same, near pair with an NLL term (`08`) | RPO weight 1.0 | `log pi(y_w)` change +8.43. No winner falls |
| Summed log-prob mixes in length (`08`) | 16 correct answers, each with and without one added true sentence | Mean 11 and 28 tokens. Summed log-prob -8.67 and -52.92. Per-token mean -0.770 and -1.894. The longer one is lower in 16 of 16 prompts under both measures |
| The loss family (`06`) | Autograd gradient of 7 losses on a margin grid | DPO, SimPO and ORPO gradients never reach zero. SLiC-HF and RRHF are exactly zero once the margin passes their threshold. IPO's gradient changes sign at its target margin. KTO's vanishes at both ends |
| Offline versus online (`07`) | 4000-completion pool, log-linear policy, DPO against the true reference, 3000 Adam steps in total. Offline: one round of labels from the reference. Online: the same label budget over 10 rounds from the current policy. 5 seeds | Gold gain at 500, 2000 and 8000 labels: offline -7.08, -6.61, -6.63. Online -1.75, +0.75, +0.91. The exact KL optimum gives +1.59 |
| Where offline DPO puts its mass (`07`) | Same | Offline policy mass on completions that never appeared in a labelled pair: 1.000, 0.999, 0.996. Online: 0.090, 0.084, 0.020 |

## What was not measured

- The paper's scale. Gao et al. fitted scaling laws with reward models from 3M to 3B parameters. This lab has a
  6-dimensional pool and MLP proxies of width 4 to 64. It shows the rise-then-fall shape and the two functional forms
  fitting. It does not show their coefficient trends. In the sweep, wider proxies were worse, not more robust, because
  they fit the labels less well.
- Best-of-n against RL speed. The tilt is the target RL aims at, not RL. No PPO was run.
- The online-versus-offline gap is a toy. The offline run goes to convergence with no early stopping, and an
  early-stopped offline run would do better. This is a statement about what the DPO loss leaves unconstrained, not a
  reproduction of Xu et al., Tajwar et al. or Tang et al.
- Likelihood displacement in `08` uses 16 hand-written questions and one seed. The 11 of 16 count is one run.
- Reward-model pathologies that need human data: sycophancy, calibration against real annotators, heterogeneous
  annotators.
- The Tülu 3 comparison table, SimPO, KTO and ORPO as trained models, Constitutional AI and RLAIF. `06` draws their
  loss shapes only.
- Any GPU run. Nothing here needs one.
