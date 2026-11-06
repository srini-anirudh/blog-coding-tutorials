# Lab: Aligning Vision-Language Models

Code companion for the theory post
[Aligning Vision-Language Models: Preferences, Hallucination and Safety](https://srini-anirudh.github.io/blogs/aligning-vision-language-models/).

The post argues that a multimodal preference signal is only useful if it depends on the image. This lab makes that
argument run. Most of it uses a tiny vision-language model trained from scratch on a toy world where every fact is
known. One script uses a real model, SmolVLM-256M-Instruct, on the synthetic scenes from C9.

What you can run:

- The split log P(y | x, I) = log P(y | x) + g(y; I | x) on a real VLM, plus visual contrastive decoding and image
  reliance along a long description.
- Two reward models with the same architecture. One learns a text shortcut and ignores the image. The image-ablation
  audit catches it.
- Standard multimodal DPO, its no-image ablation and mDPO. The DPO margin is split into a text part and an image part.
- The shared-prefix cancellation in the DPO gradient, Dense DPO weights and image-contrast token weights.
- An AI-judge loop in which errors the judge misses get no corrective signal.
- CHAIR scores that improve as captions get shorter.
- Abstention with coverage, risk and the threshold c / (1 + c).

C9 ([`c09-vlm-post-training`](../../c09-vlm-post-training)) already builds image-conditioned DPO on SmolVLM, five
kinds of hallucination negatives and a text-only probe of preference data. This lab does not rebuild those. It reuses
C9's scene generator and SmolVLM helpers, and it focuses on what the theory post adds.

## The toy world

A scene is a set of objects (car, person, dog, door, tree, sign, bike, cat). Each object has a colour and a count.
Scenes come from three templates (street, park, house), so some objects co-occur and cars are usually white. That is
the language prior. The "image" is not pixels. It is an 8 x 8 tensor, one row per object slot, holding presence, a
colour one-hot and a count one-hot, buried in Gaussian noise (sd 0.8). A response is a list of claims such as
`two red car ;`. Style words such as `lovely` or `shiny` can appear in front of a claim. They say nothing about the
image.

The policy is a 2-layer causal transformer (width 64, about 107k parameters). The 8 image slots are prefix tokens.
It is trained by SFT on 40,000 descriptions. One in ten SFT examples sees the null image, so log P(y | x, null) is a
real language prior for this model. After SFT the policy hallucinates: CHAIR_i is 0.172, and 32.5% of its colour
claims are wrong.

The toy covers three of the post's seven hallucination kinds: object existence, attributes (colour) and counts.

## Layout

```
labs/aligning-vision-language-models/
├── groundpref/
│   ├── world.py       # scenes, symbolic images, crops and edits, claim tokens, corruptions
│   ├── model.py       # TinyVLM, per-token and sequence log-probs, sampling
│   ├── objectives.py  # DPO, mDPO conditional term and anchor, margin split, Dense DPO weights
│   ├── pairs.py       # 'style' pairs (text separates them) and 'fact' pairs (only the image does)
│   ├── prefopt.py     # DPO training variants and their evaluation
│   ├── reward.py      # Bradley-Terry reward model and the image-ablation audit
│   ├── metrics.py     # CHAIR, recall, error rates, coverage, risk, effective reliability, ECE
│   ├── train.py       # SFT (cached in runs/sft.pt) and description scoring
│   └── report.py      # results files that start with the hardware and the command
├── scripts/           # one script per experiment, listed below; run_all.sh runs them in order
├── results/           # real outputs (.txt and .json)
└── tests/
```

## Setup

From the repository root, after `pip install -e ".[dev]"`. The scripts use `OMP_NUM_THREADS` threads, or at most 8.
Checkpoints go to `runs/`, which git ignores. The first script that needs the toy policy trains it (about 2 minutes on
8 CPU threads) and caches it there.

## Run everything

```bash
# Tests
python -m pytest labs/aligning-vision-language-models -m "not slow and not gpu and not network"
python -m pytest labs/aligning-vision-language-models -m network      # downloads SmolVLM-256M

# Every experiment, in order. On 8 heavily shared CPU threads this took about 90 minutes,
# most of it in prior_split_smolvlm.py. Each toy script alone takes a few minutes.
labs/aligning-vision-language-models/scripts/run_all.sh

# Or one at a time
python labs/aligning-vision-language-models/scripts/prior_split_smolvlm.py --scenes 80 --describe 30
python labs/aligning-vision-language-models/scripts/chair_silence.py
python labs/aligning-vision-language-models/scripts/reward_audit.py --seeds 0 1 --steps 1500
python labs/aligning-vision-language-models/scripts/mdpo.py --seeds 0 1
python labs/aligning-vision-language-models/scripts/token_credit.py
python labs/aligning-vision-language-models/scripts/judge_loop.py --seeds 0
python labs/aligning-vision-language-models/scripts/abstention.py
```

## Post sections and scripts

| Post section | Script | What it shows |
|---|---|---|
| Language priors and visual evidence (Figure 2) | `prior_split_smolvlm.py` | g(y; I \| x) on SmolVLM, contrastive decoding under weak evidence, g along a description |
| How hallucination is measured (CHAIR's blind spot) | `chair_silence.py` | CHAIR falls as captions are truncated, recall falls faster |
| Reward models that must look at the image (Figure 4) | `reward_audit.py` | image-ablation audit of a shortcut reward model and a grounded one |
| Multimodal DPO and the ignored image (Figure 5) | `mdpo.py` | DPO versus no-image DPO versus mDPO; margin split; anchor and conditional ablations |
| How hard should a negative be? | `mdpo.py` (`mdpo`, `mdpo_crop_large`, `mdpo_unrelated`) | small crop, full crop and unrelated image as the rejected image |
| Fine-grained feedback (Figure 7) | `token_credit.py` | exact prefix cancellation, Dense DPO, image-contrast token weights, minimal versus rewritten pairs |
| AI feedback and its failure modes (Figure 8) | `judge_loop.py` | iterative DPO against a judge that misses count errors and likes detail |
| Knowing when not to answer (Figure 10) | `abstention.py` | coverage, risk, effective reliability, c / (1 + c) with and without calibration |

## Results

Every number below was measured on an AMD EPYC 7R13, 8 CPU threads, torch 2.15.0, with other jobs running on the
machine. Timings were not measured: the machine was too loaded for them to mean anything. Raw output is in
`results/`, under the name of the script. Toy-model numbers are means over seeds where the table says so.

### Language prior and visual evidence: SmolVLM-256M (`prior_split_smolvlm.txt`)

Part A asks "What color is the <shape>? Answer with one word." about 80 scenes. It scores the four palette colours.
The prior P(y | x) is measured with a mid-grey image.

| Measurement | Value |
|---|---|
| Accuracy with the image | 0.700 |
| Accuracy with the grey image (the prior alone) | 0.175 |
| Grey-image answer | "Red" 72.5% of the time, "Blue" 27.5%, never green or yellow |
| Mean g(y; I \| x) for the true colour | +1.37 nats |
| Mean g(y; I \| x) for the wrong colours | -4.60 nats |
| Mean KL(P(. \| x, I) ‖ P(. \| x)) over the four answers | 1.38 nats |

The image moves the answer a long way from the prior. Without it, the model says "red" almost regardless of the
question.

Then the evidence is weakened by blending the image toward grey (w = 1 is the image). Visual contrastive decoding
scores each answer as (1 + a) log p(y | I) - a log p(y | I~), where I~ is the same weak image plus pixel noise
(sd 80). This is the answer-level version of VCD, over four candidates.

| Blend w | a = 0 | a = 0.5 | a = 1 | a = 2 | a = 4 |
|---|---|---|---|---|---|
| 1.0 | 0.700 | 0.675 | 0.625 | 0.500 | 0.375 |
| 0.3 | 0.625 | 0.575 | 0.562 | 0.500 | 0.363 |
| 0.15 | 0.525 | 0.550 | 0.575 | 0.600 | 0.587 |
| 0.08 | 0.450 | 0.450 | 0.450 | 0.438 | 0.438 |

Contrast helped only when the evidence was weak but present (w = 0.15: 0.525 to 0.600 at a = 2). With clear evidence,
it hurt, and large a hurt most. With almost no evidence (w = 0.08) it did nothing. This is one small model on 80
synthetic questions, so read it as the shape of the effect, not its size.

Part B asks for 30 greedy descriptions of up to 64 tokens. It then scores every generated token with the image and
with the grey image. g_t = log p(y_t | I, y<t) - log p(y_t | grey, y<t).

| Token positions | Mean g_t (nats) | Colour and shape words | Share not in the scene |
|---|---|---|---|
| 0-15 | +0.489 | 13 | 0.154 |
| 16-31 | +0.255 | 26 | 0.000 |
| 32-47 | +0.144 | 43 | 0.023 |
| 48-63 | +0.108 | 44 | 0.023 |

Image reliance falls steadily along the response, from 0.49 to 0.11 nats per token. That is the pattern M3ID
measured. We did not see hallucinated colour and shape words concentrate late, as LURE reported. Only 4 of 126
mentions were wrong, which is too few to say anything about where errors fall.

### CHAIR rewards silence (`chair_silence.txt`)

The SFT policy describes 600 held-out scenes. We keep only the first k claims.

| First k claims | CHAIR_i | CHAIR_s | Object recall | Claims per response |
|---|---|---|---|---|
| 0 (empty caption) | 0.000 | 0.000 | 0.000 | 0.00 |
| 1 | 0.150 | 0.150 | 0.282 | 1.00 |
| 2 | 0.178 | 0.298 | 0.524 | 1.93 |
| 3 | 0.176 | 0.343 | 0.653 | 2.39 |
| all | 0.172 | 0.343 | 0.688 | 2.51 |

An empty caption scores perfectly. CHAIR_s rises with every extra claim, even though the claims themselves do not get
worse (CHAIR_i stays between 0.15 and 0.18). Any comparison of hallucination rates needs recall next to it.

### Reward-model audit (`reward_audit.txt`)

Two Bradley-Terry reward models with the same architecture and the same image input. Each is trained on 30,000
pairs for 1,500 steps, two seeds. RM-S is trained on 'style' pairs, where text alone separates winner and loser.
RM-G is trained on 'fact' pairs, where only the image does. Both are audited on 1,500 held-out pairs of each kind.

| Reward model | Accuracy, style pairs | Accuracy, fact pairs | Fact pairs: object / colour / count | Image sensitivity S | Swap flip rate |
|---|---|---|---|---|---|
| RM-S (style pairs) | 1.000 | 0.485 | 0.519 / 0.495 / 0.444 | 0.0002 | 0.033 |
| RM-G (fact pairs) | 0.579 | 0.772 | 0.895 / 0.678 / 0.747 | 1.23 | 0.214 |

S is E\|r(I, x, y) - r(null, x, y)\| divided by the spread of r(I, x, y). The swap flip rate takes colour pairs the
model ranked correctly, edits the image so the rejected colour becomes true, and counts how often the ranking flips.

RM-S is perfect on its own kind of pair and at chance on pairs that need the image. Its scores do not move when the
image is removed (S below 0.001 in both seeds). The audit catches it without any new labels. RM-G reads the image,
but its swap flip rate is only 0.214 (0.302 and 0.126 for the two seeds). Its colour judgements still lean on the
colour prior when the evidence is noisy. A reward model that passes the removal test can still fail the swap test.

### Multimodal DPO and the ignored image (`mdpo.txt`)

All variants start from the same SFT policy. They train for 300 steps (batch 64, beta 0.1) on the same 4,000
'style' pairs: the winner is plain and correct, the loser has one error and style words. Two seeds each. "Fact-pair
preference" is the share of 1,500 held-out pairs, differing in one fact and nothing else, where the policy gives
the correct response the higher log-probability. beta D_text and beta D_image split the training margin, measured on
600 training pairs.

| Variant | CHAIR_i | Recall | Colour error | Count error | Fact-pair preference | beta D_text | beta D_image |
|---|---|---|---|---|---|---|---|
| SFT (start) | 0.172 | 0.688 | 0.325 | 0.217 | 0.876 | – | – |
| DPO | 0.205 | 0.416 | 0.312 | 0.258 | 0.770 | 7.19 | -0.02 |
| DPO, images removed | 0.192 | 0.463 | 0.376 | 0.250 | 0.755 | 7.78 | -2.61 |
| DPO + anchor | 0.182 | 0.717 | 0.325 | 0.232 | 0.847 | 4.77 | 0.17 |
| DPO + conditional term (no anchor) | 0.067 | 0.147 | 0.269 | 0.190 | 0.743 | 3.97 | 3.38 |
| mDPO (DPO + conditional + anchor), small crop | 0.158 | 0.674 | 0.328 | 0.216 | 0.839 | 4.41 | 0.15 |
| mDPO, crop every object | 0.165 | 0.666 | 0.325 | 0.222 | 0.819 | 2.93 | 1.45 |
| mDPO, unrelated image | 0.186 | 0.729 | 0.323 | 0.228 | 0.846 | 4.01 | 0.21 |

What this shows:

- Standard DPO takes its whole margin from text. beta D_text is 7.19 and beta D_image is -0.02. Removing every image
  reaches a similar loss (0.004 against 0.006) and a similar CHAIR_i (0.192 against 0.205). That is the shape of
  mDPO's no-image experiment, at toy scale.
- Standard DPO also made the policy worse on pairs only the image can separate (0.876 to 0.770), and it shortened
  descriptions (recall 0.688 to 0.416).
- mDPO with a small crop gave the lowest CHAIR_i among the variants that kept recall (0.158, recall 0.674).
- The conditional term without the anchor reached CHAIR_i 0.067 by saying much less (recall 0.147). Here the anchor
  mattered more than in mDPO's own ablation, where removing the conditional term hurt more. We did not reproduce that
  ordering.
- Negative difficulty: on CHAIR_i the small crop (0.158) beat the full crop (0.165) and the unrelated image (0.186),
  the same order as mDPO's crop ablation. On fact-pair preference the unrelated image was as good as the small crop.
  The differences are small and come from two seeds.

### Fine-grained feedback (`token_credit.txt`)

Part 1 uses the SFT policy on 60 pairs that differ in one colour word.

| Position relative to the edited word | Mean norm of the net push v_t | Max |
|---|---|---|
| -4 or earlier | 0.0000 | 0.0000 |
| -1 | 0.0000 | 0.0000 |
| 0 (the edited colour word) | 23.11 | 42.27 |
| +1 (the object word right after it) | 19.00 | 38.83 |
| +2 (the separator) | 0.0005 | 0.0068 |
| +3 and later | 0.32 to 0.64 | 6.89 |

The shared prefix cancels exactly: the per-token gradients are bit-identical. Tokens after the edit see a different
context, so they only nearly cancel. The test `test_shared_prefix_gradients_cancel_exactly` checks the exact part.

Cosine between the whole-pair gradient and the gradient of the edited claim alone: minimal edit 0.997, rewritten
winner (same facts, new order) 0.689. With Dense DPO weights (gamma = 5): minimal 1.000, rewritten 0.971.

Part 2, image-contrast token weights |log p(y_t | I) - log p(y_t | noised I)|: objects 1.38, colours 1.11, counts
0.74, style words 0.24, separators 0.00, end of sequence 2.30. The weight follows the content words. The end token is
high because the noised image changes how many objects the model thinks there are.

Part 3, DPO on 500 pairs, 100 steps, three seeds: minimal-edit winners gave colour error 0.320 and implicit-reward
accuracy 0.778. Rewritten winners gave 0.339 and 0.764. The direction matches the post, but the gap is small at this
scale. It does not reproduce RLHF-V's 1.4k-versus-10k comparison, which this lab does not attempt.

### AI-judge loop (`judge_loop.txt`)

Six rounds of sample, judge, DPO, one seed. The judge catches object and colour errors with probability 0.9. It
catches count errors with probability 0.15 (count-blind) or 0.9 (claim-checking). It adds a bonus per claim.
Error rates are per claim, on 600 held-out scenes, greedy decoding.

| Judge | Detail bonus | CHAIR_i | Colour error | Count error | Claims per response | Recall |
|---|---|---|---|---|---|---|
| SFT (round 0) | – | 0.172 | 0.325 | 0.217 | 2.51 | 0.688 |
| count-blind | 0.5 | 0.135 | 0.289 | 0.217 | 1.64 | 0.470 |
| claim-checking | 0.5 | 0.138 | 0.241 | 0.154 | 0.74 | 0.211 |
| count-blind | 1.0 | 0.354 | 0.334 | 0.295 | 6.87 | 0.892 |
| claim-checking | 1.0 | 0.390 | 0.337 | 0.274 | 6.49 | 0.914 |

With the smaller bonus, colour errors fell under both judges. Count errors stayed flat (0.217) under the judge that
misses them, and fell to 0.154 under the judge that checks them. Both loops also shortened the answers, more so with
the stricter judge. With the larger bonus, the preference for detail dominates: answers grow to six or seven claims
and every error rate rises, including the ones the judge detects. This is a toy loop with a simulated judge, not a
model of any paper's pipeline.

### Abstention (`abstention.txt`)

"What colour is the X?" on 8,000 test questions. A quarter are unanswerable because the object's evidence was cropped.
Score +1 right, -c wrong, 0 abstain.

| Model | ECE | Mean confidence, unanswerable | Best tau at c = 1 | ER at best tau | ER at tau = 0.5 | Best tau at c = 4 | ER at best | ER at tau = 0.8 |
|---|---|---|---|---|---|---|---|---|
| Overconfident (answerable data only) | 0.166 | 0.622 | 0.71 | +0.158 | +0.119 | 0.96 | +0.031 | -0.118 |
| Trained to abstain (cropped examples labelled "can't tell") | 0.022 | 0.287 | 0.50 | +0.213 | +0.213 | 0.85 | +0.072 | +0.063 |

For the calibrated model, the best threshold sits at c / (1 + c): exactly 0.50 at c = 1, and 0.85 against 0.80 at
c = 4. For the overconfident model, the formula picks a threshold that loses reliability (at c = 4 it is worse than
abstaining on everything). Neither model reaches useful coverage at 5% risk (0.000 and 0.039), because the noisiest
images are hard for both.

## What was not measured

- No result from a paper is reproduced at its scale. The toy model and data are ours. Numbers such as mDPO's
  MMHal-Bench scores, RLHF-V's 34.8% reduction or VL-RewardBench accuracies are quoted in the post and not measured
  here.
- No benchmark from the post (CHAIR on COCO, POPE, MMHal-Bench, AMBER, HallusionBench) is run. `chair_silence.py`
  computes CHAIR on the toy world only.
- The toy covers object, colour and count errors. Spatial relations, text in the image, actions and knowledge
  presented as evidence are not modelled.
- Multimodal safety (typographic jailbreaks, injection through screenshots, adversarial images) has no script. It
  would need a capable model and attack data that do not fit this lab.
- The SmolVLM part uses C9's synthetic scenes, not natural photos. Contrastive decoding is applied at the answer
  level over four candidate colours, not token by token during generation.
- Seeds are few (one to three). Differences of a few hundredths in the toy tables are within seed noise.
- Nothing here needs a GPU, so nothing ran on one.
