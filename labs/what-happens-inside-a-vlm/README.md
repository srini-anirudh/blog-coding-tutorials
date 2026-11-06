# Lab: What Happens Inside a VLM

This lab runs the interpretability tools from the post on a real vision-language model. The model is
SmolVLM-256M-Instruct. The images are synthetic: coloured circles, squares and triangles on a grey canvas. Because
the lab draws every image, it knows exactly which image tokens cover each object. So "ablate the circle's tokens" or
"block the circle's tokens from the question" is a precise experiment, not an estimate.

Post: https://srini-anirudh.github.io/blogs/what-happens-inside-a-vlm/

The package `vlmscope` is small. One custom attention function can record attention weights and knock out routes.
Forward hooks capture, patch and ablate the residual stream. Everything else is a script that asks one question from
the post and writes its answer to `results/`.

## The model organism

| Part | SmolVLM-256M-Instruct |
|---|---|
| Vision encoder | SigLIP, 12 layers, 512 px input, 16 px patches, 1024 patches |
| Connector | pixel shuffle (each 4 x 4 block of patches becomes one 12,288-dim vector), then one linear map to 576 dims |
| Image tokens | 64 per image, an 8 x 8 grid in raster order (image splitting is turned off) |
| Language model | SmolLM2-135M, 30 layers, 9 heads, 1D RoPE |
| Layout | image before the question, so image tokens never see the question |

This is not LLaVA-1.5, the model most findings in the post come from. It is much smaller, it has 64 image tokens
rather than 576, and each token covers a 64 x 64 pixel block. The post's own warning applies: a finding on one model
organism does not automatically transfer to another. Where the lab agrees or disagrees with a paper, read it as one
more data point on one more model, not as a replication.

## Layout

```
labs/what-happens-inside-a-vlm/
├── vlmscope/
│   ├── scenes.py      # shapes, rendering, exact object masks on the token grid, minimal pairs, BlindTest-style scenes
│   ├── model.py       # loads the VLM, builds prompts, finds image / question / last positions, logit lens
│   ├── intervene.py   # attention recording and knockout, residual capture and patching, embedding ablation
│   ├── metrics.py     # MG, ML, restoration, relative change, forced choice
│   ├── probe.py       # logistic probes, ridge reconstruction, k-NN overlap
│   └── report.py      # results files with a hardware line and the command
├── scripts/           # 01 to 07, one experiment each, plus run_all.sh
├── results/           # real outputs, .txt for reading and .json for plotting
└── tests/
```

## Setup and run

From the repository root, after `pip install -e ".[dev]"`. The first run downloads SmolVLM-256M-Instruct and
SmolLM2-135M-Instruct (about 1.3 GB together). Set `OMP_NUM_THREADS` on a shared machine. Scripts use a GPU if one
is visible and the CPU otherwise.

```bash
# Tests.
python -m pytest labs/what-happens-inside-a-vlm -m "not slow and not gpu and not network"   # offline, a tiny random Llama
python -m pytest labs/what-happens-inside-a-vlm -m network                                 # the real model

# Run everything. Each script writes results/<script>.txt and .json.
bash labs/what-happens-inside-a-vlm/scripts/run_all.sh

# Or one at a time. --n sets the number of scenes, --seed the scene generator.
python labs/what-happens-inside-a-vlm/scripts/01_blind_baseline.py
python labs/what-happens-inside-a-vlm/scripts/02_logit_lens.py
python labs/what-happens-inside-a-vlm/scripts/03_attention_sinks_ablation.py
python labs/what-happens-inside-a-vlm/scripts/04_knockout.py
python labs/what-happens-inside-a-vlm/scripts/05_patching.py
python labs/what-happens-inside-a-vlm/scripts/06_probe_vs_readout.py
python labs/what-happens-inside-a-vlm/scripts/07_connector.py
```

Every answer is scored by forced choice. The script reads the next-token logits at the last position and compares
the first tokens of the candidate answers, such as " Red", " Blue", " Green" and " Yellow". Counts are scored as the
digit after a space token, because that is how the model writes them (" 3").

## Which script covers which section of the post

| Post section | Script | What it measures |
|---|---|---|
| Does the model use the image | `01_blind_baseline.py` | MG and ML from MMStar, plus a per-example blank-image check |
| What visual tokens encode: inside the language model | `02_logit_lens.py` | logit lens at image positions and at the answer position |
| Tools: attention maps; How the image reaches the answer: collapse, visual sinks | `03_attention_sinks_ablation.py` | attention share by layer, sink tokens, their norms |
| What visual tokens encode: object information is local; Putting interpretability to work: pruning | `03_attention_sinks_ablation.py` | mean ablation of object, random, sink and high-norm tokens; FastV |
| How the image reaches the answer: a route in stages | `04_knockout.py` | attention knockout of five routes, sliding windows of layers |
| Tools: activation patching | `05_patching.py` | restoration map on image minimal pairs |
| Where simple visual tasks fail; checklist step 2 | `06_probe_vs_readout.py` | encoder, connector and LM probes against the VLM's own answer |
| The connector as a bottleneck | `07_connector.py` | reconstruction loss per token, k-NN overlap |

## Results

All numbers below come from `results/`, on AMD EPYC 7R13, 8 threads, torch 2.15.0, transformers 5.22.2. Sample sizes
are small (16 to 240 prompts per experiment), so read differences of a few points as noise.

### The headline numbers

| Experiment | Result (AMD EPYC 7R13, 8 threads) |
|---|---|
| Blind baseline, colour questions: with image / image removed / base LM | 86.7 / 38.3 / 28.3 %, so MG +48.3, ML 10.0 |
| Blind baseline, "Is there a dog?": with image / blank image / image removed / base LM | 100 / 100 / 0 / 0 % |
| Logit lens at the last position: P(" red") peaks / P(" Red") peaks | 0.82 at layer 23 / 0.90 at layer 29 (of 0 to 29) |
| Logit lens at object image tokens: own colour in the top 5 | 0 % up to layer 24, peak 10.0 % at layer 28 |
| Last position's attention to the 64 image tokens (74 % of the sequence) | 28.5 % at layer 0, 1.5 % at layer 2, 8.1 % mean over layers 2 to 29 |
| Correct answers broken by mean ablation: object + border / random, same count / top-attended non-object / 3 highest-norm | 87.3 / 7.3 / 1.8 / 10.9 % |
| FastV-style drop of 50 % of image tokens after layer 2: accuracy, attention-ranked / random | 56.7 / 70.0 % (91.7 % with no drop) |
| Knockout, image -> last position, layers 18 to 24 / 0 to 6 | -58 % / -1 % relative change in P(correct) |
| Patching restoration: object image tokens / object word in the question, any layer | 0.60 / 0.00 |
| Circles overlap: VLM answer / linear probe on the encoder | 47.5 % / 100.0 % (chance 50 %) |
| Connector: held-out R^2 of rebuilding its input / k-NN relations changed | 0.695 / 37.4 % |

### 1. Does the model use the image? (`01_blind_baseline.py`, 60 scenes, 180 prompts)

| Question type | Chance | S_v | Blank image | S_wv (no image) | S_t (base LM) | MG | ML |
|---|---|---|---|---|---|---|---|
| Colour of a named shape | 25 | 86.7 | 21.7 | 38.3 | 28.3 | +48.3 | 10.0 |
| Shape of a named colour | 33 | 95.0 | 35.0 | 35.0 | 35.0 | +60.0 | 0.0 |
| "Is there a dog in the image?" (always no) | 50 | 100.0 | 100.0 | 0.0 | 0.0 | +100.0 | 0.0 |

Two results are worth a second look. First, a blank grey image is not the same as no image. With a blank image the
model says "No" to the dog question every time. With the image removed it says "Yes" every time, and so does its base
language model. A blank image is evidence of absence. So the choice of "without the image" decides MG for that
question type: +100 here, 0 with a blank image. Second, ML is 10 points on colour questions. These scenes are
generated fresh, so this cannot be memorised benchmark items. The blind VLM simply has a stronger colour prior than
its base LM. ML measures "blind VLM beats its LM", and leakage is only one way to get there.

### 2. The logit lens (`02_logit_lens.py`, 48 scenes, 459 object tokens, 2190 background tokens)

At the last position the answer forms late, and its meaning comes before its format. The lowercase word " red"
jumps from 0.04 at layer 22 to 0.82 at layer 23. The capitalised " Red", which the model actually outputs, stays
below 0.37 until layer 29, where it reaches 0.90. Zhang et al. (2024) report the same order on LLaVA-1.5.

At image positions, colour words appear only at layers 25 to 29. At the peak (layer 28), 10.0 % of object tokens
have their own colour in the top 5. Any of the three other colours appears for 9.6 %, about 3.2 % per colour. Only
2.9 % of background tokens decode to the object's colour. So image tokens become word-like late and weakly, as Neo
et al. (2024) found, at a lower rate than their 24 % on LLaVA-1.5. Shape words almost never decode (at most 0.2 %).

### 3. Attention, sinks and ablation (`03_attention_sinks_ablation.py`, 30 scenes, 60 prompts)

Image tokens are 64 of the 86 tokens. The last position gives them 28.5 % of its attention at layer 0, then 6.5 %
at layer 1 and 1.5 % at layer 2. Over layers 2 to 29 the mean is 8.1 %, with one spike to 43.0 % at layer 23 (the
layer where the answer appears in the lens). The five prefix tokens before the image take 52.8 % on average over
layers 2 to 29. Per token, that is about 84 times what one image token gets.

The three image tokens that get the most attention are largely the same for two different questions about the same
image: 53 % shared, where two random picks of 3 from 64 would share about 5 %. So some image tokens draw attention
whatever the question, as in Kang et al. (2025). Their residual norm is only 1.29 times the median, so in this
model they are not the massive-activation tokens of Sun et al. (2024).

| Image tokens replaced with the mean image token | Accuracy % | Mean P(correct) | Correct answers broken % |
|---|---|---|---|
| None | 91.7 | 0.827 | 0.0 |
| The object's tokens plus a one-token border | 11.7 | 0.119 | 87.3 |
| Random non-object tokens, same count | 90.0 | 0.794 | 7.3 |
| Most-attended non-object tokens, same count | 96.7 | 0.843 | 1.8 |
| The 3 highest-norm tokens | 83.3 | 0.775 | 10.9 |
| FastV: the 50 % least-attended tokens, removed after layer 2 | 56.7 | 0.479 | 41.8 |
| A random 50 %, removed after layer 2 | 70.0 | 0.636 | 25.5 |

This is the post's "attention is not importance" result, measured. Removing the most-attended non-object tokens
breaks fewer answers than removing random ones. Removing the object's own tokens breaks 87 %, close to Neo et al.'s
"over 70 percent". FastV's ranking does worse than random here. Its dropped tokens also sit higher in the image (mean
row 2.85 of 0 to 7, against 3.53 for random), the position bias Endo et al. (2024) describe for 1D RoPE. With only
64 tokens per image, dropping half after layer 2 is far harsher than FastV's setting on 576 tokens.

### 4. Attention knockout (`04_knockout.py`, 16 correctly answered prompts, windows of 7 of 30 layers)

Mean relative change in P(correct colour), in %:

| Route blocked \ layers | 0-6 | 4-10 | 8-14 | 12-18 | 16-22 | 18-24 | 22-28 |
|---|---|---|---|---|---|---|---|
| Image -> question text | +1 | -14 | -15 | -42 | -17 | -1 | +0 |
| Object's tokens -> question text | +1 | +2 | +1 | -10 | -0 | -1 | -0 |
| Other image tokens -> question text | +1 | +1 | -1 | -12 | -10 | +0 | +0 |
| Question text -> last position | -98 | +12 | +9 | -94 | -87 | -73 | -4 |
| Image -> last position | -1 | -8 | -8 | -2 | +0 | -58 | -57 |

(`results/04_knockout.txt` has every window, starting every 2 layers.)

The route is not LLaVA-1.5's. Zhang et al. (2024) found the direct image-to-last route negligible on GQA. Here it is
the largest image route, at layers 18 to 28. Image-to-question matters in the middle layers (-42 % at 12 to 18), but
not in the first layers. This is the "nuance" box of the post in action: the route depends on the model and the
prompt. "Question text" here includes the chat template ("Answer with one word.", "Assistant"). Blocking it from the
last position at layers 0 to 6 removes the answer, which says more about the template than about the image.

### 5. Activation patching (`05_patching.py`, 12 minimal pairs, mean P_clean - P_corrupt = 0.90)

Each pair recolours one shape and asks for its colour. Restoration, 0 = none, 1 = full:

| Patched group \ after layer | emb | 0 | 8 | 16 | 20 | 22 | 24 | 28 | 29 |
|---|---|---|---|---|---|---|---|---|---|
| Object's image tokens | 0.60 | 0.60 | 0.61 | 0.54 | 0.49 | 0.48 | 0.00 | 0.00 | 0.00 |
| Other image tokens | 0.40 | 0.40 | 0.39 | 0.39 | 0.35 | 0.36 | 0.00 | 0.00 | 0.00 |
| The object word ("circle") | 0.00 | 0.00 | -0.00 | -0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Other text tokens | 0.00 | -0.00 | -0.00 | 0.00 | -0.00 | -0.00 | 0.00 | 0.00 | 0.00 |
| Last position | 0.00 | -0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.94 | 0.95 | 1.00 |

The colour lives at image positions until layer 22, then at the last position from layer 24. It never passes
through the question tokens. That matches the knockout's direct image-to-last route at layers 18 to 28. It also
explains the middle-layer image-to-question drop in a new way: question tokens read something they need from the
image, but not the colour itself. Note also that 40 % of the restoration comes from tokens outside the object. The
recoloured pixels change only the object's tokens at the input to SigLIP, but SigLIP's global attention spreads the
change to every token by its output.

### 6. Probe the encoder first (`06_probe_vs_readout.py`, 120 images per task, probes trained on 72, tested on 48)

| Readout | Overlap: test accuracy % | Overlap: control % | Count 1 to 4: test accuracy % | Count: control % |
|---|---|---|---|---|
| The VLM's own answer (all 120 images) | 47.5 | | 90.0 | |
| Probe on the encoder (mean of 1024 patches) | 100.0 | 41.7 | 100.0 | 37.5 |
| Probe on the connector output (mean of 64 tokens) | 100.0 | 39.6 | 100.0 | 37.5 |
| Probe on the LM, last position, layer 0 | 100.0 | 31.2 | 95.8 | 29.2 |
| Probe on the LM, last position, layer 15 | 100.0 | 39.6 | 87.5 | 25.0 |
| Probe on the LM, last position, layer 29 | 97.9 | 43.8 | 95.8 | 33.3 |

Chance is 50 % for overlap and 25 % for count. The control columns train the same probe on shuffled labels.

The overlap task is the BlindTest pattern from the post. The model answers at chance. A linear probe reads the answer
perfectly from the encoder, from the connector, and even from the language model's own last position at every
layer probed. The information reaches the position that speaks, and the model still does not say it. That is a readout
failure, and a bigger encoder would not fix it. The count task is the contrast: the model already uses what the
probe finds. The post's nuance applies too. The probe was trained with labels for this exact task, and with
outlined circles it may use a shortcut such as the amount of ink. The result shows the answer is linearly
available, not that the model should have learned it.

### 7. The connector (`07_connector.py`, 150 scenes, 3200 held-out tokens)

A ridge map from the 576-dim connector output back to its 12,288-dim input explains 69.5 % of the held-out variance.
The median relative error is 0.27 on object interiors, 0.38 on object edges and 0.30 on background. So the loss is
local, and it is highest where an object meets the background. Of each token's 10 nearest neighbours, 37.4 % change
after projection, a little below the 40 to 60 % Li et al. (2025) report across connectors.

The post says high loss on the relevant patches predicts failure. On these 100 colour questions it does not. The
Spearman correlation between an object's mean reconstruction error and P(correct colour) is +0.25. The model got 88
% of them right, and colour is easy to keep under a linear map. A test of this claim needs questions about fine
detail, such as small text.

### Timings

The CPU was shared with other jobs (load average above 150), so timings are rough. Wall time per script:
01, 867 s; 02, about 30 minutes on a node with load average above 500; 03, 204 s; 04, 199 s; 05, 228 s; 06, 532 s;
07, 285 s. Scripts 03 to 07 ran four at a time. Scripts 01 and 02 ran before the wrapper cached image embeddings
per prompt, so they ran the vision encoder on every forward pass. The numbers do not change; only the time does.

## What was not measured

- No LLaVA-1.5, Qwen2-VL or any model with 576 or more image tokens. The post's quoted numbers (for example 24
  percent logit-lens decodability near layer 26 of 33, a 60 percent knockout drop in layers 0 to 4, FastV's 45
  percent compute saving) are from those papers and were not reproduced here.
- No natural images. The scenes are synthetic shapes, chosen so object masks and minimal pairs are exact.
- Not built: gradients and integrated gradients, attention rollout, the tuned lens, sparse autoencoders, steering
  vectors, visual contrastive decoding, native early-fusion models (the "narrow gate").
- FastV is run as attention knockout of the dropped tokens, so it shows the accuracy cost but saves no compute. No
  speed was measured, because the post's claims here are about which tokens matter, not about time.
- Only one random seed per experiment. No confidence intervals.
