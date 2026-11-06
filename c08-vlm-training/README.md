# C8: Train a Real VLM

Code for the coding tutorial **Train a Real VLM** in the series *The Backward Pass: Coding Tutorials*.

C7 built `tinyvlm`, a VLM made from a pretrained SigLIP-base/16 encoder, a new 2-layer MLP projector with 2x2 pixel
shuffle, and SmolLM2-135M-Instruct. This folder trains that model with a staged recipe:

1. **Stage 1, alignment.** Vision and LM frozen. Only the 2.1M-parameter projector trains, on captions.
2. **Stage 2, multimodal pretraining.** A mixture of six synthetic tasks (captions, VQA, OCR, documents, charts,
   grounding) plus text replay, with frozen, partially unfrozen or fully joint vision, and per-module learning rates.
3. **Comparisons** of task mixtures, grounding formats, resolution strategies, LoRA placements and packing.

Every task is rendered synthetically (shapes, words, invoices, bar charts), so every label is exact and every metric
can be checked by hand. The trainer imports C7's `tinyvlm` from `../c07-vlm`. If that folder is missing, it falls back
to `vlmtrain/model.py`, a one-file model with the same contract (a network test checks the two give the same logits).

## Layout

```
c08-vlm-training/
  vlmtrain/
    synth/            shapes.py, text.py, charts.py: rendered images whose labels are exact
    data.py           the unified {"images", "messages"} sample, one generator per task
    tokenizer.py      <image> and <loc_0>..<loc_31> tokens; spread_new_rows for new embedding rows
    processor.py      chat template, image-token expansion, loss mask, padding
    resolution.py     resize, pad and tile strategies (tiles plus a thumbnail)
    packing.py        first-fit-decreasing packing, block-diagonal causal mask
    freeze.py         freeze presets align | lm | partial | full | none, trainable-parameter report
    optim.py          one AdamW group per module (vision, projector, lm, lora), warmup + cosine
    mixture.py        proportional, uniform, temperature and manual mixtures; a sampler that counts what it drew
    replay.py         WikiText-2 text replay and the perplexity check for language forgetting
    lora.py           LoRALinear, placement by regex on lm, vision, projector; merge
    grounding.py      points, boxes and mask outlines as <loc_k> tokens or plain-text numbers; RLE masks
    metrics.py        VQA exact match, ANLS, CER and WER, ChartQA relaxed accuracy, box IoU, point-in-box
    evaluate.py       generation-based evaluation per task, a grey-image control and a majority baseline
    train.py          the staged trainer: TrainConfig, run()
    c07_adapter.py    builds C7's TinyVLM (real or tiny random) and adds the location tokens
    model.py          standalone fallback with the same contract as C7's TinyVLM
    device.py         device choice, thread cap, hardware label, memory
  configs/            stage1_align.yaml, stage2_mixture.yaml
  scripts/            train.py, run_all.sh (CPU runs), run_gpu.sh (GPU runs), evaluate.py, break_it.py,
                      inspect_batch.py, show_samples.py, param_report.py, multi_image.py
  benchmarks/         bench_packing.py, bench_resolution.py
  tests/              data, processor, grounding, metrics, mixture, packing, resolution, LoRA, stages, overrides,
                      C7 contract, real weights (network)
  results/            every number below, as JSON, plus logs/ (CPU) and gpu/ (GPU)
```

## Setup

From the repository root, with the repository's environment installed (`pip install -e ".[dev]"`). The first real run
downloads `google/siglip-base-patch16-224` and `HuggingFaceTB/SmolLM2-135M-Instruct` (about 650 MB) and WikiText-2.
Runs use the GPU when there is one (CUDA, then MPS, then CPU), or the device named by `VLMTRAIN_DEVICE`. CPU runs cap
themselves at `$OMP_NUM_THREADS` threads, or 8.

## Run

```bash
cd c08-vlm-training
python scripts/train.py configs/stage1_align.yaml                     # Stage 1, saves outputs/stage1_align/
python scripts/train.py configs/stage2_mixture.yaml                   # Stage 2 from the Stage 1 projector
python scripts/train.py configs/stage2_mixture.yaml name=s2_noreplay replay_ratio=0   # any key=value override
python scripts/evaluate.py outputs/stage2_main/trainable.safetensors --name stage2_main   # seen / blind / majority
bash scripts/run_all.sh                 # every CPU run behind the tables below, one at a time (skips finished runs)
bash scripts/run_all.sh mixture         # one group: stage1 stage2 freeze mixture grounding resolution lora packing own capture
CUDA_VISIBLE_DEVICES=0 bash scripts/run_gpu.sh stage1_align    # GPU runs, one job per GPU, into results/gpu/
CUDA_VISIBLE_DEVICES=0 bash scripts/run_gpu.sh stage2_main eval_stage2_main
```

`run_gpu.sh` runs stage1_align first, then stage2_main, stage2_noreplay, eval_stage2_main and the comparison jobs
(freeze_*, mix_*, ground_*, res_*, lora_*) in any order across GPUs, because they all start from the Stage 1 projector.
Set `STEPS_S1`, `STEPS_S2`, `STEPS_CMP` and `STEPS_LONG` to change the scale. The defaults are the settings in the results.

Inspection and Break it scripts (no training, except the short Break it runs):

```bash
python scripts/inspect_batch.py         # one mixed batch: tiles, visual tokens, supervised tokens, the loss mask
python scripts/show_samples.py          # one rendered PNG per task to outputs/samples/
python scripts/param_report.py          # trainable parameters per freeze preset and LoRA placement
python scripts/multi_image.py           # 1 to 3 images in one example: slots, tiles and token budget
python scripts/break_it.py image_tokens # also: loss_mask block_mask max_length high_lr frozen_loc
python benchmarks/bench_packing.py      # padding vs grouping vs packing: wasted positions and timed steps
python benchmarks/bench_resolution.py   # visual tokens per strategy and vision-encoder time
```

## Tests

```bash
cd <repo root>
python -m pytest c08-vlm-training -m "not slow and not gpu and not network"   # offline, random tiny models
python -m pytest c08-vlm-training -m network                                  # real SigLIP + SmolLM2 weights
```

## Results

Two machines produced these numbers. Every row says which.

- **CPU** is an AMD EPYC 7R13 with 8 threads, on a shared server.
- **GPU** is one NVIDIA H100 80GB HBM3 per run, with torch 2.15.0+cu132 and CUDA 13.2, in float32 (the trainer has no mixed precision). Runs came from `scripts/run_gpu.sh`. Raw numbers are in `results/gpu/`.

All data is synthetic, so each score measures one skill that we drew ourselves. Every run is one seed. With 128 test samples a score near 0.5 has a standard error of about 0.04, and with 64 samples about 0.06. Treat differences smaller than that as noise. Seconds per step are not comparable between rows: both machines were shared with other jobs and the load changed during the runs.

### The settings are smaller than the plan

The plan was a 3000-step Stage 1 and an 8000-step Stage 2, with 2000-step comparisons. A first attempt at 8000 steps ran at 1.5 s/step while the server's CPUs were oversubscribed. After 34 minutes it was at step 1400, so I stopped it and kept none of its output. The final GPU runs are:

| Run | Steps x batch | Samples seen | Eval samples per task |
|---|---|---|---|
| Stage 1 alignment | 3000 x 32 | 96,000 | 256 |
| Stage 2 main and no-replay | 1000 x 16 | 16,000 | 128 |
| Comparisons (freeze, mixture, LoRA, resolution, grounding) | 250 x 16 | 4,000 | 128 |
| Long grounding runs | 1500 x 16 | 24,000 | 128 |

Every comparison uses the same settings and starts from the same Stage 1 projector, so they compare with each other. They do not compare with the 1000-step main run. The CPU runs are smaller still (600 x 8 for Stage 1, 500 x 4 for Stage 2, 200 x 4 or 150 x 4 for comparisons), so a CPU row and a GPU row are different experiments.

### Stage 1: alignment

Only the 2,102,400-parameter projector trains. The checkpoint is 8.0 MB.

| Hardware | Steps x batch | Caption exact match, real images | Grey images | Most common answer |
|---|---|---|---|---|
| CPU | 600 x 8 | 0.797 (n=64) | 0.000 | 0.062 |
| GPU | 3000 x 32 | 0.980 (n=256) | not run | not run |

The grey-image score of 0.000 says the captions come from the image. The model starts at loss 3.0 and reaches 0.05 on the CPU run.

### Stage 2: the six-task mixture with 10% text replay

Frozen vision, trainable projector and LM (136,654,272 trainable parameters, checkpoint 521.3 MB). "Grey" is the same model on blank images. "Majority" is a fixed answer with no model. Metrics are exact match, except OCR (character error rate, lower is better), documents (ANLS), charts (relaxed accuracy) and grounding (box IoU of at least 0.5).

| Task | GPU real (n=128) | GPU grey | GPU majority | CPU real (n=64) | CPU grey | CPU majority |
|---|---|---|---|---|---|---|
| Caption | 0.930 | 0.023 | 0.047 | 0.828 | 0.031 | 0.062 |
| VQA | 0.953 | 0.234 | 0.250 | 0.719 | 0.438 | 0.281 |
| OCR (CER, lower is better) | 0.013 | 0.847 | 0.891 | 0.469 | 0.830 | 0.896 |
| Documents (ANLS) | 0.191 | 0.197 | 0.031 | 0.167 | 0.202 | 0.016 |
| Charts (relaxed accuracy) | 0.148 | 0.117 | 0.125 | 0.125 | 0.031 | 0.125 |
| Grounding (IoU >= 0.5) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

What this table says:

- **Caption, VQA and OCR are learned.** The real-image score is far from both controls. The GPU run reads the short rendered words with a 1.3% character error rate.
- **Documents are not learned.** The score equals the grey-image score. The model answers invoice questions without reading the invoice, for example `32.50` for a total of `79.50`.
- **Charts are not learned.** The score is the same as always answering the most common value. For example, it answers `1` for bars of height 5 and 4.
- **Grounding is weak.** See the section below.

The GPU run was also evaluated at step 500 (48 samples per task). OCR error fell from 0.123 to 0.013 and VQA rose from 0.896 to 0.953 by step 1000 (128 samples), so Stage 2 had not converged.

### Language forgetting

Perplexity of the language model on held-out WikiText-2 test passages, before and after multimodal training.

| Hardware | Run | Before | After |
|---|---|---|---|
| GPU | 10% text replay, 1000 steps | 35.32 | 23.19 |
| GPU | no replay, 1000 steps | 35.32 | 49.85 |
| CPU | 10% text replay, 500 steps | 38.37 | 28.78 |
| CPU | no replay, 500 steps | 38.37 | 45.10 |
| CPU | 10% replay, token-mean loss, 100 steps | 38.37 | 31.61 |

Without replay, perplexity rises by 14.5 points on the GPU run: the LM forgets. With replay it falls. Part of that fall is not retention. The replay passages are WikiText-2 train, and the check uses WikiText-2 test, so replay also adapts the LM to the test domain. The no-replay row is the measure of forgetting. The "before" values differ between GPU and CPU because they use 128 and 64 passages.

Replay is 10% of the samples but 63 to 66% of the supervised tokens, because a passage is a hundred tokens and an answer is a few. This is why the main config averages the loss per sample (`loss_norm: sample`). The 100-step token-mean run in the table shows the other choice.

### Comparisons on GPU (250 steps x 16, NVIDIA H100 80GB HBM3)

All start from the Stage 1 projector and use the same mixture except where noted. Real images, n=128. The temperature mixture and the frozen-vision row are the same run (`freeze_lm`).

| Run | Caption | VQA | OCR CER | Doc ANLS | Chart |
|---|---|---|---|---|---|
| freeze_lm (vision frozen, T=2 mixture) | 0.953 | 0.844 | 0.692 | 0.196 | 0.117 |
| freeze_partial (top 4 vision layers, lr 1e-5) | 0.898 | 0.836 | 0.612 | 0.174 | 0.133 |
| freeze_full (joint, vision lr 1e-6) | 0.961 | 0.828 | 0.670 | 0.211 | 0.125 |
| mix_proportional | 0.961 | 0.906 | 0.754 | 0.052 | 0.109 |
| mix_uniform | 0.961 | 0.773 | 0.621 | 0.224 | 0.125 |
| mix_manual (more doc, grounding, chart, OCR) | 0.930 | 0.680 | 0.668 | 0.173 | 0.133 |
| lora_lm (r=16) | 0.945 | 0.812 | 0.854 | 0.144 | 0.125 |
| lora_vision | 0.586 | 0.445 | 1.997 | 0.005 | 0.078 |
| lora_projector | 0.836 | 0.461 | 1.929 | 0.016 | 0.047 |
| lora_all (lm, vision, projector) | 0.961 | 0.812 | 0.781 | 0.116 | 0.133 |

- **Freezing:** the three freeze presets are within noise of each other at 250 steps. No claim about which is best is supported. Partial unfreezing lowered caption exact match by 0.05 to 0.06.
- **Mixtures:** proportional mixing favours the large tasks. It has the best VQA (0.906) and the worst documents (0.052, below the 0.197 grey-image score). Uniform mixing has the best document score (0.224) and a VQA of 0.773. The hand-weighted mixture, which favours documents, grounding, charts and OCR, has the lowest VQA (0.680).
- **LoRA:** LoRA on the LM alone is close to full LM training on caption and VQA but worse on OCR (0.854 against 0.692). LoRA on the vision tower alone or the projector alone did not learn the tasks in 250 steps. In both the LM stays frozen, which is the likely reason, but I did not test it. LoRA on all three recovers most of the quality.
- All LoRA runs set `train_new_tokens=true`, which makes the embedding matrix trainable (28.3M parameters) with a hook that zeroes the gradient of the old rows. The trainable counts in `results/lora_*.json` include it.

Dynamic resolution (`res_tile` against `res_resize`, documents and OCR only, GPU 250 x 16): OCR CER 0.055 with tiles and 0.045 with one resized tile; document ANLS 0.188 and 0.185. Neither run reads documents (the grey-image score is about 0.2), so this test does not show an advantage for tiling. On CPU (200 x 4, 48 samples) tiling gave OCR CER 0.347 and document ANLS 0.268, and one resized tile gave 0.282 and 0.226. That difference is within noise. Tiling cost 245 visual tokens per invoice against 49.

### Grounding (weak result)

Targets are 60% boxes, 20% points and 20% mask outlines on shapes we drew. Real-image scores, n=128 per run.

| Run (NVIDIA H100 80GB HBM3) | Steps x batch | Box IoU >= 0.5 | Mean box IoU | Point in box | Mask IoU | Unparsable |
|---|---|---|---|---|---|---|
| tokens (`<loc_k>`), default init | 250 x 16 | 0.000 | 0.000 | 0.10 | 0.00 | 0.05 |
| tokens, `loc_init=spread` | 250 x 16 | 0.000 | 0.04 | 0.10 | 0.00 | 0.02 |
| plain text numbers | 250 x 16 | 0.000 | 0.05 | 0.07 | 0.04 | 0.00 |
| tokens, default init | 1500 x 16 | 0.029 | 0.104 | 0.10 | 0.11 | 0.00 |
| tokens, `loc_init=spread` | 1500 x 16 | 0.057 | 0.102 | 0.14 | 0.04 | 0.01 |
| plain text numbers | 1500 x 16 | 0.043 | 0.083 | 0.10 | 0.11 | 0.00 |

None of these models can ground. Box accuracy is under 6% and a point lands in the right box 10 to 14% of the time. The three formats are within noise of each other, so these runs do not say which is better. The long runs do show that grounding is undertrained, not broken: the predicted boxes sit in the right part of the image, for example `<loc_23><loc_17><loc_30><loc_25>` for the target `<loc_23><loc_15><loc_31><loc_22>`.

One cause is measured. With the default initialisation, Hugging Face puts all 32 new `<loc_k>` embedding rows within about 1e-6 of the mean row. `scripts/break_it.py frozen_loc` prints the effect: the standard deviation across the 32 rows is 3.2e-6 against 9.95e-2 for the vocabulary, and the 32 logits at a grounding prompt differ by only 5e-4. On CPU, with 200 x 4 steps, the default run never produced a coordinate (every output was `<|im_end|>`) and `loc_init=spread` cut unparsable outputs from 100% to 52%. The GPU runs at 250 x 16 see five times more samples and do produce coordinates with the default initialisation. I did not test which of the two changes matters.

`ground_text` was evaluated with a 100-token generation limit and the tokens format with 24. A text-format outline needs up to 96 tokens, so a 24-token limit would have capped its mask IoU.

### Packing and resolution benchmarks

480 samples from the T=2 mixture, 49,251 real tokens, lengths 68 to 274 (mean 103).

| Layout | Rows | Positions | Wasted |
|---|---|---|---|
| Pad, random order, batch 8 | 480 | 98,984 | 50.2% |
| Pad, sorted by length, batch 8 | 480 | 50,040 | 1.6% |
| Pack, budget 512 | 100 | 51,200 | 3.8% |

Timed forward and backward on 64 samples (6,734 real tokens), one packed row at a time:

| Hardware | Padded | Packed |
|---|---|---|
| CPU | 29.3 s, 230 real tok/s | 21.3 s, 316 real tok/s |
| GPU (NVIDIA H100 80GB HBM3) | 1.6 s, 4,245 real tok/s | 1.7 s, 3,910 real tok/s |

Packing wastes 3.8% of positions and sorting by length 1.6%, so packing does not beat sorting here. On the GPU it was not faster either. The benchmark runs each packed row alone at batch 1, so it does not test a packed batch. A 60-step training pair on CPU (`train_padded`, `train_packed`, batch 8) took 3.86 and 3.40 s/step and gave the same evaluation scores on 8 samples. `tests/test_packing.py` shows that packed and separate losses agree to 1e-4.

Visual tokens per image: 49 for resize and pad, and 49, 147, 196, 147 and 245 for tiling a 224 x 224 scene, 336 x 224 text, 448 x 160 text, 448 x 224 text and a 448 x 448 invoice. Encoding one invoice took 82 ms with one resized tile and 346 ms with 5 tiles on CPU. On GPU the same call took 10 ms and 7 ms. That is a single call on a shared server, so the order of the two is noise.

### Multiple images in one example

No training task uses more than one image. `scripts/multi_image.py` runs C7's model on 1 to 3 images (CPU):

| Example | Tiles per image | `<image>` slots | `pixel_values` | Tokens |
|---|---|---|---|---|
| scene | [1] | 49 | (1, 3, 224, 224) | 67 |
| scene + OCR | [1, 3] | 196 | (4, 3, 224, 224) | 214 |
| scene + invoice | [1, 5] | 294 | (6, 3, 224, 224) | 312 |
| three invoices | [5, 5, 5] | 735 | (15, 3, 224, 224) | 753 |

Three invoices (753 tokens) fit a 1024-token row once and do not fit a 512-token row.

### Trainable parameters

`scripts/param_report.py`, for the 222,451,392-parameter model:

| Setting | Trainable | Share |
|---|---|---|
| Stage 1 (projector) | 2,102,400 | 0.9% |
| LM + projector | 136,654,272 | 61.4% |
| + top 4 vision layers | 165,007,296 | 74.2% |
| Everything | 222,451,392 | 100% |
| LoRA r=16 on the LM (210 linears) | 4,884,480 | 2.1% |
| LoRA on vision (72) | 2,654,208 | 1.2% |
| LoRA on projector (2) | 76,800 | 0.0% |
| LoRA on all three (284) | 7,615,488 | 3.3% |

Peak GPU memory for the 1000-step Stage 2 run (batch 16, float32): 11.6 GB.

### Break it

Each is a function in `scripts/break_it.py`, with its real output in `results/logs/break_<name>.txt` (CPU).

| Name | What happens |
|---|---|
| `image_tokens` | Dropping one tile raises `ValueError: 294 <image> placeholders but 245 image feature vectors (5 crops x 49 tokens)`. |
| `loss_mask` | Supervising the prompt too logs a loss of 8.49, and training on the answer only logs 0.81. The held-out answer loss is 2.25 against 0.92. Nothing raises. |
| `block_mask` | A plain causal mask on a packed row leaves the first sample unchanged (0.0) and shifts the second sample's logits by up to 30.65. Its loss goes from 3.552 to 9.035. |
| `max_length` | 1 tile costs 66 tokens and 386 ms, 5 tiles 262 tokens and 696 ms, 17 tiles 850 tokens and 2345 ms. 36 tiles raises `ValueError: sample has 1830 tokens > max_length 1024`. |
| `high_lr` | A learning rate 1000 times too high does not produce NaN. The loss stalls at 1.99 against 0.81, with a held-out answer loss of 2.02 against 0.92. |
| `frozen_loc` | The 32 `<loc_k>` rows are 3.2e-6 apart, so the model cannot prefer one bin over another. |

### What was not measured

- **A full-scale Stage 2.** The 8000-step run did not fit the time budget. The numbers above are for 1000 steps.
- **bf16 or fp16.** All runs are float32.
- **More than one seed.** Differences between comparison rows are single-run differences.
- **Real data.** Everything is rendered shapes, words, invoices and bar charts, so the scores do not predict performance on photographs or scans.

### What was fixed along the way

- `scripts/train.py` read an override such as `lrs.vision=1e-5` as a string, because YAML 1.1 wants `1.0e-5`. The first `freeze_partial` run crashed on it. `parse_value` now returns a float (`tests/test_overrides.py`).
- The standalone model loaded SigLIP with a `vision_use_head` argument the class does not accept, and loaded SmolLM2 in bfloat16 while the rest ran in float32. Both are fixed and `tests/test_real_weights.py` checks that the standalone model and C7's model give the same logits.
- Captions said "a orange circle". They now say "an orange circle".
- Mask outlines had an encoder but no decoder or scorer. `decode_polygon`, `polygon_to_mask` and `mask_iou` close that.
- Throughput now counts real tokens, not padded positions.

## Tests

| Command | Result (CPU) |
|---|---|
| `pytest c08-vlm-training -m "not slow and not gpu and not network"` | 130 passed in 36 s |
| `pytest c08-vlm-training -m network` | 5 passed in 35 s |

There are no `slow` or `gpu` tests. The network tests download the real SigLIP and SmolLM2 weights.

