# Lab: When Language Must Point

Code companion to the post [When Language Must Point: Grounding in Vision-Language Models](https://srini-anirudh.github.io/blogs/when-language-must-point/).

The post asks how a vision-language model shows *where* it sees something. A location is geometry, and a language
model only writes tokens. This lab builds the post's central claims in small Python and runs them:

1. The best possible box has an IoU equal to the shape's fill ratio. A box can be "perfect" and still mostly background.
2. The eight coordinate formats in the post's table. The lab writes one box in each format, counts its tokens and
   parses it back. Then it parses each string with the wrong convention and shows that nothing raises an error.
3. The bin count is a resolution budget. The worst-case IoU formula `((s - Δ) / s)^2` is checked by brute force.
4. Cross-entropy cannot tell a near miss from a far miss. An IoU reward can.
5. Type, select or decode: three interfaces trained on one toy referring task and scored on masks.
6. The metrics from the post: inside tests, gIoU against cIoU, boundary IoU, the GIoU name clash, set matching
   for many regions or none, granularity mismatch and temporal IoU.

Everything is from scratch. The models are tiny, and the referring task is synthetic: 64 x 64 images with three
coloured shapes, and a query that names one colour. No paper's scale is reproduced here. Where the post quotes a
paper's number, this lab shows the mechanism behind it, not the number.

Related tutorials: [C8 VLM training](../../c08-vlm-training/) trains a real small VLM with grounding data in both
location-token and text formats. [C9 VLM post-training](../../c09-vlm-post-training/) uses IoU as an RL reward on that
model. This lab stays below them: it isolates the interface decisions on problems small enough to inspect.

## Layout

```
labs/when-language-must-point/
├── pointlab/
│   ├── geometry.py   # rasterise shapes, tightest box, fill ratio, centroid, box / mask / interval IoU
│   ├── coords.py     # the eight coordinate conventions, token counts, quantisation, Qwen's smart_resize
│   ├── scoring.py    # cross-entropy vs IoU reward, gIoU / cIoU, boundary IoU, GIoU, set-matched F1
│   ├── toy.py        # the toy referring task, a class-agnostic proposer (connected components)
│   ├── models.py     # TypedGrounder (digits or location tokens), SelectGrounder, DecodeGrounder (<SEG> -> mask)
│   ├── train.py      # training loops (CE, GRPO-style IoU reward, BCE + Dice) and evaluation
│   └── report.py     # prints results and writes them to results/ with a hardware line and the command
├── scripts/          # one script per claim, listed below
├── tests/            # pytest
└── results/          # real outputs of every script
```

## Setup

From the repository root, after `pip install -e ".[dev]"`. Set `OMP_NUM_THREADS` on a shared machine. The CPU numbers
below used 8 threads.

## Run everything

```bash
# Tests (about 20 s on CPU).
python -m pytest labs/when-language-must-point -m "not slow and not gpu and not network"
python -m pytest labs/when-language-must-point -m network   # downloads two small tokenizers
python -m pytest labs/when-language-must-point -m gpu       # needs CUDA

# Analysis scripts (seconds each, CPU).
python labs/when-language-must-point/scripts/01_shapes.py
python labs/when-language-must-point/scripts/02_encodings.py        # uses the SmolLM2 and Qwen2.5 tokenizers if it can
python labs/when-language-must-point/scripts/03_quantisation.py
python labs/when-language-must-point/scripts/04_ce_blindness.py
python labs/when-language-must-point/scripts/07_metrics.py

# Training (CPU: the short settings; GPU: the full settings, three seeds).
python labs/when-language-must-point/scripts/05_train_coordinates.py --steps 1500 --rl-steps 300 --seeds 0
python labs/when-language-must-point/scripts/06_type_select_decode.py --steps 1200 --seeds 0
python labs/when-language-must-point/scripts/05_train_coordinates.py --name 05_train_coordinates_cuda
python labs/when-language-must-point/scripts/06_type_select_decode.py --steps 3000 --name 06_type_select_decode_cuda
```

The training scripts use CUDA when it is visible and the CPU otherwise.

| Post section | Script |
|---|---|
| Point, box, polygon, mask (Figure 2, the fill-ratio derivation) | `01_shapes.py` |
| Writing coordinates as tokens (the eight-encodings table) | `02_encodings.py` |
| Normalised or absolute coordinates (the Qwen2.5-VL / Qwen3-VL nuance) | `02_encodings.py` |
| How many bins (Figure 4, the worst-case IoU formula) | `03_quantisation.py` |
| Grounding as next-token prediction: what it costs (Figures 3 and 5) | `04_ce_blindness.py`, `05_train_coordinates.py` |
| The IoU-reward insight (VLM-R1, Visual-RFT) | `04_ce_blindness.py`, `05_train_coordinates.py` |
| Handing the location to a decoder: type, select or decode (Figure 6) | `06_type_select_decode.py` |
| What counts as one region (granularity mismatch, nested regions) | `07_metrics.py` |
| Measuring grounding (Figure 9, gIoU vs cIoU, boundary IoU, matching) | `01_shapes.py`, `07_metrics.py` |

## Results

Every number names its hardware. Raw output is in `results/`, under the name of the script that wrote it. Each file
starts with a hardware line and the command.

Tests on AMD EPYC 7R13, 8 threads, torch 2.15.0: 45 passed offline in about 20 s. The `network` test (real tokenizers)
passed. The `gpu` test is listed under "not yet measured" below.

### Geometry, encodings, quantisation, metrics (AMD EPYC 7R13, 8 threads; exact, no training)

| Claim in the post | What the lab measures | Result |
|---|---|---|
| Best-box IoU equals the fill ratio | Figure 2 band: fill ratio and IoU of its tightest box | 0.338 and 0.338 |
| Ellipse fills π/4 of its box | rasterised ellipse | 0.785 (π/4 = 0.785) |
| Thin 45° text line falls below 0.1 | t / L = 0.05, formula 2Lt / (L + t)^2 vs rasterised | 0.091 vs 0.090 |
| Figure 9 shapes: fill ratio (= mask IoU of a box with box IoU 1.00) | blob / pie wedge / curved line / diagonal line | 0.735 / 0.557 / 0.334 / 0.118 |
| The centroid of a curved band lies outside it | inside test; distance to the nearest band pixel | outside; 19.1 px (misses even the 15 px pointing-game tolerance) |
| Polygons approach the shape at a cost of 2 numbers per vertex | Figure 2 band, 6 / 16 / 48 vertices: IoU and digit-split tokens | 0.28 / 0.92 / 0.99 IoU; 61 / 161 / 481 tokens |
| The eight encodings in the post's table | strings written by `coords.py` | all eight match the table character for character |
| A text box runs to roughly twenty tokens; location tokens make it four | Qwen2.5-0.5B tokenizer: CogVLM / Qwen-VL / Qwen2.5-VL JSON / Florence-2 | 17 / 23 / 32 / 4 tokens |
| A wrong parser fails silently | IoU with the true box: PaliGemma read x-first; Qwen2.5-VL pixels read as 0–1000; Qwen3-VL 0–1000 read as pixels | 0.194 / 0.199 / 0.199, no error raised |
| Qwen2.5-VL pixels belong to the resized input | 3840 x 2160 screenshot with `max_pixels = 1280 * 28 * 28` | the model sees 1316 x 728; reading its numbers as original pixels is off by 2.92x |
| Worst-case IoU = ((s − Δ)/s)^2 | brute force over 6,561 edge-error combinations, 5 cases | formula and brute force agree to 4 decimals; worst case is all four edges inward |
| 1000 bins, 3840 px, 24 px icon | worst-case IoU | 0.71 |
| 0–100 grid, 3840 px, 24 px icon | bin width and worst case | 38.4 px bin, IoU 0 |
| Kosmos-2 32 cells on a 2000 px page | bin width | 62.5 px |
| Cross-entropy is blind to distance (Figure 5) | B (90% one bin off) vs C (90% 300 bins off) | CE 2.30 and 2.30; equal gradient norms 1.273; argmax-box IoU 0.997 vs 0.513 |
| An IoU reward pulls mass toward near bins | 100 Adam steps from uniform on 1,000 logits | CE: 0.81 on the true bin, 0.82 within 20 bins. IoU reward: 0.04 on the true bin, 0.80 within 20 bins |
| One wrong hundreds digit moves an edge by 30% (Figure 3) | 584 → 884 and 584 → 589 | 30% (box IoU 0.513) and 0.5% (box IoU 0.984) |
| cIoU favours large objects | 2 large right + 8 small wrong, and the reverse | gIoU 0.211 / cIoU 0.883, and gIoU 0.872 / cIoU 0.376 |
| Boundary IoU sees edges | ragged circle edge (±3 px) | mask IoU 0.861, boundary IoU 0.196 |
| "gIoU" and "GIoU" differ | two disjoint boxes | IoU 0, GIoU −0.333 and −0.600 as they move apart |
| Many regions need set matching | three cells, first one skipped in reading order | set-matched F1 0.80, position-by-position F1 0.00 |
| Granularity mismatch | whole-line prediction against a one-word label | IoU 0.092: counted wrong although the evidence is right |

The Text4Seg row-wise run-length idea shortens the Figure 2 band's 16 x 16 text mask by 70% (511 to 153 tokens).
That is one shape, not Text4Seg's 74% over its datasets, which this lab did not reproduce.

### Toy training runs (AMD EPYC 7R13, 8 threads, torch 2.15.0; one seed, short settings)

The task: 64 x 64 scenes with three coloured shapes. The query names a colour. 6,000 training scenes, 1,000 shared
test scenes. Half the scenes let shapes touch or overlap.

**Coordinates as tokens** (`05_train_coordinates.py --steps 1500 --rl-steps 300 --seeds 0`, 526 s). Box on a 0–999
grid. Greedy decoding on the test set.

| Model | Box IoU | Acc@0.5 | Acc@0.75 | Coords > 100 bins off | NLL / coordinate (nats) | Mask IoU of the box: compact / irregular |
|---|---|---|---|---|---|---|
| digits (12 tokens, CE) | 0.857 | 0.987 | 0.892 | 0.5% | 1.50 | 0.778 / 0.350 |
| digits + IoU reward (GRPO-style) | 0.856 | 0.988 | 0.897 | 0.5% | 1.57 | 0.774 / 0.351 |
| location tokens (4 tokens, CE) | 0.842 | 0.968 | 0.876 | 1.5% | 2.13 | 0.768 / 0.349 |
| location tokens + IoU reward | 0.843 | 0.970 | 0.879 | 1.4% | 2.31 | 0.762 / 0.352 |

What this shows, and what it does not:

- The box is good and the mask is not. Both formats reach box IoU near 0.85. Scored against the true region, the
  same boxes give mask IoU 0.77 on compact shapes and 0.35 on irregular ones. The irregular targets fill 35% of their
  boxes, so 0.35 is the ceiling. The fill-ratio limit shows up in a trained model, not only in geometry.
- On this toy, the IoU reward changed almost nothing on the test set. RL samples training scenes, and there the
  supervised models already scored IoU 0.919 (digits) and 0.956 (location tokens) at the start. The reward raised the
  location-token model's sampled training IoU to 0.975. Greedy test IoU moved by 0.001. The gap between about 0.96 on
  training scenes and 0.84 on test scenes suggests the limit here is generalisation, which an on-policy reward on the
  same training scenes does not fix. The reward also made the model less sure of the exact bin: NLL per coordinate
  rose. The post's claim that RL with an IoU reward helps rests on
  VLM-R1 and Visual-RFT at real scale. This lab does not reproduce it.
- Digits did not lose to location tokens here. Twelve digit tokens beat four 1,000-way tokens by 0.015 box IoU and had
  fewer far misses. Both vocabularies start from scratch in this toy, so neither has the pretrained number sense that
  the post credits text digits with. One seed: do not read a ranking into a gap this small.

**Type, select or decode** (`06_type_select_decode.py --steps 1200 --seeds 0`, 1,006 s). Mask IoU against the true
region on the same test set.

| Interface | gIoU | cIoU | Acc@0.5 | gIoU compact | gIoU irregular | gIoU separated | gIoU touching |
|---|---|---|---|---|---|---|---|
| type: box from 4 location tokens | 0.527 | 0.423 | 0.563 | 0.762 | 0.353 | 0.563 | 0.493 |
| type, perfect box (oracle) | 0.572 | 0.440 | 0.566 | 0.875 | 0.350 | 0.619 | 0.530 |
| select: pick one proposal | 0.794 | 0.657 | 0.761 | 0.831 | 0.766 | 1.000 | 0.605 |
| decode: `<SEG>` → mask | 0.982 | 0.985 | 1.000 | 0.990 | 0.976 | 0.992 | 0.973 |

- Typing a box is capped by the fill ratio. The trained box model (0.527) gets close to a perfect box (0.572), and a
  perfect box still scores 0.350 on irregular shapes.
- Selecting is capped by the proposer. When shapes are apart, selection is perfect (1.000). When they touch, the
  class-agnostic proposer merges them, and only 54.2% of those targets have any proposal with IoU of at least 0.5.
- Decoding has neither cap. One `<SEG>` hidden state, projected into a prompt for a dot-product mask decoder,
  reaches gIoU 0.976 on irregular shapes and 0.973 in touching scenes. This is the post's division of labour: the
  language model decides what, the decoder decides where. In this toy the colour query makes the "what" easy. The
  post's M3Grounder and DocLayout-VL numbers come from real documents at 8B scale, which this lab does not reproduce.

### NVIDIA H100 80GB HBM3: not yet measured

The full settings (3,000 steps and three seeds per model, and 400 RL steps for `05`) were queued on the shared GPU
pool. They did not start within the lab's time budget, so these commands have not run yet. The GPU code path has
only been run on the CPU, through the same functions.

```bash
python labs/when-language-must-point/scripts/05_train_coordinates.py --name 05_train_coordinates_cuda
python labs/when-language-must-point/scripts/06_type_select_decode.py --steps 3000 --name 06_type_select_decode_cuda
python -m pytest labs/when-language-must-point -m gpu
```

They write `results/05_train_coordinates_cuda.*` and `results/06_type_select_decode_cuda.*`. Until they run, every
training number in this README is from one seed on the CPU, at the shorter settings shown above.

## What was not measured

- No real VLM was run. The coordinate conventions come from the papers and official code. The lab writes and
  parses their formats, but it never asked Qwen2.5-VL, Qwen3-VL, PaliGemma or Florence-2 for a box.
- No paper result is reproduced: not VLM-R1's or Visual-RFT's RL gains, not LISA's ReasonSeg scores, not Molmo's
  point-then-count, not Text4Seg's 74%, not the M3Grounder, DoCoG or DocLayout-VL numbers.
- The decoder is a dot product between a prompt and 32-channel pixel features, not SAM. The language model is a
  two-layer bidirectional Transformer, not a causal LLM.
- The toy query names a colour. It tests the interface, not reasoning about relations or implicit queries.
- Token counts use two real tokenizers that split digits (SmolLM2 and Qwen2.5). Other models' tokenizers differ.
- Grounded reasoning chains (Figure 8) and video grounding are covered only by the metric checks in `07_metrics.py`.
