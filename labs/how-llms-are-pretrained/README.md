# Lab: How LLMs Are Pretrained

Code companion for the post [How LLMs Are Pretrained](https://srini-anirudh.github.io/blogs/how-llms-are-pretrained/).
The post argues that the loss is one line and the data distribution is the real product. This lab builds the steps that
make that distribution, in small Python, and runs each one so you can watch it work.

What you can run:

- The one-line objective, token by token, on GPT-2.
- Gopher and C4 quality rules, exact dedup, MinHash dedup and 13-gram decontamination on a small crawl. The crawl shows
  which page types die at which stage, and that running dedup before the filters changes what survives.
- MinHash and LSH from scratch. The match rate equals Jaccard, and the candidate rate follows `1 - (1 - J^r)^b`.
- Byte-level BPE from scratch. A small vocabulary is exactly the large one cut at a merge rank. Fertility on English,
  Python and six languages, against GPT-2, SmolLM2 and Qwen2.5.
- Packed rows on real pretrained models: what the document mask fixes, and when position resets matter.
- Tiny byte-level models: the value of repeated data, a mixture search on three domains, and signal and noise in
  small-scale evaluations.

The packing helpers (document ids, position resets, the block-diagonal mask) come from
[C5](../../c05-pretraining/pretrain/packing.py). C5 also measures padding against packing on real document lengths
(`c05-pretraining/benchmarks/bench_packing.py`), so this lab does not repeat that.

## Layout

```
labs/how-llms-are-pretrained/
├── ptlab/
│   ├── corpora.py     # prose (WikiText-2), code (Python standard library), synthetic arithmetic, as byte streams
│   ├── filters.py     # Gopher and C4 rules, exact-dedup hash, 13-gram contamination check
│   ├── minhash.py     # shingles, MinHash, LSH banding, union-find clusters, the S-curve formula
│   ├── bpe.py         # byte-level BPE training and encoding, FLOP and logits-memory arithmetic
│   ├── packing.py     # packed row vs each document alone, four mask and position settings (uses C5)
│   ├── tinylm.py      # tiny byte-level Llama, mixture sampler, training loop, multiple-choice scorer
│   ├── mixing.py      # simplex sampling, quadratic surrogate (RegMix-style), data mixing law
│   ├── repetition.py  # effective data under repetition (Muennighoff et al.), R* fit
│   └── evalnoise.py   # signal, noise, decision accuracy
├── scripts/           # 01 to 08, one per post section, listed below
├── results/           # real outputs; the first lines of each file give the hardware and the command
└── tests/
```

## Setup

From the repository root, after `pip install -e ".[dev]"`. WikiText-2 (about 13 MB) and three small models download
from the Hugging Face Hub on first use. The code domain is your Python's own standard library, so it needs no download.
PyTorch uses every core by default, so set `OMP_NUM_THREADS` on a shared machine. The CPU numbers below used 8 threads.

## Run everything

```bash
python labs/how-llms-are-pretrained/scripts/01_objective.py        # about 30 s, CPU
python labs/how-llms-are-pretrained/scripts/02_pipeline_funnel.py  # about 25 s, CPU
python labs/how-llms-are-pretrained/scripts/03_minhash_lsh.py      # about 15 s, CPU
python labs/how-llms-are-pretrained/scripts/04_bpe_fertility.py    # about 1 min, CPU
python labs/how-llms-are-pretrained/scripts/05_packing_masks.py    # about 30 s, CPU
python labs/how-llms-are-pretrained/scripts/06_repetition.py --device cuda             # 7 tiny runs
python labs/how-llms-are-pretrained/scripts/07_mixture_search.py --device cuda --size s --fit
python labs/how-llms-are-pretrained/scripts/07_mixture_search.py --device cuda --size m
python labs/how-llms-are-pretrained/scripts/08_signal_noise.py     # reads the two 07 result files
```

Scripts 06 and 07 train many small models, so they used a GPU: each 600 to 1,000 step run takes 9 to 14 s there. They also
run on a CPU with `--device cpu`, but a 40-step trial of 3 runs took about a minute on 8 threads, so the full set would
take hours. Pass fewer `--steps` or `--runs` to try them on a CPU, and do not compare those numbers with the GPU tables.

## Test

```bash
python -m pytest labs/how-llms-are-pretrained -m "not slow and not gpu and not network"   # 20 passed, about 15 s
python -m pytest labs/how-llms-are-pretrained -m network                                  # 1 passed
```

The offline tests check the quality rules on their targets, the MinHash match rate against Jaccard, the LSH candidate
rate against its formula, dedup clusters, the BPE prefix property and round trip, both packing results on randomly
initialised RoPE and absolute-position models, the training loop, both mixture fits on data of known form, the
effective-data formula and the R* fit, and the signal and noise arithmetic.

## Results

Hardware: scripts 01 to 05 and 08 ran on **AMD EPYC 7R13, 8 threads, torch 2.15.0, CPU only**. Scripts 06 and 07 ran on
**NVIDIA H100 80GB HBM3, torch 2.15.0, CUDA 13.2**, one GPU each, bf16 autocast. Each result file starts with its own
hardware line and command. All tiny-model numbers are from one seed per run unless a column says otherwise.

### Objective, filters, dedup, BPE, packing

| Check | Result | File |
|---|---|---|
| GPT-2 loss on one 13-token sentence | 4.7193 nats per token, equal to the Hugging Face loss | `01_objective.txt` |
| MinHash match rate against true Jaccard (112 hashes, 400 pairs per J) | J = 0.498 gives 0.497; J = 0.802 gives 0.802 | `03_minhash_lsh.txt` |
| LSH candidate rate at b = 14, r = 8 | J = 0.702: 0.540 measured, 0.574 formula. J = 0.802: 0.930 and 0.927 | `03_minhash_lsh.txt` |
| 5% of words changed in a real article | true J 0.633, LSH does not flag it | `03_minhash_lsh.txt` |
| Toy crawl of 650 pages | 283 survive, 47.4% of bytes. The crawl is far cleaner than a real one | `02_pipeline_funnel.txt` |
| Dedup before the quality rules, on 60 articles crawled first as excerpt copies | 13 originals kept, against 56 when the rules run first | `02_pipeline_funnel.txt` |
| BPE, 500 merges against the first 500 of 2,000 merges | identical merges and identical encodings of 256,081 validation characters. Same prefix property in the Hugging Face trainer | `04_bpe_fertility.txt` |
| Bytes per token on Python code | ours (756 tokens) 1.45; GPT-2 2.17; SmolLM2 3.71; Qwen2.5 4.00 | `04_bpe_fertility.txt` |
| Tokens for one sentence in Telugu against English | GPT-2 15.2x, SmolLM2 15.2x, Qwen2.5 8.5x | `04_bpe_fertility.txt` |
| fp32 logits, batch 8, 8,192 tokens, 100,000 vocabulary | 26.2 GB | `04_bpe_fertility.txt` |

Packed rows, largest per-token loss difference from running the document alone (nats, fp32, three short documents):

| Setting | GPT-2 (absolute positions), documents 2 and 3 | SmolLM2-135M (RoPE), documents 2 and 3 |
|---|---|---|
| Causal mask, positions keep counting | 2.23 and 3.14 | 1.76 and 2.95 |
| Causal mask, positions reset | 2.94 and 10.1 | 5.97 and 11.4 |
| Document mask, positions keep counting | 14.6 and 10.1 | 4.4e-6 and 1.1e-5 |
| Document mask, positions reset | 0 and 0 | 0 and 0 |

The document mask alone makes packing exact for the RoPE model. The absolute-position model needs the mask and the reset.
Resetting positions without a mask does not help, because tokens still attend to the earlier documents.

### What is an epoch? (`06_repetition.py`)

Tiny byte-level Llama, 672,640 parameters. Every run sees 8,192,000 training bytes of WikiText-2, drawn from a pool of
`8,192,000 / E` unique bytes. Validation loss is in bits per byte on 256 fixed windows. "Fresh bytes needed" is read off the
single-epoch run's loss curve.

| Epochs E | Unique bytes | Final val bpb | Fresh bytes that give the same loss | Value kept (that / 8.19M) | Formula, R* = 15.4 |
|---:|---:|---:|---:|---:|---:|
| 1 | 8,192,000 | 2.2224 | 8,192,000 | 1.00 | 1.00 |
| 2 | 4,096,000 | 2.2566 | 7,170,938 | 0.88 | 0.98 |
| 4 | 2,048,000 | 2.2842 | 6,390,195 | 0.78 | 0.93 |
| 8 | 1,024,000 | 2.3510 | 4,885,600 | 0.60 | 0.83 |
| 16 | 512,000 | 2.5309 | 2,918,772 | 0.36 | 0.66 |
| 32 | 256,000 | 3.2393 | 828,188 | 0.10 | 0.38 |
| 64 | 128,000 | 5.1553 | 204,800 | 0.03 | 0.16 |

The shape matches the paper: the first repeats are nearly free and the value falls as repeats pile up. The numbers do not.
Fitting R* to these runs gives 2.5, against the paper's 15.4, and four epochs already cost 22% of the value. This model is
very small and its learning rate is held constant, so it memorises a small pool faster than the paper's models did. The
paper's fit used models up to 9B parameters. This does not test the paper's result.
The 32 and 64 epoch runs overfit hard: their best checkpoint (2.71 and 2.85 bpb) is far below their final loss.

### The mixture problem (`07_mixture_search.py`)

Three domains: WikiText-2 prose, Python standard-library code, synthetic arithmetic. Target is the mean of the three
validation losses in bits per byte. 16 random mixtures (Dirichlet(1)), 600 steps of 32 sequences of 256 bytes. Small model
672,640 parameters (about 9 s per run), larger model 3,475,712 parameters (about 11 s per run).

- Across the 16 mixtures the small model's target loss runs from 2.230 to 2.861 bpb. Putting no weight on a domain costs
  it dearly: with 0.000 weight on math, math loss is 3.96 bpb, against 1.94 to 2.32 in the other runs.
- Leave-one-out error on the target: quadratic surrogate 0.107 bpb, mixing law 0.102 bpb. Both predict held-out mixtures to
  about a tenth of a bit, which is a sixth of the spread. Neither is accurate enough to place the optimum.

| Mixture (prose / code / math) | Predicted | Measured, seed 0 | Measured, seed 1 |
|---|---:|---:|---:|
| Uniform (0.33 / 0.33 / 0.33) | 2.285 | 2.312 | 2.372 |
| Hand table (0.60 / 0.25 / 0.15) | 2.306 | 2.288 | 2.321 |
| Best of the 16 random runs (0.47 / 0.44 / 0.08) | 2.270 | 2.235 | 2.276 |
| Quadratic-surrogate optimum (0.28 / 0.48 / 0.24) | 2.243 | 2.279 | 2.299 |
| Mixing-law optimum (0.28 / 0.60 / 0.12) | 2.212 | 2.252 | 2.265 |

The "predicted" column is the mixing law's prediction, except on the surrogate row, where it is the surrogate's. The fitted
optima did not beat the best random probe. All five mixtures sit within 0.11 bpb of each other, and changing the seed alone
moves one mixture by up to 0.06. That is the flat surface the post describes: hand tuning is hard to beat, and telling
good mixtures apart needs more care than one seed.

### Signal and noise (`08_signal_noise.py`)

The same 16 recipes at two sizes. Signal is the standard deviation of the final score across recipes. Noise is the
standard deviation of one run's score over its last 5 checkpoints (20 steps apart) after removing a straight-line trend.
Decision accuracy is the share of recipe pairs ordered the same way by the small and the larger model.

| Small-model metric | Signal | Noise | SNR | Decision accuracy |
|---|---:|---:|---:|---:|
| Mean loss, 256 windows | 0.185 | 0.0029 | 64.8 | 0.88 |
| Mean loss, 8 windows | 0.189 | 0.0052 | 36.2 | 0.84 |
| Prose loss, 256 windows | 0.331 | 0.0041 | 79.9 | 0.93 |
| Code loss, 256 windows | 0.359 | 0.0040 | 90.2 | 0.97 |
| Math loss, 256 windows | 0.464 | 0.0039 | 119.1 | 0.91 |
| Arithmetic test error, 200 items | 0.051 | 0.0185 | 2.7 | 0.78 |
| Arithmetic test error, 20 items | 0.063 | 0.0643 | 1.0 | 0.64 |

The arithmetic test is at chance for every model (accuracy 0.19 to 0.36 at both sizes, chance 0.25), so it is mostly noise.
This matches the post's point that small models sit at chance on the benchmarks that matter. The 20-item version cannot
separate recipes: its noise is as large as its signal.
Across all 10 metrics, log SNR against decision accuracy gives R² = 0.82. That is ten correlated metrics from one toy
suite, so it is a direction only. The paper's 0.63 comes from 225 models and 25 corpora and was not reproduced.
The noise here is spread across checkpoints of one run. It does not include seed-to-seed noise, which section 07 shows is
larger (up to 0.06 bpb on the target).

## Post sections and scripts

| Post section | Script | What it shows |
|---|---|---|
| A one-line objective, Figure 01 | `01_objective.py` | each prediction, the probability of the true token, and the summed loss |
| Raw web is not training data, Figure 03 | `02_pipeline_funnel.py` | survivors per stage and page type, the rule that killed each, order effects |
| Deduplication, "Why MinHash works", Figure 04 | `03_minhash_lsh.py`, `02_pipeline_funnel.py` | match rate = J, the S-curve, real edited copies |
| Tokenization, Figure 07 | `04_bpe_fertility.py` | BPE from scratch, truncation by merge rank, fertility, digits, cost |
| Sequence construction, Figure 08 | `05_packing_masks.py` | document mask and position resets on GPT-2 and SmolLM2 |
| What is an epoch?, Figure 10 | `06_repetition.py` | the effective-data formula and a measured version on tiny models |
| The mixture problem, Figure 06 | `07_mixture_search.py` | random-mixture probes, a quadratic surrogate, a mixing law, checks |
| Knowing whether it worked, Figure 11 | `08_signal_noise.py` | signal, noise, SNR and decision accuracy across 10 small-scale metrics |

## What was not measured

- Nothing here runs at the post's scale. Every paper number the post quotes (FineWeb, DCLM, FineWeb-Edu, DoReMi,
  RegMix, Olmo 3, the 4-epoch result, the R² of 0.63 for SNR) is from the papers and was not reproduced.
- HTML extraction, language ID, safety filters, domain classifiers, suffix-array dedup and embedding dedup are not built.
- Quality classifiers (fastText, FineWeb-Edu) are not trained, so Figure 05's comparison of six proxies is not shown.
- Global against per-snapshot dedup needs training runs on many crawl snapshots. Not done.
- DoReMi, Aioli and online mixing are not built. The mixture search uses offline probes only.
- Curriculum, midtraining and annealing (Figure 09) are not run.
- The crawl in `02_pipeline_funnel.py` mixes real text with constructed page types. Its survival rates describe that
  crawl only, not Common Crawl.
- The multilingual sentences in `04_bpe_fertility.py` were translated for this lab. They are close in meaning, which is
  all a token count needs.
