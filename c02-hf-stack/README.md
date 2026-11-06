# C2: The Hugging Face Stack

We train a model in raw PyTorch in C1. This tutorial asks what the Hugging Face stack automates. The package `hfstack`
writes each mechanism out by hand, then checks it against the library: a raw training loop against `Trainer`, a plain
loop against `generate()`, a parser against the safetensors file format, and the LoRA formula against PEFT. It ends with a
custom LLM and a custom VLM that plug into the Auto classes.

Tutorial: https://srini-anirudh.github.io/blogs/the-hugging-face-stack/

Code: https://github.com/srini-anirudh/blog-coding-tutorials/tree/main/c02-hf-stack

## Layout

```
c02-hf-stack/
├── hfstack/
│   ├── utils.py            # pick_device, hardware_label, be_polite, tiny_llama_config, count_parameters
│   ├── inspection.py       # config_fields, shape_of, describe_outputs
│   ├── chat.py             # chat_ids and assistant-only labels, by prefix diffing and by generation tags
│   ├── collators.py        # CausalLMCollator, VisionChatCollator
│   ├── data.py             # synthetic arithmetic rows; SFT, preference and RL formats; Dataset helpers
│   ├── training.py         # LoopConfig, train_raw, training_arguments, train_with_trainer
│   ├── custom_trainer.py   # ZLossTrainer: a compute_loss override
│   ├── accelerate_loop.py  # train_accelerate
│   ├── generation.py       # greedy_no_cache, process_logits, generate_with_cache
│   ├── lora.py             # add_lora, manual_lora_forward, merge_matches
│   ├── trl_runs.py         # dataset builders per trainer, exact_answer_reward
│   ├── hub.py              # read_safetensors_header, read_tensor_by_hand, read_index
│   ├── custom_model.py     # BackwardPassConfig, BackwardPassModel, BackwardPassForCausalLM, register
│   └── custom_vlm.py       # BackwardPassVLMConfig, BackwardPassVLMForConditionalGeneration, register
├── scripts/                # one runnable demo per group of spec sections
├── benchmarks/             # bench_map, bench_loops, bench_generate
├── results/                # captured output of every script and benchmark
└── tests/
```

## Scripts

| Script | Sections it covers |
|---|---|
| `01_config_and_auto.py` | `PretrainedConfig`, `PreTrainedModel`, Auto dispatch |
| `02_tokenizer.py` | Encode, decode, BOS, EOS, PAD, padding side, truncation, added tokens, resizing |
| `03_chat_template.py` | Messages to text to token IDs to assistant-only labels |
| `04_processor.py` | `AutoProcessor` on SmolVLM, real multimodal batches, `image_grid_thw` on Qwen2-VL |
| `05_datasets.py` | Arrow, `map`, `filter`, `shuffle`, `select`, fingerprints, caching, streaming |
| `06_collator.py` | Custom collator, `-100`, and the library default that drops real EOS labels |
| `07_raw_vs_trainer.py` | The same run as a raw loop and through `Trainer`; the accumulation bug |
| `07b_trainer_eval_checkpoint.py` | Evaluation, callbacks, checkpoints, resume |
| `08_custom_trainer.py` | `compute_loss` override with a z-loss term |
| `09_accelerate.py` | The loop under Accelerate, on one process and under DDP on CPU |
| `10_outputs_and_generate.py` | Model outputs and shapes, KV cache growth, `generate()` as a loop |
| `11_lora.py` | LoRA: target modules, trainable parameters, adapter checkpoint, merge |
| `11b_lora_seeds.py` | `11_lora.py` over seeds 0, 1 and 2, plus an exact repeat of seed 0; `--device`, `--prefix` |
| `12_trl.py` | Dataset format for SFT, DPO, Reward and GRPO, with a very short run of each |
| `13_hub_and_safetensors.py` | Files on disk and on the Hub, the safetensors layout, shards, index files |
| `14_custom_model.py` | `nn.Module` against `PreTrainedModel`; a custom model through the whole stack |
| `15_custom_vlm.py` | Sub-configs, image token, multimodal forward |
| `16_side_checks.py` | Hash time of a mapped function, with repeats; the SmolVLM tile grid |

## Run

From the repository root, after `pip install -e ".[dev]"`. Most scripts download small public files from the Hub. No
token is needed.

```bash
# Tests.
python -m pytest c02-hf-stack -m "not slow and not gpu and not network"   # offline
python -m pytest c02-hf-stack -m network                                  # downloads tokenizers and processors
python -m pytest c02-hf-stack -m slow                                     # two-process DDP on CPU

# Demos, in tutorial order.
for s in 01_config_and_auto 02_tokenizer 03_chat_template 04_processor 05_datasets 06_collator \
         07_raw_vs_trainer 07b_trainer_eval_checkpoint 08_custom_trainer 09_accelerate \
         10_outputs_and_generate 11_lora 12_trl 13_hub_and_safetensors 14_custom_model 15_custom_vlm 16_side_checks; do
  python c02-hf-stack/scripts/$s.py
done

# Accelerate under DDP, two CPU processes.
OMP_NUM_THREADS=1 python -m torch.distributed.run --nproc_per_node 2 --master_port 29402 \
    c02-hf-stack/scripts/09_accelerate.py --ddp

# Options.
python c02-hf-stack/scripts/11_lora.py --steps 150 --device auto --seed 0   # auto picks CUDA, then MPS, then CPU
python c02-hf-stack/scripts/11b_lora_seeds.py --seeds 0 1 2                # writes 11_lora*.txt and 11_lora_seeds.txt

# Benchmarks. They print tables.
python c02-hf-stack/benchmarks/bench_map.py --rows 20000
python c02-hf-stack/benchmarks/bench_loops.py
python c02-hf-stack/benchmarks/bench_generate.py --device cpu --new 64
```

## Results

Hardware: AMD EPYC 7R13, 8 threads, CPU only. Library versions: torch 2.15.0, transformers 5.22.2, datasets 5.1.0,
peft 0.21.2, trl 1.17.1, accelerate 1.15.0. Every file in `results/` comes from this machine, except the `gpu_*` files
(see "GPU results" below).

The machine was shared with other jobs (load average above 70 during the benchmarks), and each timing is a single run.
Treat timings as orders of magnitude. Differences between loops of a few percent are noise. The equivalence numbers do
not depend on load.

Tests: 51 passed offline, 11 passed with `network`, 1 passed with `slow`. No test uses a GPU.

| Measurement | Result |
|---|---|
| Raw loop vs `Trainer`, 60 steps, largest loss gap | 1.43e-06 |
| Raw loop vs `Trainer`, 60 steps, largest gradient-norm gap | 0.00e+00 |
| Accumulation bug (average each micro-batch), largest loss gap vs `Trainer` | 1.16e-01 |
| Accelerate vs raw loop, one process, largest loss gap | 4.77e-07 |
| 2 processes with batch 4 vs 1 process with batch 8, DDP over gloo, largest loss gap | 9.54e-07 |
| `ZLossTrainer` with weight 0 vs default `Trainer`, largest loss gap | 9.54e-07 |
| Resume from checkpoint-20 vs uninterrupted run, largest weight gap | 0.00e+00 |
| Greedy and seeded sampling, our loop vs `model.generate`, SmolLM2-135M | identical tokens |
| Real end-of-turn tokens labelled when pad equals EOS: ours / library collator | 9 of 9 / 0 of 9 |
| SmolVLM, 640x480 image: tiles, image tokens per tile, total image tokens | 13, 64, 832 |
| `save_pretrained` of SmolLM2-135M-Instruct in float32 | 538,090,408 bytes; 272 tensors |
| LoRA r=8 on the 4 attention projections: trainable parameters | 921,600 of 135,436,608 (0.68%) |
| LoRA adapter file vs full float32 model | 3,717,656 bytes, 0.69% |
| LoRA merge, largest logit gap, seeds 0 / 1 / 2 | 1.03e-04 / 8.39e-05 / 7.82e-05 |
| LoRA training, 150 steps, batch 16, seeds 0 / 1 / 2: final logged loss | 0.295 / 0.326 / 0.267 (1.08 to 1.30 s/step) |
| Exact-match accuracy on 182 held-out sums, before / after LoRA, seeds 0 / 1 / 2 | 0.000 / 0.593, 0.560, 0.566 (mean 0.573) |
| Seed 0 run twice | identical accuracy, loss curve, merge gap and adapter hash |
| `Hasher.hash` of the mapped tokenize function, 9 repeats x 3 processes: median per process | 0.460 / 0.521 / 0.461 s (range 0.434 to 0.548 s) |
| SmolVLM tile grid: 640x480 / 300x300 | 3 x 4 + 1 global = 13 / 4 x 4 + 1 global = 17 |
| `Dataset.map` tokenizing 20,000 rows: per row / batched / batched with 2 processes | 3.44 / 1.07 / 1.08 s |
| 100 training steps, 4-layer Llama: raw loop / `Trainer` / Accelerate | 3.21 / 3.50 / 2.95 s |
| 64 new tokens on SmolLM2-135M: no cache / our KV-cache loop / `model.generate` | 11.9 / 16.9 / 14.3 tokens/s |

How to read the weak or surprising results:

- **LoRA "before" is 0.000 mostly because of format.** The base model starts every reply with an explanation such as
  "To add 86 and", and the check needs the whole reply to be the answer within 6 tokens. So the jump measures learning
  to answer directly as well as arithmetic. Across seeds 0, 1 and 2 the "after" score is 0.560 to 0.593. Three seeds
  give a range, not a confidence interval.
- **The merge gap depends on the seed.** It is 7.82e-05 to 1.03e-04 on SmolLM2-135M. `test_merge_preserves_logits`
  uses a bound of 1e-4, but on the tiny test model, so the bound does not cover SmolLM2.
- **Hash timing varies with load.** The first hash of the tokenizer in a process is the slowest (0.72 to 0.91 s).
- **Two processes did not beat one in `bench_map`.** 20,000 short rows are too few to repay the cost of starting workers.
- **GRPO learns nothing here.** The tiny random model never earns a reward, so the loss and gradient norm are both 0.0.
  The DPO and Reward losses also stay near ln 2. The TRL runs check the data contract, not quality. C6 trains real
  models with these losses.
- **`07b` eval loss does not fall** (6.27 to 6.30), because the data is random tokens. That script is about the machinery.

## GPU results

Hardware: one NVIDIA H100 80GB HBM3, torch 2.15.0, CUDA 13.2, float32, with no other job on the GPU. Nothing in this
tutorial needs a GPU. These are the same commands with `--device cuda`, and the settings match the CPU runs.

```bash
python c02-hf-stack/benchmarks/bench_generate.py --device cuda                      # results/gpu_bench_generate.txt
python c02-hf-stack/scripts/11b_lora_seeds.py --device cuda --prefix gpu_           # results/gpu_11_lora*.txt
```

| Measurement | CPU (AMD EPYC 7R13, 8 threads) | GPU (NVIDIA H100) |
|---|---|---|
| 64 new tokens on SmolLM2-135M, batch 1: no cache / KV-cache loop / `model.generate` | 11.9 / 16.9 / 14.3 tokens/s | 26.9 / 29.7 / 20.3 tokens/s |
| LoRA, 150 steps, seconds per step, seeds 0 / 1 / 2 | 1.20 / 1.30 / 1.08 | 0.22 / 0.29 / 0.28 |
| LoRA exact match after training, seeds 0 / 1 / 2 | 0.593 / 0.560 / 0.566 | 0.593 / 0.560 / 0.566 |
| LoRA final logged loss, seeds 0 / 1 / 2 | 0.295 / 0.326 / 0.267 | 0.295 / 0.326 / 0.267 |
| LoRA merge gap, seeds 0 / 1 / 2 | 1.03e-04 / 8.39e-05 / 7.82e-05 | 8.20e-05 / 5.91e-05 / 4.72e-05 |
| Seed 0 run twice on the same device | identical, same adapter hash | identical, same adapter hash |

How to read them:

- **Batch-1 generation on a 135M model barely uses an H100.** Each new token is one small forward pass, so the time goes
  to kernel launches and Python, not to arithmetic. That is why the cache helps far less on the GPU than on the CPU, and
  why `model.generate` (which does more bookkeeping per step) is the slowest of the three there.
- **The two devices agree to the printed digits on accuracy and loss for each seed.** The adapter files differ, as the
  hash shows, so the weights match only up to float rounding. The merge gap also differs for the same reason.
- No change was needed for the GPU path. Before this run, `--device cuda` had never been run.

## Not run, or not measured

- **`push_to_hub`** is shown as code and is not run, because no token was available. `13_hub_and_safetensors.py` reads
  the file list of a real Hub repo, which needs no token.
- **The Qwen2-VL video processor** is unused, because torchvision is not installed, and `AutoProcessor` raises
  `ImportError` for Qwen2-VL without it. `04_processor.py` loads only the Qwen2-VL image processor. `torchvision` is not
  a dependency of this repo.
- **FSDP, DeepSpeed, `bf16` and `fp16`** are named in the ecosystem map and in `TrainingArguments`, and are not run here.

## Version notes

- `warmup_ratio` is not a `TrainingArguments` field in transformers 5.22.2. A float below 1 in `warmup_steps` is a ratio.
- A Hub streaming iterator left half-read stalled interpreter exit in one test with datasets 5.1.0. `take(n)` read to
  its end exits cleanly, so `05_datasets.py` uses it.

## Theory

- [Inside a Training Step](https://srini-anirudh.github.io/blogs/inside-a-training-step/)
- [The Parallelism Playbook](https://srini-anirudh.github.io/blogs/the-parallelism-playbook/)
- [Choosing the Next Token](https://srini-anirudh.github.io/blogs/choosing-the-next-token/)
- [The Mechanics of LLM Inference](https://srini-anirudh.github.io/blogs/the-mechanics-of-llm-inference/)
- [How LoRA Changes a Model Without Rewriting It](https://srini-anirudh.github.io/blogs/how-lora-changes-a-model-without-rewriting-it/)
- [Teaching a Language Model to Follow Instructions](https://srini-anirudh.github.io/blogs/teaching-a-language-model-to-follow-instructions/)
