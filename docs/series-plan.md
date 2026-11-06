# The Backward Pass — Coding Tutorial Series

> **Build the thing. Inspect every tensor. Train it. Break it. Profile it. Scale it.**

This series is the implementation companion to the theory blogs on **The Backward Pass**.

The goal is **not** to create dozens of small API tutorials.

Each article should be a substantial end-to-end build where the reader understands:

1. what is being implemented,
2. why the implementation looks that way,
3. what every major tensor contains,
4. how the abstraction maps back to theory,
5. how the implementation changes when scaled,
6. how to test correctness,
7. how to profile performance,
8. how production/research frameworks implement the same idea.

---

# Series Philosophy

The series should progressively move through four abstraction levels:

```text
LEVEL 1 — RAW MECHANICS

PyTorch tensors
autograd
attention
manual training loops


        ↓


LEVEL 2 — MODEL BUILDING

LLMs
VLMs
generation
KV cache


        ↓


LEVEL 3 — TRAINING SYSTEMS

pretraining
SFT
DPO
RL
distributed training
multimodal mixtures


        ↓


LEVEL 4 — AGENTIC SYSTEMS

tool use
agent harnesses
memory
environment interaction
agent RL
```

Theoretical derivations should generally **not be repeated**.

Instead:

> For the derivation, see the corresponding theory article.  
> Here, we implement it and inspect what actually happens.

---

# Final Coding Tutorial Sequence

| # | Tutorial |
|---|---|
| **C1** | PyTorch From First Principles |
| **C2** | The Hugging Face Stack — Transformers, Datasets, Trainer & Friends |
| **C3** | Build Attention From Scratch — MHA → GQA → MLA → FlashAttention |
| **C4** | Build a Modern LLM From Scratch |
| **C5** | Train an LLM From Scratch |
| **C6** | Post-Train an LLM — SFT → DPO → GRPO |
| **C7** | Build a Vision-Language Model From Scratch |
| **C8** | Train a Real VLM |
| **C9** | Post-Train a VLM — Preferences, Hallucination & RL |
| **C10** | Build an Agent Harness From Scratch |
| **C11** | Add Memory & Context Engineering to the Agent |
| **C12** | Train an Agent With Reinforcement Learning |

---

# C1 — PyTorch From First Principles

## Goal

Understand PyTorch deeply enough that later abstractions such as `Trainer`, FSDP, PEFT, TRL, or VLM frameworks are not magic.

Build a small neural network and training loop almost entirely from raw PyTorch.

---

## 1. Tensors Are the Fundamental Object

Start with:

```python
import torch

x = torch.randn(4, 8)
```

Cover:

- shape
- rank
- dimensions
- dtype
- device
- storage
- element size
- memory footprint

Calculate:

\[
M = \text{numel}(x)\times \text{bytes-per-element}
\]

---

## 2. Tensor Storage, Strides and Views

Explain:

```python
x.shape
x.stride()
x.storage_offset()
```

Implement and inspect:

- `view`
- `reshape`
- `transpose`
- `permute`
- `contiguous`
- slicing

Show why:

```python
x.transpose(...)
```

does not necessarily move data.

Explain:

```text
logical tensor layout
        ≠
physical storage layout
```

---

## 3. Broadcasting

Implement examples:

\[
[B,N,D] + [D]
\]

and:

\[
[B,H,N,D] + [1,H,1,D].
\]

Explain broadcasting rules visually.

This is important preparation for attention.

---

## 4. CPU and GPU Tensors

Cover:

```python
x.cuda()
x.to(device)
```

Explain:

- host memory
- GPU memory
- device transfer
- synchronization
- pinned memory
- asynchronous copies

Do a small transfer benchmark.

---

## 5. Matrix Multiplication

Implement:

```python
torch.matmul
torch.mm
torch.bmm
```

Understand:

\[
[B,M,K]
\times
[B,K,N]
\rightarrow
[B,M,N].
\]

This will later become attention and Transformer GEMMs.

---

## 6. Autograd

Start from:

```python
x = torch.tensor(..., requires_grad=True)
```

Inspect:

```python
x.requires_grad
y.grad_fn
```

Then:

```python
loss.backward()
x.grad
```

Cover:

- computation graph
- leaf tensors
- non-leaf tensors
- gradient accumulation
- `retain_graph`
- `detach`
- `torch.no_grad`

Cross-link:

**Backpropagation & Autograd From First Principles**

---

## 7. Implement a Linear Layer Manually

Instead of:

```python
nn.Linear
```

start with:

\[
y=xW+b.
\]

Create:

```python
W = torch.nn.Parameter(...)
b = torch.nn.Parameter(...)
```

Then build your own:

```python
class Linear(nn.Module):
    ...
```

---

## 8. `nn.Module`

Explain:

- parameters
- submodules
- buffers
- `state_dict`
- `train()`
- `eval()`

Build:

```python
class MLP(nn.Module):
    ...
```

---

## 9. Loss Functions

Implement:

- MSE
- cross entropy

Before using:

```python
F.cross_entropy(...)
```

derive and implement:

```python
log_softmax + gather
```

---

## 10. Optimizers

Implement vanilla SGD manually:

```python
with torch.no_grad():
    p -= lr * p.grad
```

Then switch to:

```python
torch.optim.SGD
torch.optim.AdamW
```

Cross-link:

**How Models Take a Step**

---

## 11. Dataset and DataLoader

Implement:

```python
Dataset
DataLoader
```

Cover:

- batching
- shuffle
- workers
- collate functions
- pinned memory
- prefetch

---

## 12. Write the Full Training Loop

```python
for batch in loader:
    optimizer.zero_grad()

    logits = model(x)
    loss = criterion(logits, y)

    loss.backward()
    optimizer.step()
```

Explain every line.

---

## 13. Debugging

Teach readers to inspect:

```text
parameter norm
gradient norm
activation statistics
NaNs
shape mismatch
device mismatch
```

Use hooks where useful.

---

## Final Deliverable

A small but clean raw-PyTorch neural-network training project.

Reader should be able to explain exactly what happens between:

```python
loss.backward()
```

and:

```python
optimizer.step()
```

---

# C2 — The Hugging Face Stack

## Transformers, Datasets, Trainer, Accelerate, PEFT, TRL & Hub

## Core Question

> We now know how to train a model in raw PyTorch. What does the Hugging Face ecosystem automate for us?

---

# 1. Ecosystem Map

Start with:

```text
                         Hugging Face Hub
                    models / datasets / configs
                              │
          ┌───────────────────┼──────────────────┐
          │                   │                  │
     Transformers          Datasets          Tokenizers
          │                   │                  │
          └───────────────┬───┴──────────────────┘
                          ↓
                       Trainer
                          │
                     Accelerate
                          │
              PyTorch / FSDP / DeepSpeed
                          │
           ┌──────────────┴──────────────┐
           │                             │
         PEFT                           TRL
    LoRA / adapters             SFT / DPO / GRPO
```

---

# 2. `PretrainedConfig`

Load:

```python
config = AutoConfig.from_pretrained(...)
```

Inspect:

```text
hidden_size
num_hidden_layers
num_attention_heads
num_key_value_heads
intermediate_size
vocab_size
rope_theta
```

Explain why model architecture is serialized separately from weights.

---

# 3. `PreTrainedModel`

Compare:

```python
class MyModel(nn.Module)
```

with:

```python
class MyModel(PreTrainedModel)
```

Explain what HF adds:

- config
- standardized initialization
- loading
- saving
- Hub compatibility
- generation integration
- checkpoint handling

---

# 4. `AutoModel`

Explain dispatch:

```text
config.model_type
        ↓
architecture registry
        ↓
LlamaForCausalLM
```

Cover:

```python
AutoModel
AutoModelForCausalLM
AutoModelForSequenceClassification
```

---

# 5. Tokenizers

Use:

```python
AutoTokenizer
```

Inspect:

```text
input_ids
attention_mask
```

Cover:

- encode/decode
- BOS
- EOS
- PAD
- padding side
- truncation
- added tokens
- vocabulary
- tokenizer resizing

---

# 6. Chat Templates

Use:

```python
tokenizer.apply_chat_template(...)
```

Show how:

```python
[
    {"role": "system", ...},
    {"role": "user", ...},
    {"role": "assistant", ...},
]
```

becomes actual token sequences.

Cross-link:

**From Base Model to Assistant — SFT**

---

# 7. VLM Processors

Explain:

```python
AutoProcessor
```

as roughly:

```text
tokenizer
+
image processor
+
video processor
```

Inspect real multimodal batches:

```text
input_ids
attention_mask
pixel_values
image_grid_thw
...
```

---

# 8. Hugging Face Datasets

Cover:

```python
load_dataset
Dataset
DatasetDict
IterableDataset
```

Explain Apache Arrow and why this differs from:

```python
list[dict]
```

---

# 9. Dataset Transformations

Implement:

```python
dataset.map(...)
dataset.filter(...)
dataset.shuffle(...)
dataset.select(...)
```

Cover:

- batched mapping
- multiprocessing
- caching
- fingerprints
- column removal

---

# 10. Streaming

Use:

```python
streaming=True
```

Explain:

```text
Dataset
      → random-access corpus

IterableDataset
      → stream of examples
```

Discuss why foundation-model datasets often need streaming.

---

# 11. Data Collators

Explain the distinction:

```text
Dataset
   ↓
individual examples

Data Collator
   ↓
batch tensors
```

Implement custom collator for:

- padding
- labels
- `-100`
- multimodal examples

---

# 12. Trainer

Start from raw loop:

```python
for batch in loader:
    ...
```

Then show:

```python
Trainer(...).train()
```

Map:

| Raw PyTorch | Trainer |
|---|---|
| DataLoader | internal |
| forward | internal |
| backward | internal |
| optimizer | internal/custom |
| scheduler | internal/custom |
| DDP | backend |
| checkpoints | built in |
| evaluation | built in |
| logging | callbacks |

---

# 13. TrainingArguments

Organize by function.

### Batch

```text
per_device_train_batch_size
gradient_accumulation_steps
```

### Optimization

```text
learning_rate
weight_decay
warmup_ratio
lr_scheduler_type
```

### Precision

```text
bf16
fp16
```

### Checkpointing

### Evaluation

### Logging

### Distributed training

Do not enumerate every option.

---

# 14. Custom Trainer

Implement:

```python
class CustomTrainer(Trainer):
    def compute_loss(...):
        ...
```

Then discuss customization levels:

```text
Trainer
   ↓
Subclass Trainer
   ↓
Accelerate
   ↓
Raw PyTorch
```

---

# 15. Accelerate

Explain:

\[
\boxed{
Trainer=\text{training-loop abstraction}
}
\]

\[
\boxed{
Accelerate=\text{execution/distributed abstraction}
}
\]

Implement a simple training loop with:

```python
Accelerator()
accelerator.prepare(...)
accelerator.backward(...)
```

---

# 16. Model Outputs

Inspect:

```python
outputs.loss
outputs.logits
outputs.hidden_states
outputs.attentions
outputs.past_key_values
```

Understand shapes.

---

# 17. `generate()`

Demystify:

```text
generate()
 │
 ├── model forward
 ├── KV cache
 ├── logits processors
 ├── sampler
 ├── stopping criteria
 └── repeat
```

Cross-link:

**From Logits to Tokens**

---

# 18. PEFT

Use:

```python
LoraConfig
get_peft_model
```

Show:

- target modules
- trainable parameters
- adapter checkpoint
- merge

Leave theory to PEFT article.

---

# 19. TRL

Explain:

```text
Trainer
 ├── SFTTrainer
 ├── DPOTrainer
 ├── RewardTrainer
 └── GRPOTrainer
```

Show dataset formats:

### SFT

```text
messages
```

### Preference

```text
prompt
chosen
rejected
```

### RL

```text
prompt
```

with online generation.

---

# 20. Hub

Cover:

```text
config.json
model.safetensors
tokenizer.json
tokenizer_config.json
generation_config.json
README.md
```

Use:

```python
save_pretrained
from_pretrained
push_to_hub
```

---

# 21. Safetensors and Sharding

Explain:

- safetensors
- weight shards
- index files
- large-model loading

---

# 22. Custom Model Integration

Implement:

```python
class BackwardPassConfig(PretrainedConfig)
class BackwardPassModel(PreTrainedModel)
class BackwardPassForCausalLM(...)
```

Register with Auto classes.

Then save/load normally.

---

# 23. Custom VLM Integration

Show the additional pieces:

```text
vision config
text config
processor
image token IDs
custom multimodal forward
```

This becomes useful later in C7/C8.

---

# Final Principle

> Hugging Face should automate mechanics you already understand—not hide them from you.

---

# C3 — Build Attention From Scratch

## From QKV to MHA, GQA, MLA, FlashAttention and Multimodal Attention

This should be a **full attention implementation zoo**.

Cross-link:

**Attention, From First Principles**

---

# 1. Scaled Dot-Product Attention

Implement:

```python
scores = q @ k.transpose(-2, -1)
scores = scores / math.sqrt(head_dim)

scores += mask

attn = torch.softmax(scores, dim=-1)
out = attn @ v
```

Shapes:

\[
Q:B\times H\times N_q\times d_h
\]

\[
K,V:B\times H\times N_k\times d_h
\]

\[
QK^\top:B\times H\times N_q\times N_k.
\]

---

# 2. Attention Masks

Implement manually:

- bidirectional
- causal
- padding
- causal + padding
- prefix/block masks
- multimodal masks

Visualize every mask.

---

# 3. Self-Attention

\[
Q=XW_Q,\quad
K=XW_K,\quad
V=XW_V.
\]

Implement single-head first.

---

# 4. Cross-Attention

\[
Q=X_qW_Q
\]

\[
K=X_sW_K,\quad
V=X_sW_V.
\]

Test:

- text → text
- text → image
- latent queries → image

---

# 5. Multi-Head Attention

Implement manually:

\[
d_{\rm model}=Hd_h.
\]

Cover:

- reshaping
- transpose
- contiguous
- output projection

---

# 6. Packed/Fused QKV Projection

Compare:

```text
three Linear layers
```

against:

```text
one QKV Linear
```

Then split.

---

# 7. Multi-Query Attention

\[
H_Q=H
\]

\[
H_{KV}=1.
\]

Implement and compare cache size.

---

# 8. Grouped-Query Attention

\[
1<H_{KV}<H_Q.
\]

Implement:

```python
num_attention_heads=32
num_key_value_heads=8
```

Derive:

\[
n_{\rm rep}
=
H_Q/H_{KV}.
\]

---

# 9. Unified MHA/MQA/GQA

One implementation parameterized by:

```python
num_heads
num_kv_heads
```

where:

```text
Hkv = Hq → MHA
Hkv = 1  → MQA
otherwise → GQA
```

---

# 10. RoPE Inside Attention

Implement:

- frequencies
- sin/cos caches
- Q rotation
- K rotation
- offsets during decoding

Cross-link positional-encoding theory.

---

# 11. KV Cache

Implement:

### Prefill

\[
N_q=N_k=N.
\]

### Decode

\[
N_q=1,\qquad
N_k=T.
\]

Implement:

- cache creation
- append/update
- preallocated cache
- positional offsets

---

# 12. KV Memory Benchmark

Verify:

\[
M_{KV}
=
2LBSh_{KV}d_hb.
\]

Compare:

- MHA
- GQA
- MQA

for long sequences.

---

# 13. Sliding-Window Attention

Implement naive mask first.

Then demonstrate:

> A sparse mask on a dense tensor is still computationally dense.

Discuss efficient implementation.

---

# 14. Local + Global Attention

Implement:

- sliding local tokens
- global special tokens

---

# 15. Block-Sparse Attention

Build pedagogical block-sparse layouts.

Explain why specialized kernels are required for actual speedups.

---

# 16. Vision Transformer Attention

Reuse MHA without causal mask.

Input:

\[
B\times N_{\rm patches}\times D.
\]

Visualize patch-to-patch attention.

---

# 17. Windowed Vision Attention

Implement:

- window partition
- local attention
- window merge
- shifted windows

Swin-style intuition.

---

# 18. 2D Positional Attention

Implement examples of:

- 2D positional embeddings
- relative bias
- 2D RoPE

---

# 19. Linear Attention

Implement kernelized form:

\[
\phi(Q)
[
\phi(K)^\top V
].
\]

Compare against:

\[
\operatorname{softmax}(QK^\top)V.
\]

Benchmark sequence-length scaling.

---

# 20. Causal Linear Attention

Maintain:

\[
S_t
=
S_{t-1}
+
\phi(k_t)v_t^\top.
\]

Then:

\[
y_t=\phi(q_t)S_t.
\]

Show connection to recurrence.

---

# 21. Performer-Style Features

Advanced section.

Implement a simplified random-feature approximation for educational purposes.

Clearly label as advanced/research.

---

# 22. Multi-Head Latent Attention

Implement progressively.

### Step 1

Compress KV state:

\[
c_t^{KV}
=
W^{DKV}h_t.
\]

### Step 2

Reconstruct components.

### Step 3

Separate positional/RoPE component.

### Step 4

Compare cache sizes to GQA.

---

# 23. Q-Former-Style Attention

Implement learned queries:

\[
Q_L\in\mathbb R^{M\times d}
\]

cross-attending to many image tokens.

Show:

\[
M\ll N_v.
\]

---

# 24. Perceiver Resampler

Implement fixed latent tokens attending to arbitrary image-token lengths.

Compare to Q-Former mechanism.

---

# 25. VLM Fusion Patterns

Implement:

### Early fusion

```text
[image tokens][text tokens]
     ↓
self attention
```

### Cross-attention

```text
text Q → image K/V
```

### Gated cross-attention

Flamingo-like conceptual block.

---

# 26. PyTorch SDPA

Replace manual implementation with:

```python
F.scaled_dot_product_attention(...)
```

Verify numerical equivalence.

---

# 27. FlashAttention Concept

Explain the IO problem:

```text
QKᵀ
 ↓
N × N in HBM
 ↓
softmax
 ↓
V
```

versus tiling.

---

# 28. Implement Online Softmax

Write a small educational tiled attention implementation using running:

\[
m_i
\]

and normalization accumulator:

\[
l_i.
\]

Do not attempt a production CUDA kernel.

---

# 29. FlashAttention Benchmark

Benchmark:

- naive PyTorch
- SDPA math backend
- memory-efficient backend
- FlashAttention backend

Measure:

- latency
- peak memory
- sequence scaling

---

# 30. Attention Dropout

Implement explicitly.

---

# 31. Attention Biases

Add modular support for:

- causal bias
- ALiBi
- relative positional bias
- local/window biases

---

# 32. QK Normalization

Implement:

```text
Q norm
K norm
```

before dot product.

Inspect attention logits.

---

# 33. Logit Soft-Capping

Implement something like:

\[
s'=c\tanh(s/c).
\]

Show effect on score distribution.

---

# 34. Advanced Attention Variants

Short implementation sections for research variants such as:

- Differential Attention
- talking-heads
- gated attention
- normalized attention

Do not treat all as equally mainstream.

---

# 35. Unified Attention Library

End with:

```python
AttentionConfig(
    attention_type="gqa",
    hidden_size=4096,
    num_heads=32,
    num_kv_heads=8,
    causal=True,
    window_size=None,
    positional_encoding="rope",
    backend="flash",
)
```

Separate:

### Architecture

```text
MHA
MQA
GQA
MLA
LinearAttention
CrossAttention
```

from:

### Execution Backend

```text
Naive
SDPA
Flash
```

---

# 36. Attention Test Suite

Every implementation must test:

### Shapes

### Causality

Changing future tokens must not affect past tokens.

### Numerical equivalence

Manual MHA vs SDPA.

### MHA/GQA equivalence

When:

\[
H_{KV}=H_Q.
\]

### MQA/GQA equivalence

When:

\[
H_{KV}=1.
\]

### KV-cache equivalence

Cached incremental output must match full-sequence output.

### Gradient flow

### FP32 vs BF16 differences

---

# 37. Benchmark the Attention Zoo

Sequence lengths:

```text
512
1K
2K
4K
8K
16K
...
```

Measure:

### Training

- forward time
- backward time
- peak memory

### Prefill

- tokens/sec
- latency

### Decode

- ms/token
- cache memory

Compare:

- MHA
- MQA
- GQA
- MLA
- local
- linear
- naive backend
- Flash backend

---

# C4 — Build a Modern LLM From Scratch

Now use the reusable attention library from C3.

## Goal

Build a modern decoder-only model closer to Llama/Qwen/DeepSeek-style design than the original 2017 Transformer.

---

# 1. Model Configuration

Define:

```python
@dataclass
class ModelConfig:
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    intermediate_size: int
    max_seq_len: int
```

---

# 2. Token Embeddings

\[
X=E[\text{ids}].
\]

Cover weight tying.

---

# 3. RMSNorm

Implement from scratch.

---

# 4. RoPE

Reuse C3 implementation.

---

# 5. GQA Attention

Choose:

```text
GQA
+
RoPE
+
Flash/SDPA backend
```

as default modern configuration.

---

# 6. SwiGLU

Implement:

\[
\operatorname{SiLU}(xW_g)
\odot
xW_u.
\]

Then down projection.

---

# 7. Transformer Block

Build:

```text
x
├→ RMSNorm → Attention → + ─┐
│                           │
└───────────────────────────┘
 ↓
RMSNorm → SwiGLU → +
```

---

# 8. Decoder Stack

Repeat \(L\) blocks.

---

# 9. LM Head

\[
H\rightarrow |V|.
\]

---

# 10. Loss

Shift:

```text
input : A B C D
target: B C D E
```

Cross entropy.

---

# 11. Initialization

Implement sensible initialization.

Cross-link:

**Keeping a Transformer Alive**

---

# 12. Parameter Count

Derive and verify approximate counts for:

- embeddings
- attention
- MLP
- norm
- total

---

# 13. Generation

Reuse sampling stack:

- greedy
- temperature
- top-k
- top-p

---

# 14. KV-Cached Generation

Use C3 cache.

Compare:

```text
no cache
vs
cache
```

---

# 15. Optional MoE Block

Implement a minimal:

```text
router
 ↓
Top-k experts
 ↓
weighted combine
```

Cover:

- routing logits
- expert capacity
- auxiliary load-balancing concept

Keep this optional.

---

# 16. Save and Load

Implement native checkpoint.

Then optionally make it Hugging Face compatible.

---

# Final Deliverable

A small but modern LLM architecture with:

```text
RMSNorm
RoPE
GQA
SwiGLU
Flash/SDPA
KV cache
modern generation
```

---

# C5 — Train an LLM From Scratch

## Goal

Turn C4 from a model into a real pretraining system.

---

# 1. Raw Corpus

```text
documents
 ↓
tokenization
 ↓
token stream
```

---

# 2. Tokenizer

Use an existing tokenizer first.

Optionally train a small tokenizer later.

---

# 3. Document Boundaries

Insert EOS.

Explain why blindly concatenating documents can alter training semantics.

---

# 4. Sequence Construction

Create:

\[
L
\]

token windows.

---

# 5. Packing

Compare:

### Padding

```text
doc1 + PAD
doc2 + PAD
```

### Packing

```text
doc1 EOS doc2 EOS doc3
```

Inspect utilization.

---

# 6. DataLoader

Implement:

- shuffling
- worker sharding
- deterministic resume
- prefetch
- pinned memory

---

# 7. AdamW

Configure:

- LR
- β1/β2
- epsilon
- weight decay

---

# 8. Parameter Groups

Optional:

- decay
- no-decay

---

# 9. LR Schedule

Implement:

- warmup
- cosine decay
- minimum LR

Plot it.

---

# 10. Gradient Accumulation

\[
B_{\rm global}
=
B_{\rm micro}
N_{\rm GPU}
N_{\rm accum}.
\]

Verify equivalence carefully.

---

# 11. BF16 Training

Compare with FP32.

---

# 12. Gradient Clipping

Measure:

\[
\|\nabla_\theta L\|.
\]

---

# 13. Gradient Checkpointing

Measure:

\[
\text{memory}
\leftrightarrow
\text{compute}.
\]

---

# 14. Checkpointing

Save:

```text
model
optimizer
scheduler
step
RNG
data position
config
```

---

# 15. Resume Correctly

Verify resumed run produces consistent loss trajectory.

---

# 16. Logging

Track:

- train loss
- validation loss
- LR
- grad norm
- parameter norm
- tokens/sec
- memory
- step time

---

# 17. MFU

Estimate model FLOPs utilization.

Cross-link ML Systems series.

---

# 18. Multi-GPU DDP

Implement:

- rank
- world size
- distributed sampler
- all-reduce

---

# 19. FSDP

Move to sharded parameters.

Measure memory change.

---

# 20. DeepSpeed / ZeRO

Optional implementation comparison.

---

# 21. Failure Modes

Handle:

- OOM
- NaN
- exploding gradients
- dead process
- bad batch
- corrupt checkpoint

---

# Final Deliverable

A genuine multi-GPU LLM pretraining system.

---

# C6 — Post-Train an LLM

## SFT → DPO → GRPO

One base model.

One codebase.

Three increasingly sophisticated post-training stages.

---

# Stage 1 — SFT

## 1. Conversational Dataset

```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

---

## 2. Chat Template

Serialize messages.

---

## 3. Assistant-Only Labels

Build mask:

```text
system → -100
user   → -100
assistant → target tokens
```

Inspect manually.

---

## 4. Packing

Pack multiple conversations where safe.

---

## 5. Full FT vs LoRA

Train both on a small model.

Compare:

- GPU memory
- trainable parameters
- convergence
- checkpoint size

---

# Stage 2 — DPO

## 6. Preference Dataset

```text
prompt
chosen
rejected
```

---

## 7. Sequence Log-Probabilities

Implement:

\[
\log\pi(y|x)
=
\sum_t\log\pi(y_t|x,y_{<t}).
\]

---

## 8. Reference Model

Freeze:

\[
\pi_{\rm ref}.
\]

---

## 9. DPO Loss

Implement manually before using TRL.

---

## 10. Debugging

Test:

- masking
- padding
- sequence lengths
- averaging vs summing
- reference gradients disabled

---

# Stage 3 — Reward Model

Optional but educational.

Implement pairwise reward loss.

---

# Stage 4 — REINFORCE

Before GRPO:

```text
prompt
 ↓
rollout
 ↓
reward
 ↓
policy gradient
```

Show high variance.

---

# Stage 5 — GRPO

Generate:

\[
o_1,\ldots,o_G.
\]

Compute:

\[
A_i=
\frac{r_i-\mu_r}{\sigma_r}.
\]

Implement:

- old logprobs
- clipping
- KL
- token masks
- group rewards

---

# Stage 6 — Verifiable Reward

Use tasks such as:

- arithmetic
- symbolic problem
- code tests

So:

\[
R
\]

is objective.

---

# Stage 7 — Rollout Engine

Build:

```text
prompts
 ↓
generation workers
 ↓
rollouts
 ↓
reward
 ↓
training batch
 ↓
optimizer
```

This prepares readers for agent RL.

---

# C7 — Build a VLM From Scratch

## Goal

Combine a pretrained vision encoder and language model into a functioning VLM.

---

# 1. Image Processor

Implement:

```text
image
 ↓
resize
 ↓
normalize
 ↓
tensor
```

---

# 2. Vision Encoder

Use a pretrained ViT initially.

Inspect:

\[
B\times N_v\times d_v.
\]

---

# 3. Projector

Map:

\[
d_v
\rightarrow
d_{\rm LLM}.
\]

Start:

```python
Linear
```

then:

```text
Linear → GELU → Linear
```

---

# 4. Image Tokens

Insert projected visual features into the language sequence.

---

# 5. Multimodal Sequence Construction

Build:

```text
[BOS]
user
image tokens
question
assistant
```

---

# 6. Labels

Set:

```text
image tokens → -100
prompt tokens → -100
assistant → supervised
```

where appropriate.

---

# 7. Position IDs

Handle image and text positions.

Discuss simple 1D positions first.

---

# 8. Attention Mask

Build and visualize.

---

# 9. Variable Resolution

Implement:

- fixed resize
- aspect-ratio padding
- tiling
- dynamic image-token lengths

---

# 10. Multi-Image Input

Support:

```text
image 1
image 2
question
```

Track boundaries.

---

# 11. Generation

Reuse LLM decoding stack.

---

# 12. Hugging Face Compatibility

Build:

- custom config
- processor
- VLM model class
- `save_pretrained`
- `from_pretrained`

---

# C8 — Train a Real VLM

## Goal

Implement an actual staged multimodal training recipe.

---

# Stage 1 — Alignment

Train projector with frozen:

```text
vision
LLM
```

using image-caption data.

---

# Stage 2 — Multimodal Pretraining

Experiment with:

- frozen vision
- trainable LLM
- partial vision unfreezing
- joint training

---

# 3. Differential Learning Rates

Example:

```text
vision    1e-6
LLM       2e-5
projector 1e-4
```

---

# 4. Unified Multimodal Dataset Interface

Use:

```python
{
    "images": [...],
    "messages": [...]
}
```

Support:

- captions
- VQA
- OCR
- documents
- charts
- grounding

---

# 5. Task Mixtures

Implement:

### proportional

### uniform

### temperature-based

### manually weighted

Track task performance.

---

# 6. Text Replay

Mix:

```text
multimodal batches
+
text-only batches
```

Measure language forgetting.

---

# 7. Dynamic Resolution

Handle variable token counts efficiently.

---

# 8. Multimodal Packing

Investigate:

- multiple images/example
- sequence-length budgets
- padding
- packing

---

# 9. LoRA for VLMs

Try LoRA on:

- LLM
- vision encoder
- projector
- combinations

---

# 10. Grounding Data

Support:

- points
- boxes
- masks
- special grounding tokens

---

# 11. Multimodal Evaluation

Implement:

- VQA exact match
- ANLS
- OCR CER/WER
- chart numeric accuracy
- grounding IoU

---

# C9 — Post-Train a VLM

## Preferences, Hallucination & RL

---

# 1. Multimodal Preference Dataset

```text
image
prompt
chosen
rejected
```

---

# 2. Multimodal DPO

Implement standard image-conditioned DPO first.

---

# 3. Hallucination Negatives

Programmatically create:

- wrong object
- wrong count
- wrong color
- false OCR
- wrong spatial relation

---

# 4. Image-Conditioned Preference Checks

Test whether preference can be predicted without the image.

This exposes language-prior shortcuts.

---

# 5. Visual Reward Functions

Implement:

### VQA

\[
R=\mathbf1[y=y^*]
\]

### Box

\[
R=\operatorname{IoU}(B,B^*)
\]

### Mask

Dice/IoU.

### OCR

ANLS/CER.

### Structured extraction

field correctness.

---

# 6. Visual GRPO

```text
image + prompt
      ↓
G rollouts
      ↓
visual/verifiable reward
      ↓
relative advantages
      ↓
policy update
```

---

# 7. Freeze Ablation

Compare:

```text
LLM only
projector + LLM
vision + projector + LLM
```

Ask:

> Is RL improving reasoning or perception?

---

# 8. Reward Hacking

Create examples deliberately.

Inspect how policy exploits weak rewards.

---

# C10 — Build an Agent Harness From Scratch

## Goal

Turn a model capable of tool calls into an actual coding agent.

---

# 1. Minimal Agent Loop

```python
while True:
    response = model(context)

    if response.final:
        break

    result = execute(response.tool_call)
    context.append(result)
```

Then progressively add every missing production component.

---

# 2. Tool Registry

Create:

```python
@tool
def read_file(...):
    ...
```

Generate tool schemas.

---

# 3. Structured Tool Calls

Parse and validate:

```json
{
  "name": "read_file",
  "arguments": {...}
}
```

---

# 4. Filesystem Tools

Implement:

- read
- list
- search
- write
- patch

---

# 5. Shell Tool

Implement:

- subprocess
- cwd
- stdout
- stderr
- exit code
- timeout

---

# 6. Code-Editing Interface

Compare:

- whole-file rewrite
- search/replace
- patch
- unified diff

Build a reliable patch application tool.

---

# 7. Verification Tools

Implement:

```text
run_tests
run_linter
run_typecheck
build
```

Normalize results.

---

# 8. Sandbox

Run code inside a Docker container.

Support:

- workspace mount
- CPU/memory limits
- network policy
- timeout
- process cleanup

---

# 9. Permission Layer

Classify:

```text
read
workspace-write
external-side-effect
dangerous
```

Ask user when required.

---

# 10. Context Builder

Combine:

\[
\text{instructions}
+
\text{repo state}
+
\text{tools}
+
\text{history}
+
\text{task state}.
\]

---

# 11. Repository Search

Implement:

- ripgrep
- file index
- symbol index
- optional embeddings

---

# 12. Repository Map

Create compact representation:

```text
file
 ├ class
 ├ function
 └ imports
```

---

# 13. Planning State

Maintain:

```text
[x] inspect
[x] reproduce
[ ] fix
[ ] test
```

---

# 14. Event Log

Every action becomes:

```text
UserMessage
ModelResponse
ToolCall
ToolResult
Patch
TestResult
Approval
```

Store JSONL.

---

# 15. Context Compaction

When history gets too large:

- preserve task state
- preserve important failures
- summarize older turns
- retain recent interactions

---

# 16. Output Truncation

Build safe observation policies for giant shell outputs.

---

# 17. Git Integration

Expose:

- status
- diff
- log
- revert
- checkpoint

---

# 18. Tracing

Measure:

- model calls
- tool calls
- tokens
- latency
- failures
- retries

---

# Final Deliverable

A functioning coding agent that can:

```text
inspect repository
 ↓
search files
 ↓
edit code
 ↓
run tests
 ↓
observe failures
 ↓
repair
 ↓
verify
```

---

# C11 — Add Memory & Context Engineering

## Goal

Turn the short-lived agent from C10 into a stateful system.

---

# 1. Working Memory

Maintain explicit task state:

```json
{
    "goal": "...",
    "current_step": "...",
    "open_questions": [...]
}
```

---

# 2. Episodic Memory

Store past:

\[
(s,a,o,r,t)
\]

events/trajectories.

---

# 3. Semantic Memory

Extract stable facts from past interactions.

---

# 4. Procedural Memory

Store reusable strategies/scripts.

---

# 5. Memory Writer

Implement:

```python
should_store(event)
```

based on:

- novelty
- stability
- importance
- confidence
- scope

---

# 6. Vector Retrieval

Use embeddings + vector index.

---

# 7. Hybrid Retrieval

Combine:

\[
S=
\alpha S_{\rm semantic}
+
\beta S_{\rm recency}
+
\gamma S_{\rm importance}.
\]

---

# 8. Update / Supersede

Don't append contradictory facts forever.

Implement:

```text
ADD
UPDATE
SUPERSEDE
DELETE
```

---

# 9. Memory Consolidation

Convert many episodes into one reusable rule.

---

# 10. Context Budget

Given token budget \(B\), choose:

- instructions
- task state
- relevant memories
- recent events
- tool observations

---

# 11. Context Compression

Implement:

- summarization
- extraction
- dropping
- priority ordering

---

# 12. Memory Evaluation

Compare:

```text
no memory
full transcript
retrieved memory
oracle memory
```

Measure whether memory causally improves task performance.

---

# C12 — Train an Agent With Reinforcement Learning

## Goal

Move from:

> imitate successful agent trajectories

to:

> learn action strategies through environment interaction.

---

# 1. Build a Toy Environment

Example:

> Locate the correct file, modify the required value, then submit.

State:

```text
filesystem
task
history
```

Actions:

```text
list
read
search
edit
submit
```

Reward:

\[
R=
\mathbf1[\text{final state correct}].
\]

---

# 2. Environment API

Implement:

```python
reset()
observe()
step(action)
reward()
done()
```

---

# 3. Agent Trajectory

\[
\tau=
(s_0,a_0,r_0,\ldots,s_T).
\]

Record everything.

---

# 4. Behavioral Cloning

Train from demonstrations:

\[
L
=
-\sum_t
\log\pi(a_t^*|s_t).
\]

---

# 5. Distribution Shift

Show how a small agent mistake leads to states absent from demonstrations.

---

# 6. Online Rollouts

Let policy interact with environment.

---

# 7. Outcome Reward

Start with final success only.

---

# 8. GRPO-Style Agent Update

Generate multiple trajectories for the same task:

\[
\tau_1,\ldots,\tau_G.
\]

Calculate group-relative advantages.

---

# 9. Long-Horizon Credit Assignment

Compare:

### trajectory-level reward

vs

### action-level advantage.

---

# 10. Reward Shaping

Try intermediate milestones.

Then demonstrate reward hacking.

---

# 11. Exploration

Use:

- sampling
- diverse initial states
- diverse tasks
- curriculum

---

# 12. Curriculum

Progress:

```text
single tool
 ↓
multiple tools
 ↓
dependent tools
 ↓
failure recovery
 ↓
long-horizon tasks
```

---

# 13. Failure Recovery

Include environments containing:

- timeout
- invalid input
- missing file
- tool error

Reward robust recovery.

---

# 14. Offline vs Online RL

Compare:

```text
stored trajectories
```

against:

```text
live environment interaction
```

---

# 15. Agent-RL Training System

Build:

```text
task sampler
    ↓
environment workers
    ↓
agent rollouts
    ↓
trajectory store
    ↓
reward/verifier
    ↓
training batches
    ↓
policy update
    ↓
new policy
```

---

# Common Requirements Across Every Tutorial

Every coding article should contain the following.

---

## 1. Theory Cross-Links

Do not repeat derivations unnecessarily.

Example:

```text
Attention implementation
→ Attention, From First Principles

RoPE implementation
→ How Models Know Where They Are

RMSNorm
→ Geometry of Normalization

SwiGLU
→ Activation Atlas

AdamW
→ How Models Take a Step

DDP/FSDP
→ Parallelism Playbook

KV cache
→ One Token at a Time

GRPO
→ When Correctness Is the Reward
```

---

# 2. Tensor Shape Tables

Every major section should show shapes.

Example:

| Tensor | Shape |
|---|---|
| Input IDs | \(B\times N\) |
| Hidden state | \(B\times N\times D\) |
| Q | \(B\times H\times N\times d_h\) |
| KV cache | \(B\times H_{KV}\times T\times d_h\) |
| Logits | \(B\times N\times V\) |

Readers should never need to guess dimensions.

---

# 3. Start Naive, Then Optimize

Pattern:

```text
simple implementation
      ↓
verify correctness
      ↓
identify bottleneck
      ↓
optimized implementation
```

Examples:

```text
naive attention
→ SDPA
→ FlashAttention
```

```text
recompute history
→ KV cache
```

```text
single GPU
→ DDP
→ FSDP
```

---

# 4. Tests

Every repository should have tests.

At minimum:

```text
shape tests
numerical-equivalence tests
gradient tests
serialization tests
resume tests
causality tests
```

For attention:

```text
full forward == KV cached forward
```

within tolerance.

---

# 5. Benchmarking

Each article should benchmark the thing it builds.

Measure where appropriate:

- latency
- throughput
- GPU memory
- token/sec
- samples/sec
- model parameters
- trainable parameters
- checkpoint size

---

# 6. Failure Sections

Every tutorial should intentionally break something.

Examples:

### Attention

- wrong mask
- bad dimension
- missing \(1/\sqrt d\)

### Training

- too-high LR
- no warmup
- BF16 overflow/NaN

### VLM

- misaligned image tokens
- wrong loss mask
- excessive image resolution

### Agents

- malformed tool call
- command timeout
- stale context
- bad memory

Readers learn more from debugging than from perfect runs.

---

# 7. Inspect Intermediate State

Always print or visualize:

- token IDs
- tensor shapes
- masks
- attention maps
- loss masks
- gradients
- KV cache
- generated rollouts
- tool traces

Avoid tutorials where everything happens inside one library call.

---

# 8. Compare Against a Trusted Implementation

Examples:

### Attention

Compare custom result against:

```python
torch.nn.functional.scaled_dot_product_attention
```

### LLM

Load equivalent small HF architecture.

### Trainer

Compare raw PyTorch and HF Trainer loss.

### LoRA

Compare custom concept against PEFT.

---

# 9. Provide Full Repository Structure

Every tutorial should end with something similar to:

```text
project/
├── configs/
├── data/
├── models/
├── training/
├── utils/
├── tests/
├── scripts/
├── benchmarks/
└── README.md
```

Avoid giant single-file tutorial code by the end.

---

# 10. Separate Educational and Production Implementations

For example:

```text
attention_naive.py
attention_sdpa.py
```

or:

```text
sampler_reference.py
sampler_fast.py
```

The reference implementation optimizes for readability.

The production implementation optimizes for speed.

---

# 11. Include Exercises

Every tutorial should end with 3–5 meaningful extensions.

Examples for Attention:

1. Add ALiBi.
2. Add sliding-window attention.
3. Convert MHA checkpoint to GQA.
4. Measure cache memory.
5. Implement a custom attention bias.

Not trivial syntax questions.

---

# 12. Final Series Outcome

A reader who completes the entire track should have personally built:

```text
raw PyTorch training loop
        ↓
attention library
        ↓
modern LLM
        ↓
distributed LLM trainer
        ↓
SFT / DPO / GRPO pipeline
        ↓
vision-language model
        ↓
multimodal training system
        ↓
multimodal RL system
        ↓
coding-agent harness
        ↓
agent memory system
        ↓
agent RL environment
```

The key identity of this series should be:

> **You should be able to go from a tensor operation all the way to a foundation model and agent system without any layer feeling magical.**
