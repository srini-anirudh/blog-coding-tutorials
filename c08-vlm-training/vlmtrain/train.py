"""The staged trainer. One loop, configured per stage:

    Stage 1  freeze=align, tasks=[caption]               train the projector only
    Stage 2  freeze=lm|partial|full, a task mixture       train the projector and the LM (and maybe vision)
    LoRA     freeze=none, lora=[...]                      low-rank adapters on chosen modules

Every step: draw tasks from the mixture -> build unified samples -> processor (or packer) ->
forward -> per-sample loss -> backward -> clip -> step. Per-task losses are logged, so a task that
stops improving is visible long before the final evaluation.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from .c07_adapter import build_c07, build_c07_tiny, c07_available
from .data import TASK_SIZES, TASKS, make_example
from .device import laptop_threads, pick_device, seed_everything, sync
from .evaluate import evaluate, format_results
from .freeze import format_report, set_trainable, train_new_token_rows, trainable_report
from .lora import apply_lora
from .mixture import MixtureSampler, epochs_per_task, mixture_weights, with_replay
from .model import VLMConfig, build_vlm, load_partial, save_trainable, tiny_config
from .optim import build_optimizer, describe, warmup_cosine
from .packing import pack
from .processor import IGNORE, TinyVLMProcessor
from .replay import ReplayCorpus, load_passages, perplexity
from .tokenizer import IMAGE_TOKEN, build_tokenizer

HERE = Path(__file__).resolve().parent.parent


@dataclass
class TrainConfig:
    name: str = "run"
    model: str = "auto"                    # auto | tinyvlm | own | tiny | tiny_c07 (see build_model)
    steps: int = 600
    batch_size: int = 8
    freeze: str = "lm"
    vision_layers_to_train: int = 0
    lrs: dict = field(default_factory=lambda: {"vision": 1e-6, "projector": 1e-4, "lm": 2e-5, "lora": 2e-4})
    weight_decay: float = 0.0
    warmup: int = 30
    grad_clip: float = 1.0
    loss_norm: str = "token"               # token: mean over supervised tokens | sample: mean of sample means
    tasks: list = field(default_factory=lambda: list(TASKS))
    task_sizes: dict = field(default_factory=lambda: dict(TASK_SIZES))
    mixture: str = "temperature"
    temperature: float = 2.0
    manual_weights: dict | None = None
    replay_ratio: float = 0.0
    replay_max_tokens: int = 96
    strategy: str = "tile"
    max_tiles: int = 4
    grounding_format: str = "tokens"
    lora: list = field(default_factory=list)
    lora_r: int = 16
    lora_alpha: float = 32.0
    train_new_tokens: bool = False
    pack_budget: int = 0                   # 0 = pad each batch; >0 = pack batch_size samples into rows of this many tokens
    init_from: str | None = None
    eval_n: int = 64
    eval_every: int = 0
    eval_n_mid: int = 24
    ppl_passages: int = 64
    save: bool = False
    out_dir: str = "outputs"
    results_dir: str | None = "results"
    seed: int = 0
    device: str | None = None
    log_every: int = 10


def build_model(cfg: TrainConfig):
    """Tokenizer, model and processor for the configured backend.

    tinyvlm  C7's package (the default whenever ../c07-vlm exists)
    own      vlmtrain.model, the same contract in one file, so this folder also runs alone
    tiny     own model with random weights and no downloads (tests)
    tiny_c07 C7's model with random weights and no downloads (tests)
    """
    name = cfg.model
    if name == "auto":
        name = "tinyvlm" if c07_available() else "own"
    if name == "tinyvlm":
        return build_c07(cfg.strategy, cfg.max_tiles)
    if name == "tiny_c07":
        return build_c07_tiny(cfg.strategy, cfg.max_tiles, seed=cfg.seed)
    if name == "tiny":
        tok, vcfg = build_tokenizer("offline"), tiny_config()
    else:
        vcfg = VLMConfig()
        tok = build_tokenizer(vcfg.lm_name)
    model = build_vlm(vcfg, tok)
    proc = TinyVLMProcessor(tok, vcfg.tokens_per_tile, vcfg.image_size, cfg.strategy, cfg.max_tiles)
    return tok, model, proc


def per_sample_loss(logits: torch.Tensor, labels: torch.Tensor, sample_index: torch.Tensor, n_samples: int):
    """Next-token cross-entropy summed per sample.

    logits (B, N, V), labels (B, N), sample_index (B, N) -> (loss_sum (S,), token_count (S,))
    Position t predicts token t+1, so logits drop the last column and labels drop the first.
    """
    tgt = labels[:, 1:]
    tok_loss = F.cross_entropy(logits[:, :-1].float().transpose(1, 2), tgt, ignore_index=IGNORE, reduction="none")
    idx = sample_index[:, 1:].reshape(-1)
    valid = (tgt.reshape(-1) != IGNORE) & (idx >= 0)
    sums = torch.zeros(n_samples, device=logits.device).index_add_(0, idx[valid], tok_loss.reshape(-1)[valid])
    ones = torch.ones(int(valid.sum()), device=logits.device)
    counts = torch.zeros(n_samples, device=logits.device).index_add_(0, idx[valid], ones)
    return sums, counts


def make_batch(cfg: TrainConfig, proc: TinyVLMProcessor, samples: list[dict]) -> dict:
    enc = [proc.encode(s) for s in samples]
    if cfg.pack_budget:
        b = pack(enc, cfg.pack_budget, proc.pad_id)
        # sample k of the packed batch is (row, segment); number them in packing order
        offsets = torch.tensor([0] + [len(r) for r in b["rows"]]).cumsum(0)[:-1]
        b["sample_index"] = torch.where(b["seg_ids"] >= 0, b["seg_ids"] + offsets[:, None], -1)
        return b
    b = proc.collate(enc)
    b["sample_index"] = torch.where(b["attention_mask"] > 0,
                                    torch.arange(len(enc))[:, None].expand_as(b["input_ids"]), -1)
    return b


def run(cfg: TrainConfig) -> dict:
    seed_everything(cfg.seed)
    laptop_threads(2)
    device = pick_device(cfg.device)
    tok, model, proc = build_model(cfg)
    loaded = load_partial(model, HERE / cfg.init_from) if cfg.init_from else []
    set_trainable(model, cfg.freeze, cfg.vision_layers_to_train)
    lora_names = apply_lora(model, cfg.lora, cfg.lora_r, cfg.lora_alpha) if cfg.lora else []
    if cfg.train_new_tokens:
        train_new_token_rows(model, tok.convert_tokens_to_ids(IMAGE_TOKEN))
    model.to(device).train()
    report = trainable_report(model)
    print(f"[{cfg.name}] device={device} loaded={len(loaded)} tensors, lora on {len(lora_names)} linears")
    print(format_report(report))

    replay = None
    if cfg.replay_ratio > 0:
        replay = ReplayCorpus(_truncate(load_passages("train", limit=4000), tok, cfg.replay_max_tokens))
    sizes = {t: cfg.task_sizes[t] for t in cfg.tasks}
    weights = with_replay(mixture_weights(sizes, cfg.mixture, cfg.temperature, cfg.manual_weights), cfg.replay_ratio)
    all_sizes = {**sizes, **({"text": len(replay)} if replay else {})}
    sampler = MixtureSampler(weights, all_sizes, seed=cfg.seed)
    print("mixture:", {k: round(v, 3) for k, v in weights.items()})
    print("epochs over each corpus:",
          {k: round(v, 2) for k, v in epochs_per_task(weights, all_sizes, cfg.steps * cfg.batch_size).items()})

    opt = build_optimizer(model, cfg.lrs, cfg.weight_decay)
    print(describe(opt))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, warmup_cosine(cfg.warmup, cfg.steps))

    test_passages = load_passages("test", limit=cfg.ppl_passages) if cfg.ppl_passages else []
    ppl_before = perplexity(model.lm, tok, test_passages, device) if test_passages else None
    if ppl_before:
        print(f"text perplexity before: {ppl_before:.2f}")

    log, mid_evals = [], []
    win_loss: dict[str, float] = defaultdict(float)
    win_n: dict[str, int] = defaultdict(int)
    tok_by_task: dict[str, int] = defaultdict(int)
    t0, tokens_seen = time.time(), 0
    for step in range(cfg.steps):
        samples = []
        for _ in range(cfg.batch_size):
            task, i = sampler.draw()
            samples.append(replay[i] if task == "text" else make_example(task, i, "train", cfg.grounding_format))
        b = make_batch(cfg, proc, samples)
        dev = {k: b[k].to(device) for k in ("input_ids", "labels", "attention_mask", "sample_index")}
        out = model(dev["input_ids"], None if b["pixel_values"] is None else b["pixel_values"].to(device),
                    attention_mask=dev["attention_mask"],
                    position_ids=b["position_ids"].to(device) if "position_ids" in b else None)
        sums, counts = per_sample_loss(out.logits, dev["labels"], dev["sample_index"], len(b["tasks"]))
        if cfg.loss_norm == "token":
            loss = sums.sum() / counts.sum()
        else:
            loss = (sums / counts.clamp(min=1)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], cfg.grad_clip)
        opt.step()
        sched.step()

        per = (sums / counts.clamp(min=1)).detach().cpu().tolist()
        for task, l, c in zip(b["tasks"], per, counts.tolist()):
            win_loss[task] += l
            win_n[task] += 1
            tok_by_task[task] += int(c)
        tokens_seen += int(dev["attention_mask"].shape[-1] * dev["input_ids"].shape[0])
        if (step + 1) % cfg.log_every == 0 or step == 0:
            sync(device)
            rec = {"step": step + 1, "loss": round(loss.item(), 4), "grad_norm": round(gnorm.item(), 3),
                   "lr": {g["name"]: g["lr"] for g in opt.param_groups},
                   "task_loss": {k: round(win_loss[k] / win_n[k], 4) for k in win_loss},
                   "sec": round(time.time() - t0, 1)}
            log.append(rec)
            win_loss.clear()
            win_n.clear()
            print(f"step {rec['step']:>5}  loss {rec['loss']:.3f}  gnorm {rec['grad_norm']:.2f}  "
                  + " ".join(f"{k}={v:.2f}" for k, v in rec["task_loss"].items()) + f"  {rec['sec']}s")
        if cfg.eval_every and (step + 1) % cfg.eval_every == 0 and step + 1 < cfg.steps:
            res = evaluate(model, proc, device, cfg.tasks, cfg.eval_n_mid, grounding_format=cfg.grounding_format)
            mid_evals.append({"step": step + 1, **res["metrics"]})
    sync(device)
    train_sec = time.time() - t0

    res = evaluate(model, proc, device, cfg.tasks, cfg.eval_n, grounding_format=cfg.grounding_format)
    mid_evals.append({"step": cfg.steps, **res["metrics"]})
    print(format_results(res))
    ppl_after = perplexity(model.lm, tok, test_passages, device) if test_passages else None
    if ppl_after:
        print(f"text perplexity after: {ppl_after:.2f} (before {ppl_before:.2f})")
    total_sup = sum(tok_by_task.values())
    summary = {
        "name": cfg.name, "config": asdict(cfg), "device": str(device), "train_sec": round(train_sec, 1),
        "sec_per_step": round(train_sec / cfg.steps, 3), "tokens_per_sec": round(tokens_seen / train_sec),
        "trainable": report, "mixture_target": weights, "mixture_observed": sampler.observed(),
        "supervised_token_share": {k: v / total_sup for k, v in tok_by_task.items()},
        "ppl_before": ppl_before, "ppl_after": ppl_after,
        "metrics": res["metrics"], "examples": res["examples"], "curve": mid_evals, "log": log,
    }
    if cfg.save:
        n = save_trainable(model, HERE / cfg.out_dir / cfg.name / "trainable.safetensors")
        print(f"saved {n:,} trainable parameters to {cfg.out_dir}/{cfg.name}/")
    if cfg.results_dir:
        p = HERE / cfg.results_dir / f"{cfg.name}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(summary, indent=1, default=str))
    return summary


def _truncate(passages: list[str], tok, max_tokens: int) -> list[str]:
    """Cut replay passages to max_tokens so one passage cannot dominate a token-averaged loss."""
    out = []
    for t in passages:
        ids = tok(t, add_special_tokens=False)["input_ids"][:max_tokens]
        out.append(tok.decode(ids))
    return out
