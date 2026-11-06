"""Shared plumbing for the scripts: the text-pretrained base model, one SFT run, and result files."""

from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path

import torch

from .train import RunConfig, new_model, train

LAB = Path(__file__).resolve().parent.parent
RUNS = LAB / "runs"          # checkpoints, git-ignored
RESULTS = LAB / "results"
BASE_STEPS = 1500


def hardware() -> str:
    if torch.cuda.is_available():
        return f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}"
    cpu = platform.machine()
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip().replace(" Processor", "")
                break
    except OSError:
        pass
    return f"{cpu}, {torch.get_num_threads()} threads, torch {torch.__version__}"


def base_model(steps: int = BASE_STEPS, seed: int = 0):
    """The 'language model': the decoder trained on text-only chat data (the reverse task) first.
    Every SFT run starts from this checkpoint, as a real VLM starts from a pretrained LLM."""
    path = RUNS / f"base_text_{steps}_s{seed}.pt"
    model = new_model(seed)
    if path.exists():
        model.load_state_dict(torch.load(path))
        return model.eval()
    RUNS.mkdir(exist_ok=True)
    info = train(model, RunConfig(mixture={"text": 1.0}, steps=steps, seed=seed + 1000))
    torch.save(model.state_dict(), path)
    print(f"trained base text model in {info['seconds']} s -> {path.name}")
    return model.eval()


def sft(cfg: RunConfig):
    model = base_model()
    info = train(model, cfg)
    return model, info


def write_result(name: str, payload: dict, text: str = "", sweeping: bool = True) -> None:
    """results/<name>.json and .txt, each starting with the hardware and the command."""
    RESULTS.mkdir(exist_ok=True)
    cmd = "python " + " ".join([Path(sys.argv[0]).relative_to(LAB.parent.parent).as_posix()
                                if Path(sys.argv[0]).is_absolute() else sys.argv[0]] + sys.argv[1:])
    hw = hardware()
    if sweeping:  # training ran in worker processes, not in this one
        hw = hw.replace(f"{torch.get_num_threads()} threads", f"{os.environ.get('LAB_WORKERS', '4')} worker processes "
                        f"x {os.environ.get('LAB_THREADS_PER_JOB', '2')} threads")
    head = {"hardware": hw, "command": cmd}
    (RESULTS / f"{name}.json").write_text(json.dumps({**head, **payload}, indent=1) + "\n")
    if text:
        (RESULTS / f"{name}.txt").write_text(f"# hardware: {head['hardware']}\n# command: {cmd}\n\n{text}\n")


def _job(job: tuple) -> tuple:
    name, cfg, suite = job
    from . import evaluate
    torch.set_num_threads(int(os.environ.get("LAB_THREADS_PER_JOB", "2")))
    model, info = sft(cfg)
    metrics = {}
    for fn, kwargs in suite:
        kwargs = dict(kwargs)
        tag = kwargs.pop("tag", None)
        res = getattr(evaluate, fn)(model, **kwargs)
        metrics.update({(f"{k}_{tag}" if tag else k): v for k, v in res.items()})
    return name, metrics, info["seconds"]


def sweep(jobs: list[tuple]) -> dict[str, dict]:
    """Run [(name, RunConfig, [(eval_fn, kwargs), ...])] in parallel worker processes.
    Set LAB_WORKERS (default 4) and LAB_THREADS_PER_JOB (default 2) to fit the machine."""
    import multiprocessing as mp
    base_model()  # train the shared base once, before the workers need it
    workers = int(os.environ.get("LAB_WORKERS", "4"))
    out = {}
    with mp.get_context("spawn").Pool(workers) as pool:
        for name, metrics, secs in pool.imap_unordered(_job, jobs):
            print(f"{name}: {metrics} ({secs} s)", flush=True)
            out[name] = {**metrics, "train_seconds": secs}
    return {name: out[name] for name, _, _ in jobs}
