"""Start N ranks on one machine and collect what each one returns.

Every rank runs the same function (SPMD). On CPU the backend is gloo; with CUDA it is NCCL and rank r
owns cuda:r. The launcher picks a free port so tests can run back to back.
"""

from __future__ import annotations

import os
import socket
import time
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def device() -> torch.device:
    """The device this rank owns: cuda:<rank> under NCCL, else CPU."""
    if dist.is_initialized() and dist.get_backend() == "nccl":
        return torch.device("cuda", dist.get_rank() % torch.cuda.device_count())
    return torch.device("cpu")


def _worker(rank, world, port, backend, fn, args, queue):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.set_num_threads(int(os.environ.get("PPBOOK_THREADS", "1")))
    try:
        if backend == "nccl":
            torch.cuda.set_device(rank)
        dist.init_process_group(backend, rank=rank, world_size=world)
        out = fn(*args)
        dist.barrier()
        queue.put((rank, "ok", out))
    except Exception:  # report the traceback to the parent instead of hanging it
        queue.put((rank, "error", traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def spawn(fn, world: int, *args, backend: str | None = None, timeout: float = 600) -> list:
    """Run fn(*args) on `world` ranks. Returns the list of per-rank return values, indexed by rank."""
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() and torch.cuda.device_count() >= world else "gloo"
    ctx = mp.get_context("spawn")
    queue = ctx.SimpleQueue()
    port = free_port()
    procs = [ctx.Process(target=_worker, args=(r, world, port, backend, fn, args, queue)) for r in range(world)]
    for p in procs:
        p.start()
    results, deadline = {}, time.time() + timeout
    while len(results) < world:
        if not queue.empty():
            rank, status, out = queue.get()
            if status == "error":
                for p in procs:
                    p.kill()
                raise RuntimeError(f"rank {rank} failed:\n{out}")
            results[rank] = out
        elif time.time() > deadline or any(p.exitcode not in (None, 0) for p in procs):
            for p in procs:
                p.kill()
            raise RuntimeError("a rank died or the job timed out (a hang looks exactly like this)")
        else:
            time.sleep(0.01)
    for p in procs:
        p.join()
    return [results[r] for r in range(world)]


def sync() -> None:
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()


def hardware_line() -> str:
    """One line naming the hardware, for the top of every results file. No hostnames."""
    import platform
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return (f"hardware: {torch.cuda.device_count()} x {name}, torch {torch.__version__}, "
                f"CUDA {torch.version.cuda}")
    cpu = "unknown CPU"
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        cpu = platform.processor() or cpu
    threads = os.environ.get("OMP_NUM_THREADS", str(torch.get_num_threads()))
    return f"hardware: {cpu}, {threads} threads, torch {torch.__version__}, CPU only (gloo)"


class Report:
    """Print lines as they come, then write them to results/<name> under a hardware line and the command."""

    def __init__(self, name: str):
        import sys
        from pathlib import Path
        self.path = Path(__file__).resolve().parent.parent / "results" / name
        script = Path(sys.argv[0]).name
        self.lines = [hardware_line(), f"command: python labs/the-parallelism-playbook/scripts/{script} "
                      + " ".join(sys.argv[1:]), ""]
        print("\n".join(self.lines))

    def __call__(self, *parts) -> None:
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        self.lines.append(line)

    def save(self) -> None:
        self.path.parent.mkdir(exist_ok=True)
        self.path.write_text("\n".join(self.lines).rstrip() + "\n")
        print(f"wrote {self.path.name}")
