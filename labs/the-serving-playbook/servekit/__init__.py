"""servekit: the serving mechanisms from "The Serving Playbook", built small.

block_manager  paged KV blocks: free list, block tables, ref counts, copy-on-write, chained-hash prefix cache
model          a Llama forward pass over a ragged batch that reads and writes a paged KV cache
scheduler      iteration-level scheduling: token budget, slots, watermark, chunked prefill, preemption
engine         the loop that ties them together, plus a CUDA-graph decode path
alloc_sim      reserve-max versus first-fit versus paged KV memory, on a request trace
spec           the speculative sampling rule
metrics        TTFT, TPOT, goodput, workloads, results-file headers
"""

from .block_manager import BlockManager, block_hashes
from .scheduler import Scheduler, SchedulerConfig, Sequence

__all__ = ["BlockManager", "Scheduler", "SchedulerConfig", "Sequence", "block_hashes"]
