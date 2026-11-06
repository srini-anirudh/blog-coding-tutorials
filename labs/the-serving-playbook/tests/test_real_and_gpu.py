"""A real checkpoint through the paged engine, and the CUDA paths (graphs, bf16)."""

import pytest
import torch

from servekit.engine import Engine
from servekit.metrics import random_prompts
from servekit.model import load_model
from servekit.scheduler import SchedulerConfig


def serve(model, prompts, n, **kw):
    eng = Engine(model, num_blocks=512, sched=SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=48), **kw)
    seqs = [eng.add_request(p, n) for p in prompts]
    while eng.sched.has_work():
        eng.step()
    return [s.output for s in seqs]


@pytest.mark.network
def test_smollm2_matches_hf_generate():
    from transformers import AutoModelForCausalLM

    name = "HuggingFaceTB/SmolLM2-135M"
    model = load_model(name)
    hf = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
    prompts = random_prompts(3, 20, 70, model.config.vocab_size, seed=5)
    got = serve(model, prompts, 8)  # 48-token budget: the 70-token prompt is chunked
    for p, out in zip(prompts, got):
        ref = hf.generate(torch.tensor([p]), max_new_tokens=8, do_sample=False, eos_token_id=None)
        assert out == ref[0, len(p):].tolist()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_graph_decode_matches_eager():
    model = load_model("tiny", device="cuda")
    prompts = random_prompts(5, 4, 30, 256, seed=2)
    eager = serve(model, prompts, 12)
    graph = serve(model, prompts, 12, cuda_graph=True, max_model_len=256)
    assert graph == eager
    cpu = serve(load_model("tiny"), prompts, 12)
    assert eager == cpu
