"""Batched decoding for a frozen causal LM: plain sampling, and s1-style budget forcing.

Both loops call the model one token at a time with a KV cache, so the per-row rules of budget
forcing (cap the trace, or suppress the stop and append "Wait") are plain Python.
The model only needs the Hugging Face call signature: model(input_ids, attention_mask,
position_ids, past_key_values, use_cache) -> object with .logits and .past_key_values.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


def load(name: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device).eval()
    return model, tok


def chat_ids(tok, prompt: str) -> list[int]:
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return tok.encode(text, add_special_tokens=False)


def token_sets(tok) -> dict:
    """Token ids the forcing loop needs: stop attempts, end-of-answer, and the forced strings."""
    vocab = tok.get_vocab()
    text = {i: tok.convert_tokens_to_string([t]) for t, i in vocab.items()}
    eos = {tok.eos_token_id} | ({tok.pad_token_id} if tok.pad_token_id is not None else set())
    return {
        "stop": {i for i, s in text.items() if s.strip().startswith("Answer")} | eos,
        "end_answer": {i for i, s in text.items() if "\n" in s} | eos,
        "eos": eos,
        "answer": tok.encode("\nAnswer:", add_special_tokens=False),
        "pad": tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
    }


def _left_pad(seqs: list[list[int]], pad: int, device):
    n = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), n), pad, dtype=torch.long)
    mask = torch.zeros((len(seqs), n), dtype=torch.long)
    for r, s in enumerate(seqs):
        ids[r, n - len(s):] = torch.tensor(s)
        mask[r, n - len(s):] = 1
    return ids.to(device), mask.to(device)


class _Stepper:
    """Prefill once, then feed one token per row per step."""

    def __init__(self, model, seqs, pad, device):
        self.model = model
        ids, self.mask = _left_pad(seqs, pad, device)
        pos = (self.mask.cumsum(1) - 1).clamp(min=0)
        out = model(input_ids=ids, attention_mask=self.mask, position_ids=pos, use_cache=True)
        self.logits, self.past = out.logits[:, -1].float(), out.past_key_values

    def step(self, next_ids: torch.Tensor):
        self.mask = torch.cat([self.mask, torch.ones_like(self.mask[:, :1])], dim=1)
        pos = self.mask.sum(1, keepdim=True) - 1
        out = self.model(input_ids=next_ids[:, None], attention_mask=self.mask, position_ids=pos,
                         past_key_values=self.past, use_cache=True)
        self.logits, self.past = out.logits[:, -1].float(), out.past_key_values


@torch.no_grad()
def sample(model, seqs, *, eos: set, pad: int, max_new: int, temperature: float, generator=None):
    """Temperature sampling until EOS. Returns (token lists, mean log-prob per row)."""
    dev = next(model.parameters()).device
    st = _Stepper(model, seqs, pad, dev)
    B = len(seqs)
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    lp_sum = torch.zeros(B, device=dev)
    n_tok = torch.zeros(B, device=dev)
    toks = []
    eos_t = torch.tensor(sorted(eos), device=dev)
    for _ in range(max_new):
        logp = torch.log_softmax(st.logits / temperature, dim=-1)
        nxt = torch.multinomial(logp.exp(), 1, generator=generator).squeeze(1)
        nxt = torch.where(done, torch.full_like(nxt, pad), nxt)
        live = ~done
        lp_sum += torch.where(live, logp.gather(1, nxt[:, None]).squeeze(1), torch.zeros_like(lp_sum))
        n_tok += live.float()
        toks.append(nxt)
        done |= torch.isin(nxt, eos_t)
        if bool(done.all()):
            break
        st.step(nxt)
    T = torch.stack(toks, 1).tolist() if toks else [[] for _ in range(B)]
    out = []
    for row in T:
        cut = next((j for j, t in enumerate(row) if t in eos), len(row))
        out.append(row[:cut])
    return out, (lp_sum / n_tok.clamp(min=1)).tolist()


@dataclass
class Forcing:
    """Budget-forcing settings. cap: force the answer after this many thinking tokens.
    waits: how many times to suppress a stop attempt and append `wait_ids` instead."""

    cap: int | None = None
    waits: int = 0
    wait_ids: list[int] = field(default_factory=list)
    max_think: int = 768
    max_answer: int = 12


@torch.no_grad()
def budget_force(model, seqs, f: Forcing, ts: dict) -> list[dict]:
    """Greedy decoding with s1 budget forcing, one state machine per row.

    Thinking phase: every greedy token is kept unless it is a stop attempt (a token that starts
    'Answer', or EOS). A stop attempt is replaced by `wait_ids` while waits remain; otherwise, or
    once the cap is reached, the loop appends '\\nAnswer:' and switches to the answer phase.
    """
    dev = next(model.parameters()).device
    st = _Stepper(model, seqs, ts["pad"], dev)
    rows = [{"phase": 0, "queue": [], "think": [], "answer": [], "waits_used": 0, "capped": False,
             "stops_seen": 0} for _ in seqs]
    while not all(r["phase"] == 2 for r in rows):
        greedy = st.logits.argmax(-1).tolist()
        nxt = []
        for r, y in zip(rows, greedy):
            if r["phase"] == 2:
                nxt.append(ts["pad"])
                continue
            if not r["queue"] and r["phase"] == 0:
                limit = f.cap if f.cap is not None else f.max_think
                if len(r["think"]) >= limit:
                    r["capped"] = True
                    r["queue"], r["phase"] = list(ts["answer"]), 1
                elif y in ts["stop"]:
                    r["stops_seen"] += 1
                    if r["waits_used"] < f.waits:
                        r["waits_used"] += 1
                        r["queue"] = list(f.wait_ids)
                    else:
                        r["queue"], r["phase"] = list(ts["answer"]), 1
            if r["queue"]:
                t = r["queue"].pop(0)
                if r["phase"] == 0:
                    r["think"].append(t)
            else:
                t = y
                if r["phase"] == 0:
                    r["think"].append(t)
                else:
                    r["answer"].append(t)
                    if t in ts["end_answer"] or len(r["answer"]) >= f.max_answer:
                        r["phase"] = 2
            nxt.append(t)
        st.step(torch.tensor(nxt, device=dev))
    return rows
