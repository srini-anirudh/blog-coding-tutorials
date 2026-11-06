"""Corpus, tokenizer, shards and packing."""

import numpy as np
import pytest
import torch

from pretrain.corpus import SEPARATOR, download, split_documents
from pretrain.packing import (block_causal_mask, document_ids, greedy_packing_utilization, pack_stream,
                              pad_batch, padding_utilization, position_ids, best_fit_packing_utilization)
from pretrain.shards import TokenShards
from pretrain.tokenizer import load_file, load_pretrained


@pytest.mark.network
def test_existing_tokenizer_from_the_hub():
    tok = load_pretrained("openai-community/gpt2")
    assert tok.vocab_size == 50257 and tok.eos_id == 50256
    assert tok.decode(tok.encode("Once upon a time")) == "Once upon a time"


@pytest.mark.network
def test_range_download_keeps_only_whole_documents(tmp_path):
    path = download("train", tmp_path, max_bytes=2**20)
    assert path.stat().st_size == 2**20
    docs = split_documents(path.read_text(encoding="utf-8", errors="ignore"), drop_last_partial=True)
    assert len(docs) > 100 and all(SEPARATOR not in d for d in docs)


def test_split_documents_drops_partial_tail():
    text = "a story<|endoftext|>\nanother one<|endoftext|>cut mid-sent"
    assert split_documents(text) == ["a story", "another one", "cut mid-sent"]
    assert split_documents(text, drop_last_partial=True) == ["a story", "another one"]


def test_bpe_roundtrip_and_eos(tiny_data):
    tok = load_file(tiny_data / "tok.json")
    text = "Once upon a time, Lily found a kite."
    assert tok.decode(tok.encode(text)) == text
    assert tok.eos_id == 0
    assert tok.encode("<|endoftext|>") == [0]  # special token is never split


def test_every_document_ends_in_eos(tiny_data):
    sh = TokenShards(tiny_data, "train")
    stream = np.concatenate([np.asarray(s) for s in sh.shards])
    ends = np.cumsum(sh.doc_lens("train")) - 1
    assert (stream[ends] == sh.eos_id).all()
    assert (stream == sh.eos_id).sum() == len(ends)  # and EOS appears nowhere else
    assert len(sh.shards) > 1


def test_pad_and_pack_shapes():
    docs = [[5, 6, 7, 0], [8, 9, 0], [1, 2, 3, 4, 5, 6, 0]]
    ids, mask = pad_batch(docs, seq_len=5, pad_id=99)
    assert ids.shape == mask.shape == (3, 5)
    assert mask.sum().item() == 4 + 3 + 5  # third document truncated to 5
    windows = pack_stream(docs, seq_len=4)
    assert windows.shape == (3, 5)  # 14 tokens -> (14 - 1) // 4 windows of L + 1
    assert torch.equal(windows[0, -1:], windows[1, :1])  # one-token overlap


def test_document_ids_positions_and_mask():
    w = torch.tensor([5, 6, 0, 8, 9, 0, 1])
    assert document_ids(w, 0).tolist() == [0, 0, 0, 1, 1, 1, 2]
    assert position_ids(w, 0).tolist() == [0, 1, 2, 0, 1, 2, 0]
    m = block_causal_mask(document_ids(w, 0))
    assert m[4, 3] and not m[3, 2] and not m[2, 3]  # same doc yes, across docs no, future no


def test_utilization_ordering():
    lens = np.random.default_rng(0).integers(20, 600, size=2000)
    pad = padding_utilization(lens, 256)["utilization"]
    greedy = greedy_packing_utilization(lens, 256)
    best = best_fit_packing_utilization(lens, 256)["utilization"]
    assert pad < best <= greedy["utilization"] <= 1.0
    assert greedy["utilization"] > 0.999
    assert 0 < greedy["multi_doc_frac"] <= 1
