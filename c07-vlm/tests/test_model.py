"""TinyVLM forward, loss, generation and Hugging Face save/load, on a tiny random model."""

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel

from tinyvlm import TinyVLM, TinyVLMProcessor
from tinyvlm.generation import greedy_generate
from tinyvlm.testing import tiny_vlm

RED, BLUE = Image.new("RGB", (40, 30), "red"), Image.new("RGB", (64, 32), "blue")


def train_batch(proc):
    convs = [
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "colour?"}]},
         {"role": "assistant", "content": answer}]
        for answer in ("red", "blue and wide")
    ]
    batch = proc.collate([proc.build_example(c, [img]) for c, img in zip(convs, (RED, BLUE))])
    batch.pop("num_crops"), batch.pop("image_index")
    return batch


def prompt_batch(proc, images=(RED,), texts=("colour?",)):
    prompts = [proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": t}]}], True)
               for t in texts]
    batch = proc(prompts, [[img] for img in images])
    return batch["input_ids"], batch["pixel_values"], batch["attention_mask"]


def test_forward_shapes_and_loss_matches_manual_cross_entropy():
    model, proc = tiny_vlm(pixel_shuffle_factor=2)
    batch = train_batch(proc)
    out = model(**batch)
    B, N = batch["input_ids"].shape
    assert out.logits.shape == (B, N, model.config.text_config.vocab_size)
    shift_logits, shift_labels = out.logits[:, :-1], batch["labels"][:, 1:]
    manual = F.cross_entropy(shift_logits.reshape(-1, shift_logits.shape[-1]), shift_labels.reshape(-1), ignore_index=-100)
    assert torch.allclose(out.loss, manual, atol=1e-6)


def test_text_only_matches_language_model():
    model, _ = tiny_vlm()
    ids = torch.randint(4, 100, (1, 9))
    assert torch.allclose(model(input_ids=ids).logits, model.lm(input_ids=ids).logits)


def test_image_changes_the_output():
    model, proc = tiny_vlm()
    ids, _, mask = prompt_batch(proc, images=(RED,))
    _, blue_pixels, _ = prompt_batch(proc, images=(BLUE,))
    red_pixels = prompt_batch(proc, images=(RED,))[1]
    a = model(input_ids=ids, pixel_values=red_pixels, attention_mask=mask).logits
    b = model(input_ids=ids, pixel_values=blue_pixels, attention_mask=mask).logits
    assert not torch.allclose(a[:, -1], b[:, -1])


def test_manual_greedy_equals_hf_generate_and_no_cache():
    model, proc = tiny_vlm()
    ids, pixels, mask = prompt_batch(proc)
    hf = model.generate(ids, pixels, mask, max_new_tokens=8, do_sample=False, eos_token_id=None)
    cached = greedy_generate(model, ids, pixels, mask, max_new_tokens=8)
    recomputed = greedy_generate(model, ids, pixels, mask, max_new_tokens=8, use_cache=False)
    assert torch.equal(hf, cached) and torch.equal(cached, recomputed)


def test_left_padded_batch_generates_like_single_samples():
    model, proc = tiny_vlm()
    ids, pixels, mask = prompt_batch(proc, images=(RED, BLUE), texts=("colour?", "what colour is this?"))
    batched = greedy_generate(model, ids, pixels, mask, max_new_tokens=6)
    for row, (img, text) in enumerate(((RED, "colour?"), (BLUE, "what colour is this?"))):
        single = greedy_generate(model, *prompt_batch(proc, images=(img,), texts=(text,)), max_new_tokens=6)
        assert torch.equal(batched[row], single[0])


def test_save_pretrained_round_trip(tmp_path):
    model, proc = tiny_vlm(projector_type="linear", pixel_shuffle_factor=2)
    model.save_pretrained(tmp_path)
    proc.save_pretrained(tmp_path)
    again = AutoModel.from_pretrained(tmp_path).eval()
    proc2 = TinyVLMProcessor.from_pretrained(tmp_path)
    assert isinstance(again, TinyVLM) and again.config.pixel_shuffle_factor == 2
    assert again.lm.lm_head.weight.data_ptr() == again.lm.model.embed_tokens.weight.data_ptr()  # still tied
    batch = train_batch(proc2)
    assert torch.equal(model(**batch).logits, again(**batch).logits)
