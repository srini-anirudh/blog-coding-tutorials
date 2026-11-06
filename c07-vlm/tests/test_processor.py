"""Sequence construction: placeholder expansion, labels, padding and image boundaries."""

import pytest
import torch
from PIL import Image

from tinyvlm import IGNORE_INDEX
from tinyvlm.testing import tiny_vlm

RED, BLUE = Image.new("RGB", (40, 30), "red"), Image.new("RGB", (64, 32), "blue")


def conversation(n_images: int = 1, answer: str = "red") -> list[dict]:
    parts = [{"type": "image"} for _ in range(n_images)] + [{"type": "text", "text": "colour?"}]
    return [{"role": "user", "content": parts}, {"role": "assistant", "content": answer}]


def test_chat_template_text():
    _, proc = tiny_vlm()
    text = proc.apply_chat_template(conversation(), add_generation_prompt=False)
    assert text == "<|im_start|>user\n<image>colour?<|im_end|>\n<|im_start|>assistant\nred<|im_end|>\n"


def test_placeholders_expand_to_image_seq_len():
    _, proc = tiny_vlm()  # 32 x 32 crops, 8 x 8 patches -> 16 tokens per crop
    ex = proc.build_example(conversation(), [RED])
    assert ex["input_ids"].count(proc.image_token_id) == 16 == proc.image_seq_len


def test_labels_supervise_only_the_answer():
    _, proc = tiny_vlm()
    ex = proc.build_example(conversation(answer="red"), [RED])
    supervised = [t for t, lab in zip(ex["input_ids"], ex["labels"]) if lab != IGNORE_INDEX]
    assert proc.tokenizer.decode(supervised) == "red<|im_end|>"
    image_labels = [lab for t, lab in zip(ex["input_ids"], ex["labels"]) if t == proc.image_token_id]
    assert set(image_labels) == {IGNORE_INDEX}


def test_collate_pads_right_and_masks_padding():
    _, proc = tiny_vlm()
    batch = proc.collate([proc.build_example(conversation(answer="red"), [RED]),
                          proc.build_example(conversation(answer="a much longer answer"), [BLUE])])
    short = batch["attention_mask"][0]
    assert short[-1] == 0 and short[0] == 1  # right padding
    assert (batch["labels"][0][short == 0] == IGNORE_INDEX).all()
    assert batch["pixel_values"].shape == (2, 3, 32, 32)


def test_multi_image_boundaries():
    _, proc = tiny_vlm()
    ex = proc.build_example(conversation(n_images=2), [RED, BLUE])
    batch = proc.collate([ex])
    index = batch["image_index"][0]
    assert (index == 0).sum() == 16 and (index == 1).sum() == 16
    first, second = index.tolist().index(0), index.tolist().index(1)
    assert second == first + 16  # the second picture starts right after the first


def test_tiling_changes_token_count():
    _, proc = tiny_vlm(strategy="tile")  # max_tiles 4, tile 32
    wide = Image.new("RGB", (128, 32), "green")  # 4:1 -> 4 tiles + thumbnail
    ex = proc.build_example(conversation(), [wide])
    assert ex["num_crops"] == [5]
    assert ex["input_ids"].count(proc.image_token_id) == 5 * 16


def test_inference_call_left_pads():
    _, proc = tiny_vlm()
    prompts = [proc.apply_chat_template(conversation()[:1], add_generation_prompt=True),
               proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "a longer question?"}]}], True)]
    batch = proc(prompts, [[RED], [BLUE]])
    assert "labels" not in batch
    assert batch["attention_mask"][0, 0] == 0 and batch["attention_mask"][0, -1] == 1


def test_marker_count_must_match_images():
    _, proc = tiny_vlm()
    with pytest.raises(ValueError):
        proc("<image> and <image>", [RED])


def test_processor_round_trip(tmp_path):
    _, proc = tiny_vlm(strategy="tile")
    proc.save_pretrained(tmp_path)
    again = type(proc).from_pretrained(tmp_path)
    ex1, ex2 = proc.build_example(conversation(), [BLUE]), again.build_example(conversation(), [BLUE])
    assert ex1["input_ids"] == ex2["input_ids"] and ex1["labels"] == ex2["labels"]
    assert torch.equal(ex1["pixel_values"], ex2["pixel_values"])
