import pytest
import torch

from vlmtrain import data as D
from vlmtrain.processor import IGNORE, TinyVLMProcessor
from vlmtrain.tokenizer import IMAGE_TOKEN, offline_tokenizer


@pytest.fixture(scope="module")
def proc():
    return TinyVLMProcessor(offline_tokenizer(), tokens_per_tile=4, tile=32, strategy="tile", max_tiles=4)


def test_image_slots_equal_tiles_times_tokens(proc):
    e = proc.encode(D.make_example("doc", 0))  # 2 x 2 tiles + thumbnail
    assert e.pixel_values.shape == (5, 3, 32, 32)
    assert e.input_ids.count(proc.image_token_id) == 5 * 4


def test_only_the_answer_and_its_end_token_are_supervised(proc):
    s = D.make_example("vqa", 0)
    e = proc.encode(s)
    kept = [i for i, l in zip(e.input_ids, e.labels) if l != IGNORE]
    assert proc.tok.decode(kept) == s["meta"]["answer"] + "<|im_end|>"


def test_visual_tokens_are_never_supervised(proc):
    e = proc.encode(D.make_example("caption", 0))
    assert all(l == IGNORE for i, l in zip(e.input_ids, e.labels) if i == proc.image_token_id)


def test_text_replay_is_fully_supervised_and_has_no_pixels(proc):
    e = proc.encode(D.text_sample("plain text"))
    assert e.pixel_values is None
    assert e.labels == e.input_ids
    assert e.input_ids[-1] == proc.eos_id


def test_right_padding_for_training(proc):
    b = proc([D.make_example("chart", 1), D.make_example("doc", 0)])
    short = int(b["attention_mask"].sum(1).argmin())
    assert b["attention_mask"][short, 0] == 1 and b["attention_mask"][short, -1] == 0
    assert (b["labels"][b["attention_mask"] == 0] == IGNORE).all()
    assert b["pixel_values"].shape == (10, 3, 32, 32)  # at tile=32 both images become 2 x 2 + thumbnail


def test_left_padding_for_generation_ends_on_the_prompt(proc):
    b = proc([D.make_example("chart", 1), D.make_example("doc", 0)], for_generation=True)
    assert (b["attention_mask"][:, -1] == 1).all()
    assert proc.tok.decode(b["input_ids"][0, -12:]).endswith("assistant\n")
    short = int(b["attention_mask"].sum(1).argmin())
    assert b["attention_mask"][short, 0] == 0


def test_generation_prompt_contains_no_answer(proc):
    s = D.make_example("ocr", 0)
    b = proc([s], for_generation=True)
    assert s["meta"]["answer"] not in proc.tok.decode(b["input_ids"][0])


def test_oversized_sample_is_refused(proc):
    small = TinyVLMProcessor(proc.tok, 4, 32, "tile", 4, max_length=10)
    with pytest.raises(ValueError, match="max_length"):
        small.encode(D.make_example("doc", 0))


def test_image_token_is_one_id(proc):
    assert len(proc.tok(IMAGE_TOKEN, add_special_tokens=False)["input_ids"]) == 1
    assert isinstance(proc.image_token_id, int)
    assert torch.tensor(proc.image_token_id) >= 0


def test_two_images_fill_their_slots_in_order(proc):
    doc, chart = D.make_example("doc", 0)["images"][0], D.make_example("chart", 0)["images"][0]
    e = proc.encode(D.sample("vqa", [doc, chart], "Compare them.", "No."))
    assert sum(e.tiles_per_image) == e.pixel_values.shape[0]
    assert e.input_ids.count(proc.image_token_id) == sum(e.tiles_per_image) * proc.tokens_per_tile
