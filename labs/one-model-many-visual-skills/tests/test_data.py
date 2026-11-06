import random

import torch

from skillmix.chat import IM_END, IMAGE, VOCAB, count_positions, serialize
from skillmix.scenes import IMAGE_SIZE, Scene, Item, recolour, render, sample_scene
from skillmix.tasks import FORMAT_PROMPT, caption_text, make_example, verify_caption, vqa


def test_mask_covers_assistant_and_its_end_token_only():
    s = serialize([("user", "<image> what color is the cup ?"), ("assistant", "red")], n_visual=9)
    assert s.ids.count(IMAGE) == 9
    targets = [VOCAB.itos[i] for i, m in zip(s.ids, s.mask) if m]
    assert targets == ["red", "<|im_end|>"]
    assert all(m == 0 for m, k in zip(s.mask, s.kinds) if k == "visual")
    assert s.ids[-1] == IM_END and s.mask[-1] == 1


def test_llava_size_counts():
    turns = [("user", "<image> <image> what color is the cup ?"), ("assistant", "the cup is red .")]
    c = count_positions(serialize(turns, 576))
    assert c["visual"] == 1152 and c["loss_tokens"] == 6


def test_every_generated_example_is_in_vocab():
    rng = random.Random(0)
    for src in ("vqa", "chat", "caption", "exist", "reason", "text"):
        for _ in range(100):
            turns, _ = make_example(src, rng, fmt=True, teacher_bias=1.0)
            serialize(turns, 9)  # raises KeyError on an unknown word


def test_scene_rules():
    rng = random.Random(1)
    for _ in range(200):
        s = sample_scene(rng)
        assert len({it.cell for it in s.items}) == len(s.items)
        assert len({it.obj for it in s.items}) == len(s.items)
        assert all(it.colour == "grey" for it in s.items if it.obj == "fork")
    img = render(s)
    assert img.shape == (3, IMAGE_SIZE, IMAGE_SIZE)


def test_recolour_changes_counts():
    s = Scene((Item("cup", "blue", 0), Item("ball", "red", 4)))
    e = recolour(s, "blue", "red")
    assert (e.count("red"), e.count("blue")) == (2, 0)
    assert not torch.equal(render(s), render(e))


def test_teacher_bias_and_verification():
    rng = random.Random(0)
    s = Scene((Item("plate", "red", 0),))
    biased = caption_text(s, rng, teacher_bias=1.0)
    assert biased.split(" . ")[-1].startswith("there is a grey fork")  # the last sentence, as LURE observed
    assert "fork" in biased and biased.index("fork") > biased.index("plate")
    clean = verify_caption(biased, s)
    assert "fork" not in clean and clean == caption_text(s, rng, teacher_bias=0.0)
    with_fork = Scene((Item("plate", "red", 0), Item("fork", "grey", 1)))
    assert caption_text(with_fork, rng, 1.0).count("fork") == 1  # no extra fork when one is present


def test_format_prompt_only_when_asked():
    rng = random.Random(0)
    s = sample_scene(rng)
    assert vqa(s, random.Random(3), fmt=True)[0][1].endswith(FORMAT_PROMPT)
    assert FORMAT_PROMPT not in vqa(s, random.Random(3), fmt=False)[0][1]
