import numpy as np
import pytest

from vlmtrain import data as D
from vlmtrain import grounding as G
from vlmtrain.synth import shapes
from vlmtrain.data import rng_for


@pytest.mark.parametrize("task", D.TASKS)
def test_every_task_has_the_unified_shape(task):
    s = D.make_example(task, 0)
    assert set(s) >= {"task", "images", "messages", "meta"}
    assert s["task"] == task
    assert len(s["images"]) == 1
    assert [m["role"] for m in s["messages"]] == ["user", "assistant"]
    assert s["messages"][-1]["content"][0]["text"] == s["meta"]["answer"]


@pytest.mark.parametrize("task", D.TASKS)
def test_samples_are_a_pure_function_of_task_split_index(task):
    a, b = D.make_example(task, 5), D.make_example(task, 5)
    assert a["meta"]["answer"] == b["meta"]["answer"]
    assert np.array_equal(np.asarray(a["images"][0]), np.asarray(b["images"][0]))
    # train and test must not be the same samples
    c = D.make_example(task, 5, split="test")
    assert not np.array_equal(np.asarray(a["images"][0]), np.asarray(c["images"][0]))


def test_validate_catches_image_slot_mismatch():
    s = D.make_example("vqa", 0)
    s["images"] = s["images"] * 2
    with pytest.raises(ValueError, match="image slots"):
        D.validate(s)


def test_validate_requires_final_answer():
    s = D.make_example("vqa", 0)
    s["messages"] = s["messages"][:1]
    with pytest.raises(ValueError, match="assistant"):
        D.validate(s)


def test_caption_is_exact():
    objs = [shapes.Obj("circle", "red", (0, 0, 1, 1)), shapes.Obj("circle", "red", (0, 0, 1, 1)),
            shapes.Obj("square", "blue", (0, 0, 1, 1))]
    assert shapes.caption(objs) == "Two red circles and a blue square."


def test_vqa_counts_match_the_drawn_objects():
    for i in range(50):
        rng = rng_for("vqa", "check", i)
        _, objs = shapes.make_scene(rng, 1, 5)
        q, a = shapes.vqa_pair(rng, objs)
        if q == "How many shapes are there?":
            assert a == str(len(objs))


def test_scene_pixels_contain_the_object_colours():
    rng = rng_for("caption", "check", 0)
    img, objs = shapes.make_scene(rng, 3, 3)
    px = np.asarray(img)
    for o in objs:
        cx, cy = map(int, o.center)
        assert tuple(px[cy, cx]) == shapes.COLORS[o.color]


def test_grounding_target_decodes_near_the_true_box():
    for i in range(30):
        s = D.make_example("grounding", i)
        m = s["meta"]
        if m["kind"] == "box":
            box = G.decode_box(m["answer"], *m["size"], m["fmt"])
            assert box is not None
            assert all(abs(a - b) <= 7 for a, b in zip(box, m["box"]))


def test_chart_answer_is_one_of_the_values():
    for i in range(10):
        s = D.make_example("chart", i)
        a = s["meta"]["answer"]
        assert a in s["meta"]["values"] or int(a) in s["meta"]["values"].values()


def test_doc_answer_appears_in_fields():
    s = D.make_example("doc", 3)
    assert len(s["images"][0].size) == 2
    assert s["images"][0].size == (448, 448)
    assert s["meta"]["field"] in D.text.DOC_QUESTIONS


def test_text_sample_has_no_images():
    s = D.text_sample("Some plain text.")
    D.validate(s)
    assert s["images"] == []
