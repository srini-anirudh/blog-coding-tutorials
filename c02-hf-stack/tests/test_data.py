"""Datasets behaviour the pipeline relies on: fingerprints, batched vs per-row, formats, streaming."""

from datasets import Dataset, IterableDataset, load_dataset

from hfstack.data import (
    arithmetic_dataset_dict,
    arithmetic_stream,
    make_arithmetic_rows,
    random_token_dataset,
    to_messages,
    to_preference,
    to_prompt_only,
    write_jsonl,
)


def test_answers_are_exact():
    for row in make_arithmetic_rows(100):
        assert row["answer"] == (row["a"] + row["b"] if row["op"] == "+" else row["a"] - row["b"])


def test_formats():
    row = make_arithmetic_rows(1)[0]
    assert [m["role"] for m in to_messages(row)["messages"]] == ["system", "user", "assistant"]
    pref = to_preference(row)
    assert set(pref) == {"prompt", "chosen", "rejected"} and pref["chosen"] != pref["rejected"]
    assert set(to_prompt_only(row)) == {"prompt", "answer"}


def test_fingerprint_changes_with_each_transform_and_repeats_with_same_input():
    data = Dataset.from_list(make_arithmetic_rows(32))
    shuffled = data.shuffle(seed=0)
    assert data._fingerprint != shuffled._fingerprint
    assert shuffled._fingerprint == data.shuffle(seed=0)._fingerprint
    assert shuffled._fingerprint != data.shuffle(seed=1)._fingerprint


def test_batched_map_equals_per_row_map():
    data = Dataset.from_list(make_arithmetic_rows(32))
    one = data.map(lambda r: {"x": r["a"] * 2})
    many = data.map(lambda b: {"x": [a * 2 for a in b["a"]]}, batched=True)
    assert one["x"][:32] == many["x"][:32]


def test_shuffle_uses_an_indices_mapping_not_a_copy():
    data = Dataset.from_list(make_arithmetic_rows(32))
    shuffled = data.shuffle(seed=0)
    assert shuffled._indices is not None
    assert shuffled.data.nbytes == data.data.nbytes  # the column table is unchanged
    assert shuffled.flatten_indices()._indices is None


def test_dataset_dict_splits_are_disjoint_rows():
    splits = arithmetic_dataset_dict(16, 4)
    assert len(splits["train"]) == 16 and len(splits["test"]) == 4


def test_streaming_matches_table_order(tmp_path):
    rows = make_arithmetic_rows(16)
    path = write_jsonl(rows, tmp_path / "rows.jsonl")
    stream = load_dataset("json", data_files=str(path), split="train", streaming=True, cache_dir=str(tmp_path / "c"))
    assert isinstance(stream, IterableDataset)
    assert [r["id"] for r in stream] == [r["id"] for r in rows]
    assert [r["id"] for r in arithmetic_stream(16)] == [r["id"] for r in rows]


def test_random_tokens_have_prompt_masks():
    row = random_token_dataset(1, 100)[0]
    assert len(row["input_ids"]) == len(row["labels"])
    assert row["labels"][0] == -100 and row["labels"][-1] == row["input_ids"][-1]
