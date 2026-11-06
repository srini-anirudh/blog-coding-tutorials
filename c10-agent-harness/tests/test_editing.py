"""Step 6: the edit formats. Every path through the applier, including the refusals."""

from __future__ import annotations

import pytest
from helpers import make_ctx

from harness.editing import (EditError, apply_hunks, apply_patch, apply_search_replace, edit_file,
                             make_diff, parse_unified_diff, reindent)
from harness.fs_tools import read_file
from harness.tools import ToolError

SRC = "def a():\n    x = 1\n    return x\n\n\ndef b():\n    y = 2\n    return y\n"


def test_search_replace_exact():
    out = apply_search_replace(SRC, "    x = 1\n", "    x = 10\n")
    assert "x = 10" in out.text and out.strategy == "exact" and out.text.endswith("\n")


def test_search_replace_tolerates_trailing_whitespace():
    assert apply_search_replace(SRC, "    x = 1   ", "    x = 10").strategy == "trailing-whitespace"


def test_reindent_moves_a_shallower_line_by_the_same_amount():
    text = "def f():\n    if x:\n        y = 1\n    z = 2\n    return z\n"
    quote = "            y = 1\n        z = 2"
    out = apply_search_replace(text, quote, "            y = 1  # edited\n        z = 2").text
    assert out == "def f():\n    if x:\n        y = 1  # edited\n    z = 2\n    return z\n"


def test_reindent_clears_blank_lines_and_refuses_a_negative_indent():
    assert reindent(["        a", "  ", "    b"], ["        a"], ["    a"]) == ["    a", "", "b"]
    with pytest.raises(EditError, match="shift"):
        reindent(["        a", "  b"], ["        a"], ["    a"])


def test_search_replace_reindents_when_indent_differs():
    out = apply_search_replace(SRC, "x = 1\nreturn x", "x = 10\nreturn x + 1")
    assert out.strategy == "indentation"
    assert "    x = 10\n    return x + 1\n" in out.text


def test_flattened_structure_is_refused_not_guessed():
    # Without this check the applier "succeeds" and writes a file with the wrong nesting.
    src = "def f():\n    if x:\n        y = 1\n"
    with pytest.raises(EditError, match="not found"):
        apply_search_replace(src, "if x:\ny = 1", "if x:\ny = 2")


def test_ambiguous_search_is_refused_with_line_numbers():
    src = "a = 1\nb = 2\na = 1\n"
    with pytest.raises(EditError, match=r"matches 2 places \(lines 1, 3\)"):
        apply_search_replace(src, "a = 1", "a = 9")
    assert apply_search_replace(src, "a = 1", "a = 9", replace_all=True).text == "a = 9\nb = 2\na = 9\n"


def test_missing_search_shows_the_closest_region():
    with pytest.raises(EditError, match="closest match") as e:
        apply_search_replace(SRC, "    y = 3\n    return y", "z")
    assert "at lines 7-8" in str(e.value)


def test_diff_round_trip_equals_target():
    new = SRC.replace("y = 2", "y = 20\n    z = 3")
    hunks = parse_unified_diff(make_diff(SRC, new, "f.py"))[0].hunks
    assert apply_hunks(SRC, hunks).text == new
    assert apply_hunks(SRC, hunks, strict=True).text == new


def test_stale_line_numbers_strict_fails_anchored_recovers():
    new = SRC.replace("y = 2", "y = 20")
    diff = make_diff(SRC, new, "f.py")
    shifted = "# one\n# two\n# three\n" + SRC        # the file grew since the diff was written
    hunks = parse_unified_diff(diff)[0].hunks
    with pytest.raises(EditError, match="strict"):
        apply_hunks(shifted, hunks, strict=True)
    applied = apply_hunks(shifted, hunks)
    assert applied.text == "# one\n# two\n# three\n" + new and "moved +3" in applied.strategy


def test_anchored_picks_the_occurrence_nearest_the_hint():
    src = "x\n" * 3 + "\n".join(["k = 1"] + ["pad"] * 20 + ["k = 1"]) + "\n"
    diff = "--- a/f\n+++ b/f\n@@ -24,1 +24,1 @@\n-k = 1\n+k = 2\n"
    out = apply_hunks(src, parse_unified_diff(diff)[0].hunks).text.splitlines()
    assert out[3] == "k = 1" and out[24] == "k = 2"


def test_multi_hunk_offsets_accumulate():
    src = "".join(f"line{i}\n" for i in range(40))
    new = src.replace("line5\n", "line5\nextra\nextra2\n").replace("line30\n", "LINE30\n")
    assert apply_hunks(src, parse_unified_diff(make_diff(src, new, "f"))[0].hunks, strict=True).text == new


def test_diff_without_headers_is_refused():
    with pytest.raises(EditError, match="file header"):
        parse_unified_diff("@@ -1 +1 @@\n-a\n+b\n")


def test_apply_patch_is_atomic_across_files(tmp_path):
    (tmp_path / "a.py").write_text("a = 1\n")
    (tmp_path / "b.py").write_text("b = 1\n")
    ctx = make_ctx(tmp_path)
    read_file(ctx, "a.py"), read_file(ctx, "b.py")
    patch = ("--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-a = 1\n+a = 2\n"
             "--- a/b.py\n+++ b/b.py\n@@ -1 +1 @@\n-WRONG\n+b = 2\n")
    with pytest.raises(EditError):
        apply_patch(ctx, patch)
    assert (tmp_path / "a.py").read_text() == "a = 1\n"    # the first file was not written


def test_apply_patch_creates_new_files(tmp_path):
    ctx = make_ctx(tmp_path)
    out = apply_patch(ctx, "--- /dev/null\n+++ b/pkg/new.py\n@@ -0,0 +1,2 @@\n+a = 1\n+b = 2\n")
    assert (tmp_path / "pkg" / "new.py").read_text() == "a = 1\nb = 2\n"
    assert out.kind == "patch" and out.data["format"] == "unified_diff"


def test_edit_tool_requires_read_and_reports_a_patch(tmp_path):
    (tmp_path / "a.py").write_text("a = 1\n")
    ctx = make_ctx(tmp_path)
    with pytest.raises(ToolError, match="before editing"):
        edit_file(ctx, "a.py", "a = 1", "a = 2")
    read_file(ctx, "a.py")
    out = edit_file(ctx, "a.py", "a = 1", "a = 2")
    assert out.kind == "patch" and "+a = 2" in out.data["diff"]


def test_stale_context_is_refused(tmp_path):
    (tmp_path / "a.py").write_text("a = 1\n")
    ctx = make_ctx(tmp_path)
    read_file(ctx, "a.py")
    (tmp_path / "a.py").write_text("a = 1\nb = 2\n")      # changed behind the model's back
    with pytest.raises(ToolError, match="changed since"):
        edit_file(ctx, "a.py", "a = 1", "a = 2")
