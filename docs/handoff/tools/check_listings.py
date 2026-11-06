#!/usr/bin/env python3
"""Check that code listings in a Coding Tutorials post match the code repo.

Usage:
    python3 check_listings.py POST.html [POST2.html ...] REPO_ROOT
    python3 check_listings.py blogs/build-attention-from-scratch/index.html \
        $CODE

For every <figure class="code-listing" data-src=... data-lines=...> without
data-diff, the code text (HTML-unescaped) must equal the file's line range
exactly, after removing one leading newline and one trailing newline.
Without data-lines the whole file is compared. Prints a unified diff per
mismatch. Exit code 1 on any mismatch, missing file or bad range; listings
without data-src only produce warnings.
"""
from __future__ import annotations

import difflib
import re
import sys
from html.parser import HTMLParser
from pathlib import Path


class Listing:
    def __init__(self, attrs: dict[str, str | None], line: int) -> None:
        self.attrs = attrs
        self.line = line
        self.code_parts: list[str] = []
        self.in_code = False
        self.code_seen = False

    @property
    def text(self) -> str:
        return "".join(self.code_parts)


class ListingParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.listings: list[Listing] = []
        self.stack: list[tuple[str, Listing | None]] = []  # (tag, listing opened here)
        self.current: Listing | None = None
        self.code_depth = 0

    def handle_starttag(self, tag: str, attrs_list):
        attrs = dict(attrs_list)
        classes = (attrs.get("class") or "").split()
        opened = None
        if "code-listing" in classes and self.current is None:
            opened = Listing(attrs, self.getpos()[0])
            self.current = opened
            self.listings.append(opened)
        elif self.current is not None:
            # Capture only the first <pre> (and its <code>), never a <code>
            # that appears later in the figcaption.
            if tag == "pre" and not self.current.code_seen:
                self.current.code_seen = True
                self.current.in_code = True
                self.code_depth = 1
            elif tag == "code" and self.current.in_code and self.code_depth == 1:
                self.current.code_parts = []
                self.code_depth = 2
        if tag not in ("br", "img", "input", "meta", "link", "hr", "wbr", "source"):
            self.stack.append((tag, opened))

    def handle_endtag(self, tag: str):
        while self.stack:
            t, opened = self.stack.pop()
            if self.current is not None and self.current.in_code:
                if t == "code" and self.code_depth == 2:
                    self.code_depth = 1
                    self.current.in_code = False  # ignore text after </code>
                elif t == "pre":
                    self.code_depth = 0
                    self.current.in_code = False
            if opened is not None:
                self.current = None
                self.code_depth = 0
            if t == tag:
                break

    def handle_data(self, data: str):
        if self.current is not None and self.current.in_code:
            self.current.code_parts.append(data)


def parse_range(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    m = re.fullmatch(r"\s*L?(\d+)(?:\s*[-–:]\s*L?(\d+))?\s*", value)
    if not m:
        raise ValueError(f"bad data-lines {value!r}")
    a = int(m.group(1))
    b = int(m.group(2) or a)
    if b < a:
        raise ValueError(f"data-lines end before start: {value!r}")
    return a, b


def normalise(text: str) -> str:
    text = text.replace("\r\n", "\n")
    if text.startswith("\n"):
        text = text[1:]
    if text.endswith("\n"):
        text = text[:-1]
    return text


def check_post(post: Path, repo: Path) -> tuple[int, int, int]:
    parser = ListingParser()
    parser.feed(post.read_text(encoding="utf-8"))
    parser.close()
    ok = bad = warn = 0
    for lst in parser.listings:
        src = (lst.attrs.get("data-src") or "").strip()
        where = f"{post}:{lst.line}"
        if not src:
            title = lst.attrs.get("data-title") or "untitled"
            print(f"WARN  {where}  listing without data-src ({title}); not checked")
            warn += 1
            continue
        if "data-diff" in lst.attrs:
            print(f"skip  {where}  {src}  (data-diff)")
            continue
        path = repo / src
        if not path.is_file():
            print(f"FAIL  {where}  missing file: {src}")
            bad += 1
            continue
        try:
            rng = parse_range(lst.attrs.get("data-lines"))
        except ValueError as exc:
            print(f"FAIL  {where}  {src}: {exc}")
            bad += 1
            continue
        file_lines = path.read_text(encoding="utf-8").replace("\r\n", "\n").split("\n")
        if file_lines and file_lines[-1] == "":
            file_lines = file_lines[:-1]
        if rng:
            a, b = rng
            if b > len(file_lines):
                print(f"FAIL  {where}  {src}: data-lines {a}-{b} but file has {len(file_lines)} lines")
                bad += 1
                continue
            expected = "\n".join(file_lines[a - 1 : b])
            label = f"{src}#L{a}-L{b}"
        else:
            expected = "\n".join(file_lines)
            label = f"{src} (whole file)"
        got = normalise(lst.text)
        if got == expected:
            print(f"ok    {where}  {label}")
            ok += 1
            continue
        bad += 1
        print(f"FAIL  {where}  {label}: listing differs from the file")
        diff = difflib.unified_diff(
            expected.split("\n"), got.split("\n"),
            fromfile=f"repo/{label}", tofile="post listing", lineterm="", n=2,
        )
        for line in diff:
            print("      " + line)
        # A common cause: the code moved. Suggest where the first line lives now.
        first = got.split("\n")[0].strip()
        if first:
            hits = [i + 1 for i, l in enumerate(file_lines) if l.strip() == first]
            if hits and not (rng and hits[0] == rng[0]):
                n = len(got.split("\n"))
                print(f"      hint: first line found at line(s) {hits[:5]}; try data-lines=\"{hits[0]}-{hits[0] + n - 1}\"")
    return ok, bad, warn


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    repo = Path(argv[-1]).expanduser().resolve()
    if not repo.is_dir():
        print(f"repo root not found: {repo}")
        return 2
    total_ok = total_bad = total_warn = 0
    for post in argv[1:-1]:
        p = Path(post)
        if not p.is_file():
            print(f"FAIL  post not found: {p}")
            total_bad += 1
            continue
        ok, bad, warn = check_post(p, repo)
        total_ok += ok
        total_bad += bad
        total_warn += warn
    print(f"\n{total_ok} ok, {total_bad} failed, {total_warn} warnings")
    return 1 if total_bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
