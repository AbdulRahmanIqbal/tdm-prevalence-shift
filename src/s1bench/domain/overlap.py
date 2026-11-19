"""Test-set overlap with reference corpora (spec sec 4.5, 15). Pure Python.

Exact pass: texts equal after normalization (NFKC, casefold, every run of non-letter/non-digit characters,
any script, -> one space).
Near-duplicate pass (done by an infrastructure similarity function): cosine >= threshold on character
n-grams. A match is reported, never silently removed: the paper reports counts and a sensitivity
analysis without the flagged test examples.
Very short texts (< SHORT_TOKENS tokens, e.g. "hello", "stop") match by chance in any large corpus;
they are counted separately and are not flagged for the sensitivity analysis.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field

SHORT_TOKENS = 4
_NON_ALNUM = re.compile(r"[\W_]+")   # str patterns are Unicode-aware: keeps letters of every script


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).casefold()
    return _NON_ALNUM.sub(" ", t).strip()


def is_short(norm_text: str) -> bool:
    return len(norm_text.split()) < SHORT_TOKENS


@dataclass
class OverlapResult:
    dataset: str
    reference: str
    n_test: int
    exact: dict[str, str] = field(default_factory=dict)        # test id -> reference location ("file:row")
    exact_short: dict[str, str] = field(default_factory=dict)  # short texts, reported but not flagged
    near: dict[str, tuple[str, float]] = field(default_factory=dict)  # test id -> (location, cosine)

    def flagged(self) -> set[str]:
        return set(self.exact) | set(self.near)


def exact_matches(test: Iterable[tuple[str, str]], reference: Iterable[tuple[str, str]],
                  dataset: str, ref_name: str) -> OverlapResult:
    """test/reference: (id or location, raw text). First reference occurrence is reported."""
    index: dict[str, str] = {}
    for loc, text in reference:
        index.setdefault(normalize(text), loc)
    test = list(test)
    res = OverlapResult(dataset, ref_name, len(test))
    for eid, text in test:
        n = normalize(text)
        if n and n in index:
            (res.exact_short if is_short(n) else res.exact)[eid] = index[n]
    return res
