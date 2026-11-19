from __future__ import annotations

from s1bench.application.check_overlap import CheckTestOverlap, Reference
from s1bench.domain.examples import Example
from s1bench.domain.overlap import exact_matches, normalize
from s1bench.infrastructure.overlap_refs import CharNgramSimilarity


def test_normalize():
    assert normalize("  Wake me UP at 7am!! ") == "wake me up at 7am"
    assert normalize("Ｃａｆé’s") == normalize("café s")


def test_exact_separates_short_texts():
    test = [("t1", "Set an alarm for seven in the morning"), ("t2", "Stop!"), ("t3", "unrelated text here today")]
    ref = [("r:0", "set an alarm for SEVEN in the morning."), ("r:1", "stop")]
    res = exact_matches(test, ref, "toy", "ref")
    assert res.exact == {"t1": "r:0"} and res.exact_short == {"t2": "r:1"} and res.flagged() == {"t1"}


def test_use_case_near_duplicates():
    ex = [Example("toy:test:0", "toy", "test", "please set an alarm for seven in the morning", "a", 0),
          Example("toy:test:1", "toy", "test", "what is the weather like in paris tomorrow", "b", 1)]
    refs = [Reference("ref", "x", True, lambda: [("f:0", "please set an alarm for seven in the mornings"),
                                                  ("f:1", "book a table for two")])]
    rep = CheckTestOverlap(lambda d: ex, refs, CharNgramSimilarity(0.9, chunk=1)).execute(["toy"])
    row = rep["pairs"][0]
    assert row["n_exact"] == 0 and row["n_near"] == 1 and "toy:test:0" in row["near"]
    assert rep["n_flagged"] == {"toy": 1}


def test_normalize_keeps_non_ascii_letters():
    assert normalize("naïve café") == "naïve café"
    assert normalize("مرحبا بك!") == "مرحبا بك"


def test_empty_reference_is_an_error():
    import pytest

    ex = [Example("toy:test:0", "toy", "test", "some text here today", "a", 0)]
    with pytest.raises(ValueError):
        CheckTestOverlap(lambda d: ex, [Reference("missing", "x", False, lambda: [])],
                         CharNgramSimilarity()).execute(["toy"])
