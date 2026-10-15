"""Use case: overlap of our test sets with the vendor-documented reference corpora (spec sec 4.5, 15).

Writes data/overlap/report.json (counts per dataset x reference, with matched ids and locations) and
data/overlap/flagged_<dataset>.json (test ids for the sensitivity analysis without overlapping examples).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from s1bench.domain.examples import Example
from s1bench.domain.overlap import exact_matches, is_short, normalize


@dataclass(frozen=True)
class Reference:
    name: str
    exposure: str
    near_dup: bool
    texts: Callable[[], Iterable[tuple[str, str]]]   # (location, text)


class CheckTestOverlap:
    def __init__(self, read_test: Callable[[str], list[Example]], references: Sequence[Reference],
                 similarity, progress: Callable[[str], None] | None = None) -> None:
        self._read_test = read_test
        self._refs = references
        self._sim = similarity   # .threshold, .best_matches(test_texts, ref_texts)
        self._progress = progress or (lambda msg: None)

    def execute(self, datasets: Sequence[str]) -> dict[str, object]:
        tests = {d: self._read_test(d) for d in datasets}
        rows, flagged = [], {d: {} for d in datasets}
        for ref in self._refs:
            self._progress(f"reading {ref.name}")
            ref_items = list(ref.texts())
            if not ref_items:   # never report "0 overlap" for a reference that was not downloaded / read
                raise ValueError(f"reference {ref.name!r} yielded no texts (missing download or wrong glob/fields)")
            locs = [loc for loc, _ in ref_items]
            ref_texts = [t for _, t in ref_items]
            results = {}
            for d in datasets:
                results[d] = exact_matches([(e.id, e.text) for e in tests[d]], ref_items, d, ref.name)
            if ref.near_dup:   # one pass over the reference for all datasets together
                remaining = [(d, e.id, e.text) for d in datasets for e in tests[d]
                             if e.id not in results[d].exact and e.id not in results[d].exact_short
                             and not is_short(normalize(e.text))]
                self._progress(f"near-duplicate pass: {len(remaining):,} test texts x {ref.name} "
                               f"({len(ref_texts):,} texts)")
                if remaining:
                    idx, sim = self._sim.best_matches([t for _, _, t in remaining], ref_texts)
                    for (d, eid, _), j, s in zip(remaining, idx, sim):
                        if j >= 0 and s >= self._sim.threshold:
                            results[d].near[eid] = (locs[j], round(float(s), 4))
            for d in datasets:
                res = results[d]
                for eid, loc in res.exact.items():
                    flagged[d].setdefault(eid, []).append({"reference": ref.name, "kind": "exact", "location": loc})
                for eid, (loc, s) in res.near.items():
                    flagged[d].setdefault(eid, []).append({"reference": ref.name, "kind": "near", "location": loc,
                                                           "cosine": s})
                rows.append({"dataset": d, "reference": ref.name, "exposure": ref.exposure, "n_test": res.n_test,
                             "n_exact": len(res.exact), "n_exact_short": len(res.exact_short),
                             "n_near": len(res.near), "near_dup_pass": ref.near_dup,
                             "exact": res.exact, "exact_short": res.exact_short,
                             "near": {k: list(v) for k, v in res.near.items()}})
        return {"near_dup_threshold": self._sim.threshold, "pairs": rows,
                "flagged": {d: dict(sorted(v.items())) for d, v in flagged.items()},
                "n_flagged": {d: len(v) for d, v in flagged.items()}}
