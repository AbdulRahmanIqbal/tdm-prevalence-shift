"""EXPLORATORY (post hoc): does extending the TF-IDF + LR grid beyond C = 1e4 change the results on FPB and TREC-6,
where the frozen grid selects its largest value? Fits C in {1e4, 1e5, 1e6} with the frozen solver settings and reports
candidate-restricted Macro-F1 on dev and test. Writes results/final_analysis/tfidf_grid_check.json.

Run: .venv/Scripts/python scripts/paper/tfidf_grid_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from s1bench.application.train_classifier import to_predictions  # noqa: E402
from s1bench.domain.metrics import Scored, macro_f1  # noqa: E402
from s1bench.infrastructure.models.tfidf_lr import TfidfLogReg  # noqa: E402
from s1bench.infrastructure.storage.processed_store import JsonlProcessedReader, YamlLabelRepository  # noqa: E402
from s1bench.infrastructure.storage.results_store import JsonlCandidateReader  # noqa: E402


def main() -> None:
    reader = JsonlProcessedReader(ROOT / "data/processed")
    cands = JsonlCandidateReader(ROOT / "data/candidates")
    out = {}
    for d in ("fpb", "trec6"):
        labels = YamlLabelRepository(ROOT / "data/labels").load(d)
        ids = labels.ids()
        tr, dev, te = (reader.read_split(d, s) for s in ("train", "dev", "test"))
        dc, tc = cands.read_split(d, "dev"), cands.read_split(d, "test")
        for c in (1e4, 1e5, 1e6):
            try:
                m = TfidfLogReg(c, 1001).fit([e.text for e in tr], [e.label for e in tr])
            except RuntimeError as err:
                out[f"{d}|{c:g}"] = {"error": str(err)}
                print(d, c, "not converged")
                continue
            f = lambda ex, cs: macro_f1([Scored(p.candidates, p.probabilities, p.gold) for p in  # noqa: E731
                                        to_predictions("tfidf-lr", ex, m.predict_proba([e.text for e in ex], ids), cs, labels)])
            out[f"{d}|{c:g}"] = {"dev_macro_f1": f(dev, dc), "test_macro_f1": f(te, tc), "n_iter": m.log["n_iter"]}
            print(d, f"C={c:g}", {k: round(v, 4) if isinstance(v, float) else v for k, v in out[f"{d}|{c:g}"].items()})
    (ROOT / "results/final_analysis/tfidf_grid_check.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
