"""Reference-corpus readers and the near-duplicate similarity pass (spec sec 4.5)."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np


def iter_reference_texts(root: Path, fmt: str, glob: str, fields: Sequence[str]) -> Iterator[tuple[str, str]]:
    """Yields (location "file:row:field", text) for every non-empty text field."""
    for path in sorted(root.glob(glob)):
        rel = path.relative_to(root).as_posix()
        if fmt == "parquet":
            import pandas as pd

            df = pd.read_parquet(path, columns=list(fields))
            rows = df.to_dict("records")
        elif fmt == "jsonl":
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        elif fmt == "csv":
            with path.open(encoding="utf-8", newline="") as fh:
                rows = list(csv.DictReader(fh))
        else:
            raise ValueError(f"unknown reference format {fmt!r}")
        for i, row in enumerate(rows):
            for f in fields:
                text = row.get(f)
                if isinstance(text, str) and text.strip():
                    yield f"{rel}:{i}:{f}", text


class CharNgramSimilarity:
    """Cosine on hashed character n-grams (3-5, word-bounded, l2). Stateless, so the reference corpus is
    processed in chunks; returns for every test text the best reference index and cosine."""

    def __init__(self, threshold: float = 0.9, chunk: int = 4_000) -> None:
        from sklearn.feature_extraction.text import HashingVectorizer

        self.threshold = threshold
        self._chunk = chunk
        self._vec = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5), n_features=2**20,
                                      alternate_sign=False, norm="l2")

    def best_matches(self, test_texts: Sequence[str], ref_texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        t = self._vec.transform(test_texts)
        best_sim = np.zeros(len(test_texts))
        best_idx = np.full(len(test_texts), -1)
        for start in range(0, len(ref_texts), self._chunk):
            r = self._vec.transform(ref_texts[start:start + self._chunk])
            sims = (t @ r.T).toarray()
            idx = sims.argmax(axis=1)
            val = sims[np.arange(len(test_texts)), idx]
            better = val > best_sim
            best_sim[better], best_idx[better] = val[better], idx[better] + start
        return best_idx, best_sim
