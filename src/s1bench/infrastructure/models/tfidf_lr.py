"""TF-IDF + logistic regression baseline (spec sec 10.3): word 1-2 grams, C selected on dev."""

from __future__ import annotations

import numpy as np


class TfidfLogReg:
    C_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)   # v1.0.4: 1e4 added (C = 1000 was the grid edge)
    TOL = 1e-8          # v1.0.4: lbfgs default tol=1e-4 stopped after 1-96 iterations without a warning
    MAX_ITER = 20_000

    def __init__(self, c: float, seed: int, max_iter: int = MAX_ITER) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        self.c = c
        self._vec = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, sublinear_tf=True, min_df=1)
        self._clf = LogisticRegression(C=c, max_iter=max_iter, tol=self.TOL, random_state=seed)
        self._max_iter = max_iter
        self._classes: list[str] = []
        self.log: dict[str, object] = {}

    failed = False  # LR has no failed-run notion; kept for a uniform trainer interface

    def fit(self, texts: list[str], labels: list[str], dev_texts=None, dev_labels=None,
            label_ids=None, dev_candidates=None) -> "TfidfLogReg":
        x = self._vec.fit_transform(texts)
        self._clf.fit(x, labels)
        self._classes = [str(c) for c in self._clf.classes_]
        n_iter = int(max(self._clf.n_iter_))
        self.log = {"c": self.c, "n_iter": n_iter, "converged": n_iter < self._max_iter, "tol": self.TOL}
        if n_iter >= self._max_iter:
            raise RuntimeError(f"logistic regression did not converge in {self._max_iter} iterations (C={self.c})")
        return self

    def release(self) -> None:
        """Uniform trainer interface (frees accelerator memory for DeBERTa; nothing to do here)."""

    def predict_proba(self, texts: list[str], label_ids: list[str]) -> np.ndarray:
        """(n, len(label_ids)) aligned to label_ids; classes unseen in training get probability 0."""
        p = self._clf.predict_proba(self._vec.transform(texts))
        out = np.zeros((len(texts), len(label_ids)))
        col = {lid: i for i, lid in enumerate(label_ids)}
        for j, c in enumerate(self._classes):
            out[:, col[c]] = p[:, j]
        return out

    def time_single(self, texts: list[str]) -> list[float]:
        """Per-example inference latency in ms (spec 9; v1.1): vectorize + predict_proba for one text (CPU)."""
        import time

        out = []
        for text in texts:
            start = time.perf_counter()
            self._clf.predict_proba(self._vec.transform([text]))
            out.append((time.perf_counter() - start) * 1000)
        return out

    @staticmethod
    def timing_info() -> dict[str, object]:
        import platform

        return {"device": "cpu", "batch_size": 1, "processor": platform.processor()}

    @staticmethod
    def describe() -> dict[str, object]:
        import sklearn

        return {"name": "tfidf-lr", "sklearn": sklearn.__version__, "ngram_range": [1, 2],
                "sublinear_tf": True, "c_grid": list(TfidfLogReg.C_GRID), "selection": "dev macro-F1 (candidate-restricted)",
                "solver": "lbfgs", "tol": TfidfLogReg.TOL, "max_iter": TfidfLogReg.MAX_ITER}
