"""Use case: Exp 1 metric table from saved predictions (spec sec 9, 10.1, 13). No new inference.

Status handling (spec sec 13):
- failed rows are excluded (a file with failed > 1% must be rerun; reported in `status`);
- blocked rows (refusal / safety block) count as **wrong + abstain** for the decision metrics (Accuracy,
  Macro-F1, MCC, AURC, Acc@coverage): they get a separate abstain prediction with confidence 0, so they are
  rejected first in the risk-coverage curve and are errors at full coverage. Macro-F1 averages over the
  dataset's classes only (the abstain pseudo-class is not averaged in);
- probability metrics (ECE, Brier, NLL) use ok rows only (a blocked row has no probabilities);
- if blocked > 1%, the decision metrics are also reported with blocked rows excluded (sensitivity).
Probabilities are renormalised over the candidates (Jev returns 2-decimal values that sum to 0.99-1.01).
"""

from __future__ import annotations

from collections.abc import Sequence

from s1bench.domain import metrics as mt
from s1bench.domain.predictions import STATUS_BLOCKED, STATUS_FAILED, STATUS_OK, Prediction

ABSTAIN = "__abstain__"
BLOCKED_SENSITIVITY_LIMIT = 0.01


def _scored(p: Prediction) -> mt.Scored:
    total = sum(p.probabilities)
    probs = tuple(x / total for x in p.probabilities) if total > 0 else p.probabilities
    return mt.Scored(tuple(p.candidates), probs, p.gold)


def macro_f1_dataset_classes(rows: Sequence[mt.Scored]) -> float:
    """Macro-F1 over the classes present in gold or (non-abstain) predictions; an abstained example is a false
    negative for its gold class and a false positive for no class."""
    classes = sorted(({r.gold for r in rows} | {r.predicted for r in rows}) - {ABSTAIN})
    f1s = []
    for c in classes:
        tp = sum(1 for r in rows if r.predicted == c and r.gold == c)
        fp = sum(1 for r in rows if r.predicted == c and r.gold != c)
        fn = sum(1 for r in rows if r.predicted != c and r.gold == c)
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2 * tp / denom)
    return sum(f1s) / len(f1s)


def _decision_metrics(rows: Sequence[mt.Scored]) -> dict[str, float]:
    return {"accuracy": mt.accuracy(rows), "macro_f1": macro_f1_dataset_classes(rows), "mcc": mt.mcc(rows),
            "aurc": mt.aurc(rows), "acc_at_80": mt.accuracy_at_coverage(rows, 0.8),
            "acc_at_50": mt.accuracy_at_coverage(rows, 0.5)}


def exp1_row(predictions: Sequence[Prediction]) -> dict[str, object]:
    status: dict[str, int] = {}
    for p in predictions:
        status[p.status] = status.get(p.status, 0) + 1
    ok = [_scored(p) for p in predictions if p.status == STATUS_OK and p.probabilities]
    blocked = [mt.Scored((ABSTAIN,), (0.0,), p.gold) for p in predictions if p.status == STATUS_BLOCKED]
    n = len(predictions)
    row: dict[str, object] = {
        "n": n, "status": status, "n_scored": len(ok) + len(blocked),
        "failed_share": status.get(STATUS_FAILED, 0) / n if n else 0.0,
        "blocked_share": len(blocked) / n if n else 0.0,
    }
    if not ok and not blocked:
        return row
    row.update(_decision_metrics(ok + blocked))
    if ok:
        row.update({"ece": mt.ece(ok), "brier": mt.brier(ok), "nll": mt.nll(ok)})
    if row["blocked_share"] > BLOCKED_SENSITIVITY_LIMIT and ok:
        row["sensitivity_blocked_excluded"] = _decision_metrics(ok)
    return row
