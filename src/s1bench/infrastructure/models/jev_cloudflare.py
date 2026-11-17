"""Jev (TypeSafe) via Cloudflare Workers AI (spec sec 2, 7.2, 8, 13).

Request: {"model": "typesafe/jev", "input": {"state": text, "questions": {"label": {"type": "choice",
          "instructions": task question, "criteria": {label name: description, ...}}}}}
Option keys are the human-readable label NAMES in candidate order (Jev reads names semantically);
the returned per-option `probabilities` (sum ~1) are the readout. `confidence` is not a probability.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from s1bench.domain.candidates import CandidateSet
from s1bench.domain.examples import Example, LabelSet
from s1bench.domain.predictions import (
    STATUS_BLOCKED,
    STATUS_FAILED,
    STATUS_OK,
    Prediction,
    argmax_label,
)

_BLOCK_HINTS = ("safety", "moderation", "content filter", "content policy", "blocked", "refus", "flagged")
# v1.1.1 (audit 2026-10-08): transport errors are never safety blocks. Windows reports a refused connection as
# "...the target machine actively refused it", which matched "refus" and turned an outage into `blocked`.
_TRANSPORT_ERRORS = (ConnectionError, TimeoutError, OSError)


class JevCloudflareModel:
    name = "jev"

    def __init__(self, account_id: str, api_token: str, model: str = "typesafe/jev",
                 timeout_s: float = 30.0, retries: int = 4) -> None:
        self._url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run"
        self._token = api_token
        self._model = model
        self._timeout = timeout_s
        self._retries = retries

    def describe(self) -> dict[str, object]:
        return {"name": self.name, "provider": "Cloudflare Workers AI", "model": self._model,
                "readout": "native per-option probabilities, option keys = label names",
                "retries": self._retries, "timeout_s": self._timeout}

    def _payload(self, example: Example, candidates: CandidateSet, labels: LabelSet) -> dict[str, object]:
        by_id = {lab.id: lab for lab in labels.labels}
        criteria = {by_id[cid].name: by_id[cid].description for cid in candidates.candidates}
        return {"model": self._model, "input": {"state": example.text, "questions": {
            "label": {"type": "choice", "instructions": labels.task_question, "criteria": criteria}}}}

    def _post(self, payload: dict[str, object]) -> dict[str, object]:
        req = urllib.request.Request(
            self._url, data=json.dumps(payload).encode("utf-8"), method="POST",
            headers={"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:  # keep the body: Cloudflare puts the reason there
            body = e.read().decode("utf-8", "replace")[:500]
            raise RuntimeError(f"HTTP {e.code}: {body}") from None

    def predict(self, example: Example, candidates: CandidateSet, labels: LabelSet) -> Prediction:
        payload = self._payload(example, candidates, labels)
        id_to_name = {lab.id: lab.name for lab in labels.labels}
        base = dict(example_id=example.id, model=self.name, dataset=example.dataset, split=example.split,
                    candidates=candidates.candidates, gold=example.label)
        last_error = ""
        errors: list[str] = []          # every failed attempt's reason (kept in raw / error for audit)
        parse_failures = 0
        for attempt in range(self._retries + 1):
            start = time.perf_counter()
            try:
                body = self._post(payload)
                latency_ms = (time.perf_counter() - start) * 1000
                if not body.get("success", False):
                    raise RuntimeError(json.dumps(body.get("errors"))[:500])
                envelope = body["result"]
                if envelope.get("state") == "Failed":
                    raise RuntimeError(str(envelope.get("error", "Jev call failed"))[:500])
                result = envelope.get("result", envelope)
                answer = result["answers"]["label"]
                probs_by_name = answer["probabilities"]
                probabilities = tuple(float(probs_by_name[id_to_name[cid]]) for cid in candidates.candidates)
                predicted = argmax_label(candidates.candidates, probabilities)  # spec sec 8: argmax
                return Prediction(
                    **base, probabilities=probabilities, predicted=predicted, status=STATUS_OK,
                    latency_ms=round(latency_ms, 2),
                    input_tokens=(result.get("usage") or {}).get("input_tokens"),
                    model_version=result.get("model") or envelope.get("model"),
                    raw={"choice": answer.get("choice"), "confidence": answer.get("confidence"),
                         "probabilities": probs_by_name, "attempts": attempt + 1,
                         "attempt_errors": [x[:200] for x in errors]},
                )
            except Exception as e:  # classify, then retry
                last_error = f"{type(e).__name__}: {e}"
                errors.append(last_error)
                transport = isinstance(e, _TRANSPORT_ERRORS) or isinstance(getattr(e, "reason", None), OSError)
                if not transport and any(h in last_error.lower() for h in _BLOCK_HINTS):
                    if attempt >= 1:  # one retry for blocks (spec sec 13), then give up
                        return Prediction(**base, probabilities=(), predicted=None, status=STATUS_BLOCKED,
                                          error=" | ".join(x[:150] for x in errors)[:500])
                elif isinstance(e, (KeyError, TypeError, ValueError)):
                    # unexpected response shape: deterministic most of the time; retry once only (each call is paid)
                    parse_failures += 1
                    if parse_failures > 1:
                        break
                elif attempt == self._retries:
                    break
                rate_limited = "rate" in last_error.lower() or "429" in last_error
                time.sleep((6.0 if rate_limited else 1.5) * 2 ** attempt)   # exponential backoff (spec sec 13)
        return Prediction(**base, probabilities=(), predicted=None, status=STATUS_FAILED,
                          error=" | ".join(x[:150] for x in errors)[:500])
