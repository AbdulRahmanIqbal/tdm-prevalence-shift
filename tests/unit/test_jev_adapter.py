from __future__ import annotations

from s1bench.domain.candidates import CandidateSet
from s1bench.domain.examples import Example, Label, LabelSet
from s1bench.infrastructure.models.jev_cloudflare import JevCloudflareModel

LABELS = LabelSet("toy", "Which?", (Label("0", "negative", "bad"), Label("1", "neutral", "meh"),
                                     Label("2", "positive", "good")))
EX = Example("toy:test:test000001", "toy", "test", "Profit rose.", "2", 2)
CS = CandidateSet(EX.id, ("1", "2", "0"), 1)


def _model(response=None, error=None, calls=None):
    m = JevCloudflareModel("acct", "token", retries=1)

    def fake_post(payload):
        if calls is not None:
            calls.append(payload)
        if error:
            raise RuntimeError(error)
        return response

    m._post = fake_post  # type: ignore[method-assign]
    return m


def test_payload_uses_label_names_in_candidate_order_and_parses_probabilities():
    calls = []
    resp = {"success": True, "result": {"result": {"model": "jev-1.13.0", "usage": {"input_tokens": 42},
            "answers": {"label": {"choice": "positive", "confidence": 0.5,
                                  "probabilities": {"neutral": 0.2, "positive": 0.7, "negative": 0.1}}}}}}
    p = _model(resp, calls=calls).predict(EX, CS, LABELS)
    assert list(calls[0]["input"]["questions"]["label"]["criteria"]) == ["neutral", "positive", "negative"]
    assert p.probabilities == (0.2, 0.7, 0.1) and p.predicted == "2" and p.correct
    assert p.model_version == "jev-1.13.0" and p.input_tokens == 42


def test_safety_block_is_reported_as_blocked_not_failed():
    p = _model(error="HTTP 400: request blocked by content filter").predict(EX, CS, LABELS)
    assert p.status == "blocked" and p.predicted is None
