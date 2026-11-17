"""Revision protocol (v1.3, G5 + G6): CLI guards of the prompt-sensitivity and bf16-offload options (no model is
loaded: every case must stop before the weights are touched)."""

from __future__ import annotations

import pytest

from s1bench.cli.main import main

PCFG = "configs/experiments/prompt_sensitivity.yaml"


@pytest.mark.parametrize("argv", [
    ["predict", "--model", "jev", "--dataset", "fpb", "--prompt-config", PCFG, "--prompt-variant", "plain"],
    ["predict", "--model", "qwen3.5-4b", "--dataset", "fpb", "--prompt-config", PCFG],
    ["predict", "--model", "qwen3.5-4b", "--dataset", "fpb", "--prompt-variant", "plain"],
    ["predict", "--model", "qwen3.5-4b", "--dataset", "fpb", "--prompt-config", PCFG, "--prompt-variant", "nope"],
])
def test_prompt_variant_guards(argv):
    assert main(argv) == 2


def test_offload_rejects_quantization_and_cpu():
    from s1bench.infrastructure.models.hf_letter_scorer import HfLetterScoringModel

    for kw in ({"quantization": "8bit", "device": "cuda"}, {"device": "cpu"}):
        with pytest.raises(ValueError, match="offload"):
            HfLetterScoringModel("m", "repo", "rev", {}, "nimble", load_model=False, offload_gpu_gib=9.0, **kw)


def test_prompt_config_variants_use_known_formats():
    import yaml

    cfg = yaml.safe_load(open(PCFG, encoding="utf-8"))
    assert {v["format"] for v in cfg["variants"].values()} == {"plain", "plain_names"}
    assert set(cfg["models"]) == {"qwen3.5-4b", "tev1-4b", "qwen3.5-9b", "nimble-9b"}
