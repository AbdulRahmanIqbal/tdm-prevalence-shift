"""Exp 4 condition (b): QLoRA task fine-tune (spec 10.3, 10.4; v1.2). Pure helpers, the use case with a fake model,
and the model mechanics on a tiny random Qwen3.5 (CPU, offline): LoRA targets, batched readout parity, adapter
save/reload and checkpoint restore. The 4B model itself is checked by the dev pilot on a GPU."""

from __future__ import annotations

import re

import pytest
import torch
import yaml

from s1bench.application.train_letter_model import TrainLetterModel, majority_predictions, restarts_used
from s1bench.domain.candidates import CandidateSet
from s1bench.domain.examples import Example, Label, LabelSet
from s1bench.domain.predictions import STATUS_FAILED, Prediction
from s1bench.infrastructure.models.qlora_letter import (
    adapter_key_check,
    candidate_logits,
    eval_steps,
    gold_position,
    label_free,
    new_adapter_dir,
    planned_steps,
    right_pad,
    target_pattern,
)

LABELS = LabelSet("toy", "q?", (Label("a", "alpha", "d"), Label("b", "beta", "d"), Label("c", "gamma", "d")))
CFG = yaml.safe_load(open("configs/experiments/qlora.yaml", encoding="utf-8"))
TARGETS = CFG["lora"]["target_modules"]


# ---------------------------------------------------------------------------------------------- helpers
@pytest.mark.parametrize("n, expected", [(100, (20, 500)), (500, (20, 640)), (2500, (5, 785))])
def test_planned_steps_follow_the_deberta_rule(n, expected):
    assert planned_steps(n, 16, CFG["schedule"]) == expected


def test_eval_steps_every_n_and_last():
    assert eval_steps(500, 100) == [100, 200, 300, 400, 500]
    assert eval_steps(785, 100)[-2:] == [700, 785]
    assert eval_steps(4, 2) == [2, 4]


def test_frozen_config_matches_protocol():
    assert CFG["budgets"] == [100, 500, 2500] and CFG["seeds"] == [1001, 1002, 1003]
    assert (CFG["lora"]["r"], CFG["lora"]["alpha"], CFG["lora"]["dropout"], CFG["optimizer"]["lr"]) == (16, 32, 0.05, 1e-4)
    assert CFG["optimizer"]["micro_batch"] * CFG["optimizer"]["grad_accumulation"] == 16
    assert CFG["quantization"]["quant_type"] == "nf4" and CFG["predict_base"] == "bfloat16"
    assert CFG["schedule"]["eval_every"] == 100 and CFG["schedule"]["min_steps"] == 500


def test_gold_position_checks_the_candidate_file():
    ex = Example("toy:train:0", "toy", "train", "t", "b", 1)
    assert gold_position((ex, CandidateSet(ex.id, ("c", "b", "a"), 1))) == 1
    with pytest.raises(ValueError):
        gold_position((ex, CandidateSet(ex.id, ("c", "b", "a"), 0)))


def test_label_free_view_hides_gold():
    ex, cs = label_free((Example("x", "toy", "test", "t", "b", 1), CandidateSet("x", ("a", "b"), 1)))
    assert (ex.label, ex.label_index, cs.gold_position, ex.text, cs.candidates) == ("", -1, -1, "t", ("a", "b"))


def test_right_pad_keeps_last_real_token():
    ids, mask, last = right_pad([[5, 6, 7], [8]], pad_id=0)
    assert ids.tolist() == [[5, 6, 7], [8, 0, 0]] and mask.tolist() == [[1, 1, 1], [1, 0, 0]]
    assert last.tolist() == [2, 0]


def test_candidate_logits_match_letter_readout_and_mask_unused_letters():
    torch.manual_seed(0)
    h, w = torch.randn(2, 8), torch.randn(30, 8)
    letter_ids = [3, 7, 11, 13]
    out = candidate_logits(h, w, letter_ids, torch.tensor([4, 2]))
    ref = h.float() @ w[letter_ids].float().T            # HfLetterScoringModel._letters_from_hidden
    assert torch.allclose(out[0], ref[0])
    assert torch.allclose(out[1, :2], ref[1, :2]) and torch.isinf(out[1, 2:]).all()
    assert torch.isfinite(torch.nn.functional.cross_entropy(out, torch.tensor([3, 1])))


def test_adapter_key_check():
    keys = {"m.l0.q.lora_A.weight", "m.l0.q.lora_B.weight"}
    adapter_key_check(keys, {k.replace(".weight", ".default.weight") for k in keys}, 1)
    with pytest.raises(RuntimeError):
        adapter_key_check(keys, {"m.l0.q.lora_A.default.weight"}, 1)        # lora_B not applied
    with pytest.raises(RuntimeError):
        adapter_key_check(keys, {k.replace(".weight", ".default.weight") for k in keys}, 2)   # module count


def test_adapter_dirs_are_per_dataset_and_never_overwritten(tmp_path):
    name = "qwen3.5-4b-qlora@b100s1001"
    a = new_adapter_dir(tmp_path, "clinc150", name, 1001)
    b = new_adapter_dir(tmp_path, "fpb", name, 1001)                  # same model name, other dataset
    assert a != b and a.parent.name == "clinc150" and b.name == f"{name}__seed1001"
    (a / "adapter_model.safetensors").write_bytes(b"x")
    with pytest.raises(FileExistsError):
        new_adapter_dir(tmp_path, "clinc150", name, 1001)
    assert new_adapter_dir(tmp_path, "clinc150", name, 1006) != a     # reserve seed: own folder


# ---------------------------------------------------------------------------------------------- tiny Qwen3.5
@pytest.fixture(scope="module")
def tiny():
    import transformers
    from transformers import Qwen3_5Config

    cfg = Qwen3_5Config()
    tc = cfg.text_config
    for k, v in dict(hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
                     num_key_value_heads=2, head_dim=16, linear_num_value_heads=4, linear_num_key_heads=2,
                     linear_key_head_dim=16, linear_value_head_dim=16, vocab_size=300).items():
        setattr(tc, k, v)
    tc.layer_types = ["linear_attention"] * 3 + ["full_attention"]
    vc = cfg.vision_config
    for k, v in dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64).items():
        if hasattr(vc, k):
            setattr(vc, k, v)
    torch.manual_seed(0)
    model = transformers.AutoModelForImageTextToText.from_config(cfg).float().eval()
    return cfg, model


def _single(model, seq, letters, embeds=False):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    ids = torch.tensor([seq])
    kw = {"inputs_embeds": base.get_input_embeddings()(ids)} if embeds else {"input_ids": ids}
    h = base.model(**kw, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state[0, -1]
    return h.float() @ base.get_output_embeddings().weight[letters].float().T


def test_lora_targets_are_exactly_the_language_model_projections(tiny):
    _, model = tiny
    pat = target_pattern(TARGETS)
    linear = {n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
    hit = {n for n in linear if re.fullmatch(pat, n)}
    assert len(hit) == 3 * 5 + 1 * 4 + 4 * 3            # linear_attn (5), self_attn (4), mlp (3) per layer
    assert not any(n.startswith("model.visual") for n in hit) and "lm_head" not in hit
    assert all(n.startswith("model.language_model.layers.") for n in hit)
    assert {n.rsplit(".", 1)[1] for n in hit} == set(TARGETS)


def test_batched_readout_equals_unbatched_zero_shot_readout(tiny):
    _, model = tiny
    letters = list(range(10, 34))
    g = torch.Generator().manual_seed(1)
    seqs = [torch.randint(5, 300, (n,), generator=g).tolist() for n in (70, 13, 129, 64)]
    with torch.no_grad():
        ids, mask, last = right_pad(seqs, 0)
        h = model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state[torch.arange(4), last]
        batched = candidate_logits(h, model.get_output_embeddings().weight, letters, torch.tensor([24, 24, 24, 24]))
        ref = torch.stack([_single(model, s, letters) for s in seqs])
        ref_embeds = torch.stack([_single(model, s, letters, embeds=True) for s in seqs])   # cpu-embed route
    assert float((batched - ref).abs().max()) < 1e-5
    assert torch.equal(ref, ref_embeds) or float((ref - ref_embeds).abs().max()) < 1e-6


def test_adapter_save_reload_and_best_state_restore(tiny, tmp_path):
    import copy

    import transformers
    from peft import LoraConfig, PeftModel, get_peft_model
    from safetensors import safe_open

    cfg, model = tiny
    base_state = copy.deepcopy(model.state_dict())
    pm = get_peft_model(copy.deepcopy(model), LoraConfig(r=4, lora_alpha=8, lora_dropout=0.05,
                                                         target_modules=target_pattern(TARGETS)))
    trainable = {n: p for n, p in pm.named_parameters() if p.requires_grad}
    n_modules = len({n.rsplit(".lora_", 1)[0] for n in trainable})
    with torch.no_grad():
        for n, p in trainable.items():
            if "lora_B" in n:
                p.normal_(0, 0.05)
    best = {n: p.detach().clone() for n, p in trainable.items()}           # as fit() keeps the best state
    with torch.no_grad():
        for p in trainable.values():
            p.add_(1.0)                                                     # training moves on
    res = pm.load_state_dict(best, strict=False)
    assert not res.unexpected_keys and not set(best) & set(res.missing_keys)
    assert all(torch.equal(dict(pm.named_parameters())[n], v) for n, v in best.items())
    pm.eval()
    letters = list(range(10, 34))
    seq = list(range(20, 80))
    with torch.no_grad():
        a = _single(pm, seq, letters)
    pm.save_pretrained(str(tmp_path), save_embedding_layers=False)
    with safe_open(str(tmp_path / "adapter_model.safetensors"), "pt") as fh:
        file_keys = set(fh.keys())
    fresh = transformers.AutoModelForImageTextToText.from_config(cfg).float()
    fresh.load_state_dict(base_state, strict=False)
    reloaded = PeftModel.from_pretrained(fresh, str(tmp_path)).eval()
    adapter_key_check(file_keys, {k for k, _ in reloaded.named_parameters() if "lora_" in k}, n_modules)
    with torch.no_grad():
        b = _single(reloaded, seq, letters)
    assert float((a - b).abs().max()) == 0.0


# ---------------------------------------------------------------------------------------------- use case
def _ex(split, n):
    return [Example(f"toy:{split}:{i}", "toy", split, f"text {i}", "abc"[i % 3] if i % 5 else "a", 0) for i in range(n)]


class Reader:
    def __init__(self):
        self.data = {s: _ex(s, n) for s, n in (("train", 60), ("dev", 30), ("test", 30))}

    def read_split(self, dataset, split):
        return self.data[split]


class Cands:
    def read_split(self, dataset, split):
        return {e.id: CandidateSet(e.id, ("a", "b", "c"), "abc".index(e.label)) for e in Reader().data[split]}


class Labels:
    def load(self, dataset):
        return LABELS


class Store:
    def __init__(self):
        self.written = {}

    def write(self, model, dataset, split, preds):
        self.written[(model, split)] = preds


class FakeLetterModel:
    """Deployed model: right on every item unless its seed is in FAIL (then always wrong); BROKEN seeds return a
    failed prediction. Majority dev accuracy is fixed at 0.4; MAJORITY overrides it (unrounded value)."""
    FAIL: set[int] = set()
    BROKEN: set[int] = set()
    THIRD: set[int] = set()            # right on exactly one third of the items
    MAJORITY: float | None = None
    seen: list[tuple[int, str, int]] = []
    released: list[int] = []

    def __init__(self, seed, name):
        self.seed, self.name, self.failed = seed, name, False
        self.majority_dev_accuracy = 0.4 if self.MAJORITY is None else self.MAJORITY
        self.log = {"seed": seed, "majority_dev_accuracy": round(self.majority_dev_accuracy, 4)}

    def fit(self, train, dev, labels):
        assert all(isinstance(cs, CandidateSet) for _, cs in train + dev)
        FakeLetterModel.seen.append((self.seed, self.name, len(train)))
        return self

    def predict(self, items, labels):
        if self.seed in self.BROKEN:
            raise RuntimeError("1 of n predictions failed")
        out = []
        for e, cs in items:
            wrong = self.seed in self.FAIL or (self.seed in self.THIRD and int(e.id.rsplit(":", 1)[1]) % 3)
            pred = next(c for c in cs.candidates if c != e.label) if wrong else e.label
            probs = tuple(1.0 if c == pred else 0.0 for c in cs.candidates)
            out.append(Prediction(e.id, self.name, e.dataset, e.split, cs.candidates, probs, pred, e.label))
        return out

    def release(self):
        FakeLetterModel.released.append(self.seed)


def _uc(store=None, used=None):
    return TrainLetterModel(FakeLetterModel, Reader(), Cands(), Labels(), store or Store(),
                            reserve_seeds=(1006, 1007, 1008), reserve_used=used)


def test_failed_rule_uses_the_deployed_model_and_reserve_pool_is_shared():
    FakeLetterModel.FAIL, FakeLetterModel.BROKEN = {1001, 1002, 1006, 1007}, set()
    FakeLetterModel.seen, FakeLetterModel.released = [], []
    uc = _uc()
    r1 = uc.execute("q@b30s1001", "toy", 30, 1001)     # 1001 -> 1006 -> 1007 fail, 1008 ok
    r2 = uc.execute("q@b30s1002", "toy", 30, 1002)     # pool exhausted
    assert [x["restart_seed"] for x in r1["restarts"]] == [1006, 1007, 1008] and not r1["still_failed"]
    assert r1["trained_seed"] == 1008 and r1["train_log"]["dev_accuracy_deployed"] == 1.0
    assert r2["restarts"] == [] and r2["still_failed"] and r2["scored_as"] == "majority_class"
    assert {name for seed, name, _ in FakeLetterModel.seen if seed in (1006, 1007, 1008)} == {"q@b30s1001"}
    assert {1001, 1006, 1007} <= set(FakeLetterModel.released)


def test_failed_rule_compares_against_the_unrounded_majority():
    # accuracy 1/3 equals the majority 1/3 (which rounds down to 0.3333): spec 10.3 says equality = failed
    FakeLetterModel.FAIL, FakeLetterModel.BROKEN, FakeLetterModel.THIRD = set(), set(), {1001}
    FakeLetterModel.MAJORITY = 1 / 3
    try:
        r = _uc().execute("q@b30s1001", "toy", 30, 1001)
    finally:
        FakeLetterModel.THIRD, FakeLetterModel.MAJORITY = set(), None
    assert r["restarts"][0]["failed_seed"] == 1001
    assert r["restarts"][0]["failed_run"]["dev_accuracy_deployed"] == 0.3333


def test_reserve_pool_continues_across_calls():
    FakeLetterModel.FAIL, FakeLetterModel.BROKEN = {1002}, set()
    old = [{"model": "q@b30s1001", "dataset": "toy", "budget": 30, "restarts": [{}, {}, {}]}]
    r = _uc(used=restarts_used(old, "q")).execute("q@b30s1002", "toy", 30, 1002)
    assert r["restarts"] == [] and r["still_failed"]


def test_failed_prediction_raises_and_writes_nothing():
    FakeLetterModel.FAIL, FakeLetterModel.BROKEN = set(), {1001}
    store = Store()
    with pytest.raises(RuntimeError):
        _uc(store).execute("q@b30s1001", "toy", 30, 1001)
    assert store.written == {}


def test_predictions_written_and_test_skipped_on_request():
    FakeLetterModel.FAIL, FakeLetterModel.BROKEN = set(), set()
    store = Store()
    r = _uc(store).execute("q@b30s1001", "toy", 30, 1001, predict_test=False)
    assert ("q@b30s1001", "dev") in store.written and ("q@b30s1001", "test") not in store.written
    assert r["test_macro_f1"] is None and r["n_train"] == 30 and r["dev_macro_f1"] == 1.0


def test_budget_must_be_below_train_size():
    with pytest.raises(ValueError):
        _uc().execute("q", "toy", 60, 1001)


def test_majority_predictions_follow_train_frequencies_within_candidates():
    items = [(Example("x", "toy", "test", "t", "c", 2), CandidateSet("x", ("b", "c"), 1)),
             (Example("y", "toy", "test", "t", "c", 2), CandidateSet("y", ("c",), 0))]
    preds = majority_predictions("m", items, ["a", "a", "b", "b", "b"], LABELS)
    assert preds[0].probabilities == (1.0, 0.0) and preds[0].predicted == "b"
    assert preds[1].probabilities == (1.0,) and preds[1].predicted == "c"
    assert all(abs(sum(p.probabilities) - 1) < 1e-12 for p in preds) and STATUS_FAILED != preds[0].status
