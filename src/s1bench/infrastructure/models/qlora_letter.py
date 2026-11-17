"""QLoRA task fine-tune of a local decoder, read out by candidate letters (Exp 4 condition b; spec 10.3, 10.4; v1.2).

Prompt, assistant prefix and readout are those of the zero-shot runs (HfLetterScoringModel, unchanged): the letter
logits are the fp32 product of the final hidden state at the answer position with the lm_head rows of the candidate
letters, and the probabilities are their softmax (T = 1). Training minimises the cross-entropy over the same K
candidate letters. The base model is loaded in 4-bit NF4 with double quantization (QLoRA, Dettmers et al. 2023); LoRA
adapters (settings in configs/experiments/qlora.yaml) are trained on the language model only.

Selection follows DeBERTa (spec 10.3): every `eval_every` optimizer steps the candidate-restricted dev macro-F1 of the
4-bit training model is computed (batched, selection only) and the best adapter state is kept on the CPU. The kept
adapter is saved, verified tensor by tensor, and loaded onto the base model in the precision of the zero-shot runs
(`predict_base`, bf16); this deployed model gives every reported prediction (unbatched zero-shot readout, label-free
inputs) and the failed-run decision (decided by the use case from its dev predictions).
"""

from __future__ import annotations

import gc
import hashlib
import math
import random
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np

from s1bench.domain.candidates import CandidateSet
from s1bench.domain.examples import Example, LabelSet
from s1bench.domain.predictions import STATUS_OK, Prediction
from s1bench.infrastructure.models.deberta_classifier import candidate_majority_accuracy
from s1bench.infrastructure.models.hf_letter_scorer import LETTERS, HfLetterScoringModel

Item = tuple[Example, CandidateSet]
_ENCODINGS: dict[tuple, list[int]] = {}          # prompt token ids, cached across seeds and restarts


# ---------------------------------------------------------------------------------------------- pure helpers
def planned_steps(n_train: int, effective_batch: int, schedule: dict) -> tuple[int, int]:
    """(epochs, optimizer steps): DeBERTa rule of spec 10.3 (20 epochs up to `small_budget` examples, else 5;
    at least `min_steps` optimizer steps). Batches are drawn continuously over reshuffled epochs."""
    if schedule.get("max_steps") is not None:    # revision protocol (v1.3): budgets.realistic_steps
        from s1bench.domain.budgets import realistic_steps

        epochs = schedule["epochs_small"]
        return epochs, realistic_steps(n_train, effective_batch, epochs, schedule["min_steps"], schedule["max_steps"])
    epochs = schedule["epochs_small"] if n_train <= schedule["small_budget"] else schedule["epochs_large"]
    return epochs, max(schedule["min_steps"], epochs * math.ceil(n_train / effective_batch))


def eval_steps(total: int, every: int) -> list[int]:
    """Optimizer steps after which the dev set is evaluated: every `every` steps and the last step."""
    return sorted(set(range(every, total + 1, every)) | {total})


def target_pattern(modules: Sequence[str]) -> str:
    """PEFT target regex (re.fullmatch): the listed projections inside the language-model layers only."""
    return r"model\.language_model\.layers\.\d+\.\w+\.(?:" + "|".join(modules) + ")"


def gold_position(item: Item) -> int:
    example, cands = item
    pos = cands.candidates.index(example.label)
    if pos != cands.gold_position:
        raise ValueError(f"{example.id}: gold position mismatch in the candidate file")
    return pos


def right_pad(seqs: Sequence[Sequence[int]], pad_id: int):
    """Right-padded ids, attention mask and the index of each sequence's last real token. With right padding the
    causal (and recurrent) layers never see a pad token before the answer position."""
    import torch

    n = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), n), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(list(s), dtype=torch.long)
        mask[i, : len(s)] = 1
    return ids, mask, torch.tensor([len(s) - 1 for s in seqs], dtype=torch.long)


def candidate_logits(h_last, head_weight, letter_ids: Sequence[int], ks):
    """fp32 letter logits (B, len(letter_ids)); positions beyond each example's K candidates are -inf."""
    import torch

    logits = h_last.float() @ head_weight[list(letter_ids)].float().T
    col = torch.arange(len(letter_ids), device=logits.device)
    return logits.masked_fill(col[None, :] >= ks[:, None], float("-inf"))


def label_free(item: Item) -> Item:
    """The view every model receives (as RunPredictions): no gold label, no gold position."""
    ex, cs = item
    return replace(ex, label="", label_index=-1), replace(cs, gold_position=-1)


def new_adapter_dir(root: Path, dataset: str, name: str, seed: int) -> Path:
    """<root>/<dataset>/<name>__seed<seed>, created empty. A saved adapter is never overwritten."""
    path = root / dataset / f"{name}__seed{seed}"
    if (path / "adapter_model.safetensors").exists():
        raise FileExistsError(f"{path} already holds an adapter; use a new run id")
    path.mkdir(parents=True, exist_ok=True)
    return path


def adapter_key_check(file_keys: set[str], model_lora_keys: set[str], n_modules: int) -> None:
    """Every tensor of the saved adapter must match a LoRA parameter of the loaded model and vice versa, with one
    lora_A and one lora_B per trained module (PEFT only warns on missing keys)."""
    model_keys = {k.replace(".default", "") for k in model_lora_keys}
    missing, unexpected = sorted(file_keys - model_keys), sorted(model_keys - file_keys)
    if missing or unexpected or len(file_keys) != 2 * n_modules:
        raise RuntimeError(f"adapter not applied: {len(missing)} file tensors unmatched, {len(unexpected)} model LoRA "
                           f"tensors without weights, {len(file_keys)} tensors for {n_modules} modules "
                           f"(e.g. {(missing or unexpected)[:2]})")


def _bnb_config(q: dict, dtype):
    import transformers

    return transformers.BitsAndBytesConfig(load_in_4bit=q["load_in_4bit"], bnb_4bit_quant_type=q["quant_type"],
                                           bnb_4bit_use_double_quant=q["double_quant"], bnb_4bit_compute_dtype=dtype)


def _load_base(repo: str, revision: str, dtype, quantization_config=None, device_map=None):
    """Same model classes, in the same order, as HfLetterScoringModel._load_model."""
    import transformers

    kwargs: dict[str, object] = {"dtype": dtype}
    if quantization_config is not None:
        kwargs.update(quantization_config=quantization_config, device_map=device_map)
    errors = []
    for cls_name in ("AutoModelForImageTextToText", "AutoModelForCausalLM"):
        try:
            return getattr(transformers, cls_name).from_pretrained(repo, revision=revision, **kwargs)
        except (ValueError, KeyError, TypeError) as e:
            errors.append(f"{cls_name}: {e}")
    raise RuntimeError("could not load base model:\n" + "\n".join(errors))


def _live_cuda_tensors(top: int = 5) -> list[tuple[str, int]]:
    """Diagnostics: shape/dtype and MiB of the largest CUDA tensors still reachable by the garbage collector."""
    import torch

    found = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                found.append((f"{tuple(obj.shape)} {obj.dtype}", obj.numel() * obj.element_size() // 2**20))
        except Exception:  # noqa: BLE001 - objects that fail introspection are skipped
            continue
    return sorted(found, key=lambda x: -x[1])[:top]


def _free_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ---------------------------------------------------------------------------------------------- classifier
class QloraLetterClassifier:
    def __init__(self, seed: int, name: str, model_cfg: dict, prompt_format: dict, cfg: dict,
                 device: str = "cuda", smoke: dict | None = None, adapter_root: Path | None = None) -> None:
        self.seed, self.name = seed, name
        self._mcfg, self._format, self._cfg = model_cfg, prompt_format, cfg
        self._device = device if ":" in device else f"{device}:0"
        self._sched = {**cfg["schedule"], **(smoke or {})}
        self._smoke = bool(smoke)
        self._adapter_root = adapter_root
        self.failed = False
        self.majority_dev_accuracy: float | None = None   # set by fit()
        self.log: dict[str, object] = {}
        self._model = None
        self._scorer: HfLetterScoringModel | None = None
        self._adapter_dir: Path | None = None
        self._tmp_adapter = False

    # ------------------------------------------------------------------ setup
    def hparams(self) -> dict[str, object]:
        o = self._cfg["optimizer"]
        return {"base_repo": self._mcfg["repo"], "base_revision": self._mcfg["revision"],
                "prompt_format": self._mcfg["prompt_format"], "quantization": self._cfg["quantization"],
                "lora": self._cfg["lora"], "optimizer": {**o, "type": "torch.optim.AdamW (bias-corrected)"},
                "effective_batch": o["micro_batch"] * o["grad_accumulation"], "schedule": self._sched,
                "loss": self._cfg["loss"], "predict_base": self._cfg["predict_base"], "device": self._device,
                "seed": self.seed, "dev_smoke": self._smoke}

    def _new_scorer(self) -> HfLetterScoringModel:
        return HfLetterScoringModel(
            name=self.name, repo=self._mcfg["repo"], revision=self._mcfg["revision"], prompt_format=self._format,
            format_name=self._mcfg["prompt_format"], dtype="bfloat16", device=self._device, load_model=False)

    def _load_training_model(self) -> None:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        q, lcfg = self._cfg["quantization"], self._cfg["lora"]
        compute = getattr(torch, q["compute_dtype"])
        model = _load_base(self._mcfg["repo"], self._mcfg["revision"], compute, _bnb_config(q, compute), self._device)
        inner = getattr(model, "model", None)
        if inner is not None and getattr(inner, "visual", None) is not None:
            inner.visual = None                       # text only, as in the zero-shot runs
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs={"use_reentrant": False})
        model = get_peft_model(model, LoraConfig(r=lcfg["r"], lora_alpha=lcfg["alpha"], lora_dropout=lcfg["dropout"],
                                                 bias="none", target_modules=target_pattern(lcfg["target_modules"])))
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        if not trainable or any("lora_" not in n for n in trainable):
            raise RuntimeError(f"unexpected trainable parameters: {[n for n in trainable if 'lora_' not in n][:3]}")
        self._model = model
        self.log["trainable_parameters"] = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
        self.log["lora_modules"] = len({n.rsplit(".lora_", 1)[0] for n in trainable})

    def _encode(self, items: Sequence[Item], labels: LabelSet) -> list[list[int]]:
        out = []
        for ex, cs in items:
            key = (self._mcfg["repo"], self._mcfg["revision"], self._mcfg["prompt_format"], ex.id, cs.candidates)
            if key not in _ENCODINGS:
                ex_free, cs_free = label_free((ex, cs))
                prompt = self._scorer.render(ex_free, cs_free, labels)
                self._scorer.check_answer_boundary(prompt, len(cs.candidates))
                _ENCODINGS[key] = self._scorer._tok(prompt, add_special_tokens=False)["input_ids"]
            out.append(_ENCODINGS[key])
        return out

    def _forward_letters(self, seqs: Sequence[Sequence[int]], ks: Sequence[int]):
        import torch

        base = self._model.get_base_model()
        pad = self._scorer._tok.pad_token_id if self._scorer._tok.pad_token_id is not None else 0
        ids, mask, last = right_pad(seqs, pad)
        dev = base.get_output_embeddings().weight.device
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self._device.startswith("cuda")):
            out = base.model(input_ids=ids.to(dev), attention_mask=mask.to(dev), use_cache=False)
        h = out.last_hidden_state[torch.arange(len(seqs), device=dev), last.to(dev)]
        k = torch.tensor(list(ks), device=dev)
        return candidate_logits(h, base.get_output_embeddings().weight, self._scorer._letter_ids, k)

    def _dev_scores(self, seqs, items: Sequence[Item]) -> tuple[float, float]:
        """Batched candidate-restricted dev macro-F1 and accuracy of the training model (selection only); examples
        are batched by length, the metric does not depend on order."""
        import torch

        from s1bench.domain.metrics import Scored, accuracy, macro_f1

        self._model.eval()
        order = sorted(range(len(items)), key=lambda i: len(seqs[i]))
        rows = []
        mb = self._sched["eval_micro_batch"]
        with torch.inference_mode():
            for i in range(0, len(order), mb):
                idx = order[i:i + mb]
                logits = self._forward_letters([seqs[j] for j in idx], [len(items[j][1].candidates) for j in idx])
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                for j, p in zip(idx, probs):
                    ex, cs = items[j]
                    rows.append(Scored(cs.candidates, tuple(float(x) for x in p[: len(cs.candidates)]), ex.label))
        self._model.train()
        return macro_f1(rows), accuracy(rows)

    # ------------------------------------------------------------------ API
    def fit(self, train: Sequence[Item], dev: Sequence[Item], labels: LabelSet) -> "QloraLetterClassifier":
        """Train on the 4-bit base (own frame, so optimizer, scheduler and activations are gone afterwards), then
        deploy the kept adapter on the prediction base."""
        self._train(train, dev, labels)
        self._deploy(labels.dataset)
        return self

    def _train(self, train: Sequence[Item], dev: Sequence[Item], labels: LabelSet) -> None:
        import torch
        from transformers import get_linear_schedule_with_warmup

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._scorer = self._new_scorer()
        self._load_training_model()
        o = self._cfg["optimizer"]
        eff = o["micro_batch"] * o["grad_accumulation"]
        epochs, total = planned_steps(len(train), eff, self._sched)
        evals = set(eval_steps(total, self._sched["eval_every"]))
        seqs, dev_seqs = self._encode(train, labels), self._encode(dev, labels)
        gold = [gold_position(it) for it in train]
        params = {n: p for n, p in self._model.named_parameters() if p.requires_grad}
        opt = torch.optim.AdamW(list(params.values()), lr=o["lr"], weight_decay=o["weight_decay"])
        sched = get_linear_schedule_with_warmup(opt, int(o["warmup_ratio"] * total), total)
        rng = random.Random(self.seed)
        order: list[int] = []
        best = (-1.0, None, 0.0, 0)        # (dev macro-F1, trainable state on CPU, dev accuracy, step)
        history, step, losses = [], 0, []
        t_train, eval_seconds = time.perf_counter(), 0.0
        patience = self._sched.get("patience")   # revision protocol (v1.3) early stopping; None = frozen rule
        since_best, stopped_at = 0, None
        self._model.train()
        while step < total and stopped_at is None:
            opt.zero_grad(set_to_none=True)
            step_loss = 0.0
            for _ in range(o["grad_accumulation"]):
                if len(order) < o["micro_batch"]:
                    nxt = list(range(len(train)))
                    rng.shuffle(nxt)
                    order += nxt
                idx, order = order[: o["micro_batch"]], order[o["micro_batch"]:]
                logits = self._forward_letters([seqs[j] for j in idx], [len(train[j][1].candidates) for j in idx])
                target = torch.tensor([gold[j] for j in idx], device=logits.device)
                loss = torch.nn.functional.cross_entropy(logits, target) / o["grad_accumulation"]
                loss.backward()
                step_loss += float(loss.detach())
            if not math.isfinite(step_loss):
                raise RuntimeError(f"non-finite training loss at step {step + 1} (seed {self.seed})")
            torch.nn.utils.clip_grad_norm_(list(params.values()), o["grad_clip"])
            opt.step()
            sched.step()
            step += 1
            losses.append(step_loss)
            if step in evals:
                t_eval = time.perf_counter()
                f1, acc = self._dev_scores(dev_seqs, dev)
                eval_seconds += time.perf_counter() - t_eval
                history.append({"step": step, "train_loss": round(float(np.mean(losses)), 4),
                                "dev_macro_f1": round(f1, 4), "dev_accuracy": round(acc, 4)})
                losses = []
                if f1 > best[0]:
                    best = (f1, {n: p.detach().cpu().clone() for n, p in params.items()}, acc, step)
                    since_best = 0
                else:
                    since_best += 1
                if patience is not None and since_best >= patience:
                    stopped_at = step
        res = self._model.load_state_dict(best[1], strict=False)
        if res.unexpected_keys or set(best[1]) & set(res.missing_keys):
            raise RuntimeError(f"best checkpoint not restored: {res.unexpected_keys[:2]}")
        col = {lid: i for i, lid in enumerate(labels.ids())}
        majority = candidate_majority_accuracy([col[e.label] for e, _ in train], [col[e.label] for e, _ in dev],
                                               [list(cs.candidates) for _, cs in dev], col)
        self.majority_dev_accuracy = float(majority)   # unrounded: the failed-run rule compares against this value
        self.log.update({"hparams": self.hparams(), "selection": "candidate-restricted dev macro-F1 (4-bit, batched)",
                         "total_steps": total, "epochs_planned": epochs, "best_step": best[3],
                         "best_dev_macro_f1_4bit": round(best[0], 4), "best_dev_accuracy_4bit": round(best[2], 4),
                         "majority_dev_accuracy": round(majority, 4),
                         "majority_kind": "candidate-restricted train-frequency predictor", "history": history,
                         "train_loop_seconds": round(time.perf_counter() - t_train, 1),
                         "dev_eval_seconds": round(eval_seconds, 1), "n_evals": len(history)})
        if patience is not None:
            self.log["stopped_early_at_step"] = stopped_at
        if torch.cuda.is_available():
            self.log["peak_vram_training_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        del opt, sched, params

    def _deploy(self, dataset: str) -> None:
        """Save the kept adapter, free the 4-bit training model, and load base (predict_base) + adapter exactly as the
        zero-shot runs load a model (same classes, slimmed for text). Every adapter tensor must be applied.
        Adapters live at <adapter_root>/<dataset>/<name>__seed<seed> and are never overwritten."""
        import tempfile

        import torch
        from peft import PeftModel

        n_modules = int(self.log["lora_modules"])
        if self._adapter_root is not None:
            self._adapter_dir = new_adapter_dir(Path(self._adapter_root), dataset, self.name, self.seed)
        else:
            self._adapter_dir, self._tmp_adapter = Path(tempfile.mkdtemp(prefix="qlora_adapter_")), True
        self._model.save_pretrained(str(self._adapter_dir), save_embedding_layers=False)
        self._model = None
        _free_cuda()
        if torch.cuda.is_available():
            left = torch.cuda.memory_allocated() / 2**20
            if left > 512:
                raise RuntimeError(f"{left:.0f} MiB still allocated after freeing the training model; "
                                   f"largest live CUDA tensors: {_live_cuda_tensors()}")
            torch.cuda.reset_peak_memory_stats()
        weights = self._adapter_dir / "adapter_model.safetensors"
        from safetensors import safe_open

        with safe_open(str(weights), "pt") as fh:
            file_keys = set(fh.keys())
        base_dtype = self._cfg["predict_base"]
        q = self._cfg["quantization"]
        if base_dtype == "4bit":                       # dev smoke on a small GPU only (same NF4 config as training)
            compute = getattr(torch, q["compute_dtype"])
            base = _load_base(self._mcfg["repo"], self._mcfg["revision"], compute, _bnb_config(q, compute), self._device)
        else:
            base = _load_base(self._mcfg["repo"], self._mcfg["revision"], getattr(torch, base_dtype))
        model = PeftModel.from_pretrained(base, str(self._adapter_dir))
        params = dict(model.named_parameters())
        adapter_key_check(file_keys, {k for k in params if "lora_" in k}, n_modules)
        if any(float(v.detach().abs().sum()) == 0.0 for k, v in params.items() if "lora_B" in k):
            raise RuntimeError("trained adapter has all-zero lora_B tensors")
        if base_dtype != "4bit":
            model = model.to(self._device)
        self._scorer._model = model.eval()
        self._scorer._memory = self._scorer._slim_for_text(self._scorer._model)
        self.log["deployed"] = {"base_dtype": base_dtype, "adapter_tensors": len(file_keys), "lora_modules": n_modules,
                                "adapter_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                                "adapter_dir": None if self._tmp_adapter else self._adapter_dir.as_posix(),
                                "memory": self._scorer._memory}
        if torch.cuda.is_available():
            self.log["gpu"] = torch.cuda.get_device_name(0)

    def predict(self, items: Sequence[Item], labels: LabelSet) -> list[Prediction]:
        """Deployed model, unbatched zero-shot readout, label-free inputs. Raises if any prediction fails, so that
        nothing is written for a broken run."""
        import torch

        version = f"{self._mcfg['revision']}+qlora:{self.name}:seed{self.seed}"
        out = []
        for ex, cs in items:
            p = self._scorer.predict(*label_free((ex, cs)), labels)
            out.append(replace(p, gold=ex.label, model=self.name, model_version=version))
        bad = [p for p in out if p.status != STATUS_OK]
        if bad:
            raise RuntimeError(f"{len(bad)} of {len(out)} predictions failed, e.g. {bad[0].error}")
        if torch.cuda.is_available():
            self.log["peak_vram_prediction_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        return out

    def release(self) -> None:
        import shutil

        self._model = None
        if self._scorer is not None:
            self._scorer._model = None
            self._scorer = None
        if self._tmp_adapter and self._adapter_dir is not None:
            shutil.rmtree(self._adapter_dir, ignore_errors=True)
        _free_cuda()


__all__ = ["QloraLetterClassifier", "planned_steps", "eval_steps", "target_pattern", "gold_position", "right_pad",
           "candidate_logits", "label_free", "adapter_key_check", "LETTERS"]
