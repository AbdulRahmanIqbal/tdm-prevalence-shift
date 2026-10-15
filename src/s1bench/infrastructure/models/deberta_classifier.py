"""Fine-tuned DeBERTa-v3-large classifier (spec sec 10.3).

Stability rules (Mosbach et al. 2021; Zhang et al. 2021): AdamW with bias correction (torch default),
linear warmup over the first 10% of steps then linear decay, weight decay 0.01, dropout 0.1,
at least `min_steps` optimizer steps for small budgets, best checkpoint by dev macro-F1.
Failed run (Dodge et al. 2020 / Mosbach et al. 2021): dev accuracy of the kept (best) checkpoint <= dev accuracy of
the majority-class predictor. v1.1.1 (audit 2026-10-08): both on the SAME scale, i.e. inside each example's
candidate set: the majority predictor = train class frequencies restricted to the candidates (the spec 10.3
fallback). Before, the candidate-restricted accuracy (chance ~5% on CLINC150/HWU64) was compared with the
all-label majority accuracy (0.7% / 1.8%), so the rule could never fire there.
v1.0.4: checkpoint selection uses the candidate-restricted dev macro-F1 (the paper metric, same as the C/LR
selection); the best state is kept on the CPU; every hyperparameter is logged; real runs refuse CPU.
"""

from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np


def candidate_majority_accuracy(y_train: list[int], y_dev: list[int], dev_candidates: list[list[str]],
                                col: dict[str, int]) -> float:
    """Dev accuracy of the spec 10.3 fallback predictor: per example, the candidate with the highest train class
    frequency (ties: earliest candidate position, as argmax_label). Same scale as candidate-restricted accuracy."""
    freq = np.bincount(np.array(y_train, dtype=int), minlength=len(col))
    pred = [max(cands, key=lambda c: (freq[col[c]], -cands.index(c))) for cands in dev_candidates]
    return float(np.mean([col[c] == g for c, g in zip(pred, y_dev)]))


class DebertaClassifier:
    def __init__(self, seed: int, repo: str = "microsoft/deberta-v3-large",
                 revision: str = "64a8c8eab3e352a784c658aef62be1662607476f", lr: float = 1e-5,
                 batch_size: int = 16, max_length: int = 128, min_steps: int = 500,
                 epochs_small: int = 20, epochs_large: int = 5, small_budget: int = 500,
                 eval_every: int = 100, device: str = "cpu", fp16: bool = True) -> None:
        self.seed = seed
        self.repo, self.revision = repo, revision
        self.lr, self.batch_size, self.max_length = lr, batch_size, max_length
        self.min_steps, self.eval_every = min_steps, eval_every
        self.epochs_small, self.epochs_large, self.small_budget = epochs_small, epochs_large, small_budget
        self.device = device
        self.fp16 = fp16 and device.startswith("cuda")
        self.warmup_ratio, self.weight_decay, self.grad_clip = 0.1, 0.01, 1.0
        self.failed = False
        self.log: dict[str, object] = {}
        self._model = None
        self._tok = None
        self._label_ids: list[str] = []

    # ------------------------------------------------------------------ helpers
    def _batches(self, texts, labels=None, shuffle=False, rng=None):
        order = list(range(len(texts)))
        if shuffle:
            rng.shuffle(order)
        for i in range(0, len(order), self.batch_size):
            idx = order[i:i + self.batch_size]
            enc = self._tok([texts[j] for j in idx], truncation=True, max_length=self.max_length,
                            padding=True, return_tensors="pt").to(self.device)
            if labels is not None:
                import torch
                enc["labels"] = torch.tensor([labels[j] for j in idx], device=self.device)
            yield enc

    def _probs(self, texts: list[str]) -> np.ndarray:
        import torch

        self._model.eval()
        out = []
        with torch.inference_mode():
            for enc in self._batches(texts):
                with torch.autocast(device_type="cuda", enabled=self.fp16):
                    logits = self._model(**enc).logits.float()
                out.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self._label_ids)))

    @staticmethod
    def _macro_f1(gold: list[int], pred: list[int]) -> float:
        classes = sorted(set(gold) | set(pred))
        f1s = []
        for c in classes:
            tp = sum(1 for g, p in zip(gold, pred) if g == c and p == c)
            fp = sum(1 for g, p in zip(gold, pred) if g != c and p == c)
            fn = sum(1 for g, p in zip(gold, pred) if g == c and p != c)
            f1s.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
        return sum(f1s) / len(f1s)

    # ------------------------------------------------------------------ API
    def hparams(self) -> dict[str, object]:
        return {"repo": self.repo, "revision": self.revision, "lr": self.lr, "batch_size": self.batch_size,
                "max_length": self.max_length, "min_steps": self.min_steps, "epochs_small": self.epochs_small,
                "epochs_large": self.epochs_large, "small_budget": self.small_budget, "eval_every": self.eval_every,
                "warmup_ratio": self.warmup_ratio, "weight_decay": self.weight_decay, "grad_clip": self.grad_clip,
                "fp16": self.fp16, "device": self.device, "optimizer": "torch.optim.AdamW (bias-corrected)",
                "schedule": "linear warmup then linear decay to 0", "seed": self.seed}

    def release(self) -> None:
        """Free accelerator memory (a failed or non-selected model must not stay on the GPU)."""
        self._model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def fit(self, texts: list[str], labels: list[str], dev_texts: list[str] | None = None,
            dev_labels: list[str] | None = None, label_ids: list[str] | None = None,
            dev_candidates: list[list[str]] | None = None) -> "DebertaClassifier":
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

        if dev_texts is None or dev_labels is None or label_ids is None:
            raise ValueError("DeBERTa needs dev data (early stopping, failed-run rule) and the label ids")
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self._label_ids = list(label_ids)
        col = {lid: i for i, lid in enumerate(self._label_ids)}
        y = [col[lab] for lab in labels]
        y_dev = [col[lab] for lab in dev_labels]
        # candidate mask for selection (paper metric = argmax within the example's frozen candidate set)
        dev_mask = np.ones((len(dev_texts), len(self._label_ids)), dtype=bool)
        if dev_candidates is not None:
            dev_mask[:] = False
            for r, cands in enumerate(dev_candidates):
                dev_mask[r, [col[c] for c in cands]] = True

        self._tok = AutoTokenizer.from_pretrained(self.repo, revision=self.revision)
        # Master weights must be fp32 for mixed-precision training (transformers >= 5 otherwise keeps the
        # checkpoint dtype, and GradScaler refuses to unscale fp16 gradients).
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.repo, revision=self.revision, num_labels=len(self._label_ids),
            dtype=torch.float32).to(self.device)

        epochs = self.epochs_small if len(texts) <= self.small_budget else self.epochs_large
        steps_per_epoch = math.ceil(len(texts) / self.batch_size)
        total_steps = max(self.min_steps, epochs * steps_per_epoch)
        no_decay = ("bias", "LayerNorm.weight", "layernorm", "layer_norm")
        groups = [
            {"params": [p for n, p in self._model.named_parameters() if not any(k in n for k in no_decay)],
             "weight_decay": self.weight_decay},
            {"params": [p for n, p in self._model.named_parameters() if any(k in n for k in no_decay)],
             "weight_decay": 0.0},
        ]
        opt = torch.optim.AdamW(groups, lr=self.lr)  # torch AdamW includes bias correction
        sched = get_linear_schedule_with_warmup(opt, int(self.warmup_ratio * total_steps), total_steps)
        scaler = torch.amp.GradScaler("cuda", enabled=self.fp16)
        rng = random.Random(self.seed)

        best = (-1.0, None, 0.0)  # (dev macro-F1, state_dict, dev accuracy)
        step = 0
        history = []
        while step < total_steps:
            self._model.train()
            for enc in self._batches(texts, y, shuffle=True, rng=rng):
                with torch.autocast(device_type="cuda", enabled=self.fp16):
                    loss = self._model(**enc).loss
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                if step % self.eval_every == 0 or step == total_steps:
                    pred = np.where(dev_mask, self._probs(dev_texts), -1.0).argmax(axis=1).tolist()
                    f1 = self._macro_f1(y_dev, pred)
                    acc = float(np.mean(np.array(pred) == np.array(y_dev)))
                    history.append({"step": step, "dev_macro_f1": round(f1, 4), "dev_accuracy": round(acc, 4)})
                    if f1 > best[0]:
                        best = (f1, {k: v.detach().cpu().clone() for k, v in self._model.state_dict().items()}, acc)
                    self._model.train()
                if step >= total_steps:
                    break
        if best[1] is not None:
            self._model.load_state_dict(best[1])
        if dev_candidates is not None:   # same scale as the candidate-restricted dev accuracy (v1.1.1)
            majority = candidate_majority_accuracy(y, y_dev, dev_candidates, col)
            majority_kind = "candidate-restricted train-frequency predictor"
        else:
            majority = Counter(y_dev).most_common(1)[0][1] / len(y_dev)
            majority_kind = "all-label majority class"
        self.failed = best[2] <= majority
        self.log = {"hparams": self.hparams(), "selection": "candidate-restricted dev macro-F1" if dev_candidates
                    is not None else "full-label dev macro-F1",
                    "total_steps": total_steps, "epochs_planned": epochs, "best_dev_macro_f1": round(best[0], 4),
                    "best_dev_accuracy": round(best[2], 4), "majority_dev_accuracy": round(majority, 4),
                    "majority_kind": majority_kind,
                    "failed": self.failed, "history": history}
        return self

    def time_single(self, texts: list[str]) -> list[float]:
        """Per-example inference latency (spec 9; v1.1): batch size 1, tokenization outside the timer (as for
        the decoder models), forward + softmax timed with CUDA synchronisation. Returns milliseconds."""
        import time

        import torch

        self._model.eval()
        out = []
        with torch.inference_mode():
            for text in texts:
                enc = self._tok([text], truncation=True, max_length=self.max_length,
                                return_tensors="pt").to(self.device)
                if self.device.startswith("cuda"):
                    torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.autocast(device_type="cuda", enabled=self.fp16):
                    torch.softmax(self._model(**enc).logits.float(), dim=-1)
                if self.device.startswith("cuda"):
                    torch.cuda.synchronize()
                out.append((time.perf_counter() - start) * 1000)
        return out

    def timing_info(self) -> dict[str, object]:
        import torch

        info: dict[str, object] = {"device": self.device, "batch_size": 1, "fp16_autocast": self.fp16}
        if self.device.startswith("cuda"):
            info["gpu"] = torch.cuda.get_device_name(0)
        return info

    def predict_proba(self, texts: list[str], label_ids: list[str]) -> np.ndarray:
        if list(label_ids) != self._label_ids:
            raise ValueError("label order mismatch")
        return self._probs(texts)
