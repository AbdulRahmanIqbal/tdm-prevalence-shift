"""Local decoder models read out by candidate-letter scoring (spec sec 7.2, 7.3, 8).

The prompt is rendered in the TDM partner's native format with enable_thinking=False, so the base
model and its fine-tune see identical tokens. The prediction is the argmax of a softmax (T = 1)
over the candidate letters' logits at the first answer position (right after the assistant prefix).

v1.0.4 (code audit 2026-10-07):
- All four models load as Qwen3_5ForConditionalGeneration (their config `architectures`, and the class the
  Nimble adapter was trained on: adapter keys `...model.language_model.layers...`). Loading the causal-LM
  class silently left the LoRA un-applied (0/496 adapter keys matched), so P2 compared Qwen with Qwen.
  After loading an adapter we now require every adapter tensor to be loaded and every lora_B to be non-zero.
- Letter logits are computed in fp32 from the final hidden state (h @ W_letters^T); the bf16 lm_head output
  had 0.125-wide steps that created exact ties (resolved toward letter A by the position tie-break).
- On CUDA the timer stops after torch.cuda.synchronize().

Memory (pilot 2026-10-07, RTX 4070 Ti 12 GB): Nimble-9B 8-bit reserved 11.55 GB and spilled into shared
system memory under Windows (latency 11.6 s vs 0.36 s for Qwen3.5-9B on the same prompts). For text-only
scoring, identically for all four local models: the unused vision encoder is removed after loading, and the
input-embedding lookup runs from a CPU copy of the table (pure row gather: bit-identical embeddings). When
input and output embeddings are not tied (Qwen3.5-9B) the GPU copy of the input table is freed (~2 GB).

Prefix caching (spec 9, audit C4; v1.1): `time_prefix_cache` measures latency with and without a cached shared
prefix (the longest common token prefix of the rendered prompts: system prompt + chat template + the opening of
the user JSON; the example text comes before the options in both native formats). Latency only: every reported
prediction comes from `predict` (no cache).

Prompt sensitivity (revision protocol v1.3, G5; exploratory, added after the results): two generic formats that belong
to neither TDM, from configs/experiments/prompt_sensitivity.yaml (the frozen formats above are unchanged):
- 'plain': plain-text multiple choice (task question, text, lettered options with descriptions); letter readout as above.
- 'plain_names': plain-text question with the option names; option-likelihood readout: softmax (T = 1) over the summed
  fp32 log-probabilities of each option name's tokens as the answer continuation (one forward pass per option).
"""

from __future__ import annotations

import copy
import json
import re
import string
import time

from s1bench.domain.candidates import CandidateSet
from s1bench.domain.examples import Example, LabelSet
from s1bench.domain.predictions import STATUS_FAILED, STATUS_OK, Prediction, argmax_label

LETTERS = string.ascii_uppercase


def label_key(name: str) -> str:
    """Tev1 option `key`: a short semantic slug of the label name (Tev1 trains on keys like 'yes'/'no');
    raw dataset ids such as FPB's '0'/'1'/'2' are not used."""
    return re.sub(r"[^0-9a-z]+", "_", name.lower()).strip("_")


def _nimble_safe_json(value: object) -> str:
    """Verbatim from Nimble parallel_schema.safe_json."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e")


class HfLetterScoringModel:
    def __init__(self, name: str, repo: str, revision: str, prompt_format: dict[str, str],
                 format_name: str, dtype: str = "bfloat16", device: str = "cpu",
                 adapter_repo: str | None = None, adapter_revision: str | None = None,
                 quantization: str | None = None, load_model: bool = True,
                 offload_gpu_gib: float | None = None) -> None:
        import torch
        from transformers import AutoTokenizer

        self.name = name
        self._repo, self._revision = repo, revision
        self._format, self._format_name = prompt_format, format_name
        self._device = device
        self._dtype_name = dtype
        self._adapter = (adapter_repo, adapter_revision)
        self._quant = quantization
        self._offload = offload_gpu_gib
        if offload_gpu_gib is not None and (quantization or not device.startswith("cuda")):
            raise ValueError(f"{name}: CPU offload is for unquantized models on --device cuda")
        if quantization and load_model and not device.startswith("cuda"):
            raise ValueError(f"{name}: {quantization} quantization needs --device cuda (bitsandbytes places the "
                             "model on cuda:0)")
        torch_dtype = getattr(torch, dtype)
        self._tok = AutoTokenizer.from_pretrained(repo, revision=revision)
        self._model = None
        self._adapter_check: dict[str, object] | None = None
        self._memory: dict[str, object] = {}
        self._cpu_embed = None
        if load_model:
            model = self._load_model(repo, revision, torch_dtype, quantization)
            if adapter_repo:
                from peft import PeftModel  # Nimble = LoRA adapter on the SAME base weights (spec sec 2)

                model = PeftModel.from_pretrained(model, adapter_repo, revision=adapter_revision)
                self._adapter_check = self.verify_adapter(model, adapter_repo, adapter_revision)
            if offload_gpu_gib is not None:   # G6: slim on the CPU, then spread the layers over GPU and CPU
                self._model = model.eval()
                self._memory = self._slim_for_text(self._model)
                self._memory["offload"] = self._dispatch_offload(self._model, offload_gpu_gib, torch_dtype)
            else:
                if quantization is None:
                    model = model.to(device)
                self._model = model.eval()
                self._memory = self._slim_for_text(self._model)
        self._letter_ids = []
        for letter in LETTERS[:24]:
            ids = self._tok.encode(letter, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"letter {letter!r} is not a single token for {repo}")
            self._letter_ids.append(ids[0])

    @staticmethod
    def _load_model(repo: str, revision: str, torch_dtype, quantization: str | None = None):
        import transformers

        kwargs: dict[str, object] = {"dtype": torch_dtype}
        if quantization in ("8bit", "4bit"):
            kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
                load_in_8bit=quantization == "8bit", load_in_4bit=quantization == "4bit",
                bnb_4bit_compute_dtype=torch_dtype)
            kwargs["device_map"] = "cuda:0"
        errors = []
        # ConditionalGeneration first: the checkpoints' own class and the class Nimble's adapter targets.
        for cls_name in ("AutoModelForImageTextToText", "AutoModelForCausalLM"):
            try:
                cls = getattr(transformers, cls_name)
                return cls.from_pretrained(repo, revision=revision, **kwargs)
            except (ValueError, KeyError, TypeError) as e:
                errors.append(f"{cls_name}: {e}")
        raise RuntimeError("could not load model:\n" + "\n".join(errors))

    @staticmethod
    def _dispatch_offload(model, gpu_gib: float, torch_dtype) -> dict[str, object]:
        """Revision protocol v1.3 (G6): bf16 weights larger than the GPU. accelerate places whole decoder layers on
        cuda:0 up to gpu_gib and keeps the rest in CPU memory; an offloaded layer's weights (LoRA included) are
        copied to cuda:0 for its forward pass, so every computation runs on the GPU in bf16 as without offload.
        lm_head and the final norm are kept on the GPU (the letter readout reads lm_head rows directly)."""
        from accelerate import dispatch_model, infer_auto_device_map

        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        no_split = list(getattr(base, "_no_split_modules", None) or [])
        dm = infer_auto_device_map(model, max_memory={0: f"{gpu_gib}GiB", "cpu": "512GiB"},
                                   no_split_module_classes=no_split, dtype=torch_dtype)
        heads = [k for k in dm if k.endswith("lm_head")]
        if len(heads) != 1:
            raise RuntimeError(f"offload: lm_head is not a separate device-map entry ({len(heads)} found)")
        for k in list(dm):
            if k.endswith("lm_head") or k.endswith("language_model.norm") or k.endswith("model.norm"):
                dm[k] = 0
        dispatch_model(model, device_map=dm, main_device=0)
        on_cpu = sorted(k for k, v in dm.items() if v in ("cpu", "disk"))
        if any(v == "disk" for v in dm.values()):
            raise RuntimeError("offload: device map spills to disk; not enough CPU memory")
        return {"offload_gpu_gib": gpu_gib, "no_split_module_classes": no_split,
                "device_map_entries": len(dm), "entries_on_cpu": len(on_cpu), "cpu_entries": on_cpu,
                "execution_device": "cuda:0"}

    @staticmethod
    def verify_adapter(model, adapter_repo: str, adapter_revision: str | None) -> dict[str, object]:
        """Fail hard unless the adapter really changed the model: every tensor in the adapter file must match
        a LoRA parameter of the model, and every lora_B must be non-zero (PEFT only warns on missing keys)."""
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open

        path = hf_hub_download(adapter_repo, "adapter_model.safetensors", revision=adapter_revision)
        with safe_open(path, "pt") as fh:
            file_keys = set(fh.keys())
        params = dict(model.named_parameters())
        model_keys = {k.replace(".default", "") for k in params if "lora_" in k}
        missing, unexpected = sorted(file_keys - model_keys), sorted(model_keys - file_keys)
        zero_b = [k for k, v in params.items() if "lora_B" in k and float(v.detach().abs().sum()) == 0.0]
        if missing or unexpected or zero_b or not file_keys:
            raise RuntimeError(f"adapter {adapter_repo}@{adapter_revision} not applied: {len(missing)} file tensors "
                               f"unmatched, {len(unexpected)} model LoRA tensors without weights, "
                               f"{len(zero_b)} zero lora_B (e.g. {(missing or unexpected or zero_b)[:2]})")
        return {"adapter_tensors": len(file_keys), "matched": len(file_keys), "zero_lora_B": 0}

    def describe(self) -> dict[str, object]:
        import torch

        info: dict[str, object] = {
            "name": self.name, "repo": self._repo, "revision": self._revision, "dtype": self._dtype_name,
            "device": self._device, "prompt_format": self._format_name, "enable_thinking": False,
            "readout": ("softmax (T=1) over summed option-name token log-probabilities"
                        if self._format_name == "plain_names" else
                        "softmax (T=1) over candidate-letter logits at first answer position"),
            "model_class": type(self._model).__name__,
            "adapter": list(self._adapter) if self._adapter[0] else None, "quantization": self._quant,
            "adapter_check": self._adapter_check, "memory": self._memory,
            "cpu_offload": self._offload is not None,
            "letter_logits": "fp32: final hidden state @ lm_head rows of the candidate letters",
        }
        if self._device.startswith("cuda"):
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda_synchronized_timing"] = True
            info["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)  # tensors only
            info["peak_vram_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)  # + allocator cache
        return info

    def render(self, example: Example, candidates: CandidateSet, labels: LabelSet) -> str:
        """Native format of the TDM partner (spec 7.2): 'tev1' or 'nimble'."""
        by_id = {lab.id: lab for lab in labels.labels}
        if self._format_name == "tev1":
            # Tev1 build_dataset.py: json.dumps default separators, keys state/question/options
            options = [
                {"label": LETTERS[i], "key": label_key(by_id[cid].name),
                 "description": f"{by_id[cid].name}: {by_id[cid].description}"}
                for i, cid in enumerate(candidates.candidates)
            ]
            payload = {"state": example.text, "question": labels.task_question, "options": options}
            content = json.dumps(payload, ensure_ascii=False)
        elif self._format_name == "nimble":
            # Nimble parallel_schema.py (bd792f44): one enum field "label"; choice value = label name,
            # choice description = label description; codes A.. in candidate order.
            field = {"name": "label", "description": labels.task_question, "choices": [
                {"code": LETTERS[i], "value": by_id[cid].name, "description": by_id[cid].description}
                for i, cid in enumerate(candidates.candidates)]}
            content = (_nimble_safe_json({"context": example.text, "schema": [field]})
                       + "\n\nRequested field: " + _nimble_safe_json("label"))
        elif self._format_name == "plain":
            opts = "\n".join(f"{LETTERS[i]}. {by_id[cid].name}: {by_id[cid].description}"
                             for i, cid in enumerate(candidates.candidates))
            content = (f"{labels.task_question}\n\nText: {example.text}\n\nOptions:\n{opts}\n\n"
                       "Answer with the letter of the correct option only.")
        elif self._format_name == "plain_names":
            opts = "\n".join(f"- {by_id[cid].name}: {by_id[cid].description}" for cid in candidates.candidates)
            content = (f"{labels.task_question}\n\nText: {example.text}\n\nOptions:\n{opts}\n\n"
                       "Answer with the name of the correct option only.")
        else:
            raise ValueError(f"unknown prompt format {self._format_name!r}")
        messages = ([{"role": "system", "content": self._format["system"]}] if self._format.get("system") else [])
        messages.append({"role": "user", "content": content})
        prompt = self._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False)
        if not prompt.endswith(self._format["expected_prefix"]):
            raise ValueError(f"unexpected assistant prefix: {prompt[-60:]!r}")
        return prompt

    def _first_device(self):
        base = self._model.get_base_model() if hasattr(self._model, "get_base_model") else self._model
        return base.get_output_embeddings().weight.device

    def _slim_for_text(self, model) -> dict[str, object]:
        """Drop the vision encoder and serve input embeddings from a CPU copy (see module docstring)."""
        import torch

        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        info: dict[str, object] = {"base_model_class": type(base).__name__}
        inner = getattr(base, "model", None)
        if inner is not None and getattr(inner, "visual", None) is not None:
            info["vision_params_removed"] = int(sum(p.numel() for p in inner.visual.parameters()))
            inner.visual = None
        emb = base.get_input_embeddings()
        out_emb = base.get_output_embeddings()
        tied = out_emb is not None and out_emb.weight.data_ptr() == emb.weight.data_ptr()
        self._cpu_embed = torch.nn.Embedding.from_pretrained(emb.weight.detach().to("cpu").clone(), freeze=True)
        if not tied:   # nothing else uses the GPU input table once inputs_embeds are passed
            emb.weight = torch.nn.Parameter(torch.empty(0, emb.weight.shape[1], dtype=emb.weight.dtype,
                                                        device=emb.weight.device), requires_grad=False)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        info.update({"tied_embeddings": bool(tied), "input_embeddings": "cpu copy",
                     "gpu_memory_freed_mb": round((before - after) / 2**20, 1)})
        return info

    def _letter_logits(self, ids, letter_ids: list[int]):
        """(fp32 logits of the candidate letters, native full-vocab logits for diagnostics) at the last position.
        Uses the language backbone + lm_head of the (possibly PEFT-wrapped) model; LoRA layers stay active."""
        import torch

        base = self._model.get_base_model() if hasattr(self._model, "get_base_model") else self._model
        dev = ids["input_ids"].device
        if self._cpu_embed is not None:
            embeds = self._cpu_embed(ids["input_ids"].to("cpu")).to(dev)
            out = base.model(inputs_embeds=embeds, attention_mask=ids["attention_mask"], use_cache=False)
        else:
            out = base.model(input_ids=ids["input_ids"], attention_mask=ids["attention_mask"], use_cache=False)
        h = out.last_hidden_state[0, -1]
        return self._letters_from_hidden(h, letter_ids), base.get_output_embeddings()(h)

    def _letters_from_hidden(self, h, letter_ids: list[int]):
        base = self._model.get_base_model() if hasattr(self._model, "get_base_model") else self._model
        head = base.get_output_embeddings()
        letters = h.float() @ head.weight[letter_ids].float().T
        if getattr(head, "bias", None) is not None:
            letters = letters + head.bias[letter_ids].float()
        return letters

    def _embed(self, input_ids, dev):
        return self._cpu_embed(input_ids.to("cpu")).to(dev) if self._cpu_embed is not None else None

    def time_prefix_cache(self, items: list[tuple[Example, CandidateSet, LabelSet]],
                          warmup: int = 10) -> dict[str, object]:
        """Latency with prefix caching OFF (full forward, exactly as predict) and ON (shared-prefix KV/state
        cache computed once; per example: deep copy of the cache + forward of the suffix only; the copy is
        timed, so ON is conservative). Both paths per example, interleaved; the first `warmup` examples are
        dropped from the percentiles. Also reports the largest probability difference between the paths."""
        import numpy as np
        import torch

        base = self._model.get_base_model() if hasattr(self._model, "get_base_model") else self._model
        dev = self._first_device()
        cuda = self._device.startswith("cuda")
        rows = []
        for ex, cs, labels in items:
            prompt = self.render(ex, cs, labels)
            self.check_answer_boundary(prompt, len(cs.candidates))
            rows.append((self._tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"],
                         len(cs.candidates)))
        seqs = [r[0][0].tolist() for r in rows]
        n_prefix = min(len(q) for q in seqs) - 1          # at least one suffix token per example
        for q in seqs[1:]:
            n_prefix = next((i for i, (a, b) in enumerate(zip(seqs[0][:n_prefix], q[:n_prefix])) if a != b), n_prefix)
        if n_prefix < 1:
            raise ValueError("no shared token prefix")

        def run(ids, cache=None):
            ids = ids.to(dev)
            embeds = self._embed(ids, dev)
            kw = {"inputs_embeds": embeds} if embeds is not None else {"input_ids": ids}
            return base.model(**kw, past_key_values=cache, use_cache=True)

        def sync():
            if cuda:
                torch.cuda.synchronize()

        off_ms, on_ms, max_diff, flips = [], [], 0.0, 0
        with torch.inference_mode():
            prefix_cache = run(rows[0][0][:, :n_prefix]).past_key_values
            sync()
            for ids, k in rows:
                letter_ids = self._letter_ids[:k]
                start = time.perf_counter()
                full = {"input_ids": ids.to(dev), "attention_mask": torch.ones_like(ids).to(dev)}
                off_letters, _ = self._letter_logits(full, letter_ids)
                sync()
                off_ms.append((time.perf_counter() - start) * 1000)
                start = time.perf_counter()
                out = run(ids[:, n_prefix:], copy.deepcopy(prefix_cache))
                on_letters = self._letters_from_hidden(out.last_hidden_state[0, -1], letter_ids)
                sync()
                on_ms.append((time.perf_counter() - start) * 1000)
                p_off, p_on = torch.softmax(off_letters, -1), torch.softmax(on_letters, -1)
                max_diff = max(max_diff, float((p_off - p_on).abs().max()))
                flips += int(int(p_off.argmax()) != int(p_on.argmax()))

        def pct(xs):
            a = np.array(xs[warmup:], dtype=float)
            return {"p50_ms": round(float(np.percentile(a, 50)), 2), "p95_ms": round(float(np.percentile(a, 95)), 2),
                    "mean_ms": round(float(a.mean()), 2), "n": int(len(a))}

        return {"model": self.name, "n_examples": len(rows), "warmup_dropped": warmup, "prefix_tokens": n_prefix,
                "mean_input_tokens": round(float(np.mean([len(q) for q in seqs])), 1),
                "cache_off": pct(off_ms), "cache_on": pct(on_ms), "cache_copy_timed": True,
                "max_abs_prob_difference_on_vs_off": max_diff, "argmax_differs_on_vs_off": flips,
                "note": "latency only; reported predictions always use the uncached path"}

    def check_answer_boundary(self, prompt: str, k: int) -> None:
        """Every candidate letter must be ONE ordinary token right after the prompt (Nimble's own check)."""
        ids = self._tok.encode(prompt, add_special_tokens=False)
        for i, letter in enumerate(LETTERS[:k]):
            combined = self._tok.encode(prompt + letter, add_special_tokens=False)
            if combined[: len(ids)] != ids or combined[len(ids):] != [self._letter_ids[i]]:
                raise ValueError(f"letter {letter} is not a single boundary token")

    def _name_scores(self, prompt: str, names: list[str]):
        """Summed fp32 log-probability of each option name's tokens right after the prompt (G5 readout)."""
        import torch

        base = self._model.get_base_model() if hasattr(self._model, "get_base_model") else self._model
        head = base.get_output_embeddings()
        dev = self._first_device()
        p_ids = self._tok.encode(prompt, add_special_tokens=False)
        scores, n_tokens = [], 0
        for name in names:
            n_ids = self._tok.encode(name, add_special_tokens=False)
            if self._tok.encode(prompt + name, add_special_tokens=False) != p_ids + n_ids:
                raise ValueError(f"option name {name!r} does not tokenize separately after the prompt")
            ids = torch.tensor([p_ids + n_ids], device=dev)
            mask = torch.ones_like(ids)
            if self._cpu_embed is not None:
                out = base.model(inputs_embeds=self._cpu_embed(ids.to("cpu")).to(dev), attention_mask=mask,
                                 use_cache=False)
            else:
                out = base.model(input_ids=ids, attention_mask=mask, use_cache=False)
            h = out.last_hidden_state[0, len(p_ids) - 1: len(p_ids) + len(n_ids) - 1].float()
            logits = h @ head.weight.float().T
            if getattr(head, "bias", None) is not None:
                logits = logits + head.bias.float()
            logp = torch.log_softmax(logits, dim=-1)
            scores.append(float(logp[torch.arange(len(n_ids)), torch.tensor(n_ids, device=logp.device)].sum()))
            n_tokens += len(p_ids) + len(n_ids)
        return torch.tensor(scores), n_tokens

    def _predict_names(self, example: Example, candidates: CandidateSet, labels: LabelSet, base: dict) -> Prediction:
        import torch

        by_id = {lab.id: lab for lab in labels.labels}
        prompt = self.render(example, candidates, labels)
        start = time.perf_counter()
        with torch.inference_mode():
            scores, n_tokens = self._name_scores(prompt, [by_id[c].name for c in candidates.candidates])
            if self._device.startswith("cuda"):
                torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start) * 1000
        probs = torch.softmax(scores, dim=-1).tolist()
        return Prediction(**base, probabilities=tuple(probs), predicted=argmax_label(candidates.candidates, probs),
                          status=STATUS_OK, latency_ms=round(latency_ms, 2), input_tokens=n_tokens,
                          model_version=self._revision,
                          raw={"name_logprob_sums_fp32": [round(float(x), 5) for x in scores.tolist()]})

    def predict(self, example: Example, candidates: CandidateSet, labels: LabelSet) -> Prediction:
        import torch

        base = dict(example_id=example.id, model=self.name, dataset=example.dataset, split=example.split,
                    candidates=candidates.candidates, gold=example.label)
        try:
            if self._format_name == "plain_names":
                return self._predict_names(example, candidates, labels, base)
            prompt = self.render(example, candidates, labels)
            self.check_answer_boundary(prompt, len(candidates.candidates))
            ids = self._tok(prompt, return_tensors="pt", add_special_tokens=False).to(self._first_device())
            k = len(candidates.candidates)
            start = time.perf_counter()
            with torch.inference_mode():
                cand_logits, full_logits = self._letter_logits(ids, self._letter_ids[:k])
                if self._device.startswith("cuda"):
                    torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - start) * 1000
            probs = torch.softmax(cand_logits, dim=-1).tolist()
            full = torch.softmax(full_logits.float(), dim=-1)
            letter_mass = float(full[self._letter_ids[:k]].sum())
            top_token = self._tok.decode([int(torch.argmax(full_logits))])
            return Prediction(
                **base, probabilities=tuple(probs), predicted=argmax_label(candidates.candidates, probs),
                status=STATUS_OK, latency_ms=round(latency_ms, 2), input_tokens=int(ids["input_ids"].shape[1]),
                model_version=self._revision,
                raw={"letter_mass_full_vocab": round(letter_mass, 6), "top_token": top_token,
                     "letter_logits_fp32": [round(float(x), 5) for x in cand_logits.tolist()]},
            )
        except Exception as e:  # never crash a run; record the failure (spec sec 13)
            return Prediction(**base, probabilities=(), predicted=None, status=STATUS_FAILED,
                              error=f"{type(e).__name__}: {e}"[:500])
