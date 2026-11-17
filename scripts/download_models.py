"""Download every Stage-1 model at its pinned revision into HF_HOME (GPU machine setup).

Usage (from the project root, HF_HOME set to a big drive):
    python scripts/download_models.py            # Stage 1 core models (~38 GB)
    python scripts/download_models.py --list     # show what would be downloaded
Revisions come from configs/models.yaml (decoders, Nimble adapters) and the DeBERTa default in
src/s1bench/infrastructure/models/deberta_classifier.py; nothing is downloaded at a floating revision.
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

import yaml

sys.path.insert(0, "src")
from s1bench.infrastructure.models.deberta_classifier import DebertaClassifier  # noqa: E402

WEIGHTS = ["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model", "*.py", "LICENSE*", "README*"]


def targets() -> list[tuple[str, str, list[str]]]:
    cfg = yaml.safe_load(Path("configs/models.yaml").read_text(encoding="utf-8"))
    out: dict[tuple[str, str], list[str]] = {}
    for m in cfg["models"].values():
        out[(m["repo"], m["revision"])] = WEIGHTS
        if m.get("adapter_repo"):
            out[(m["adapter_repo"], m["adapter_revision"])] = WEIGHTS
    sig = inspect.signature(DebertaClassifier.__init__).parameters
    # DeBERTa: pytorch_model.bin + tokenizer only (pytorch_model.generator.bin, 2.3 GB, is not used)
    out[(sig["repo"].default, sig["revision"].default)] = ["*.json", "pytorch_model.bin", "spm.model", "*.txt"]
    return [(r, v, p) for (r, v), p in out.items()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    from huggingface_hub import snapshot_download

    for repo, rev, patterns in targets():
        print(f"{repo} @ {rev}")
        if not args.list:
            snapshot_download(repo, revision=rev, allow_patterns=patterns)
    return 0


if __name__ == "__main__":
    sys.exit(main())
