"""Read data/raw/<dataset>/ into RawExample rows. One function per dataset; no cleaning here."""

from __future__ import annotations

import glob
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from s1bench.domain.examples import RawExample

# HateXplain invariants (Mathew et al. 2021; verified on the raw files, docs/raw_data_facts.txt).
_HATEXPLAIN_SPLIT_SIZES = {"train": 15383, "val": 1922, "test": 1924}
_HATEXPLAIN_TEST_COUNTS = {"hatespeech": 594, "offensive": 548, "normal": 782}
_HATEXPLAIN_UNDECIDED = 919


def _parquet(pattern: str) -> pd.DataFrame:
    matches = glob.glob(pattern)
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one file for {pattern}, found {matches}")
    return pd.read_parquet(matches[0])


class RawDatasetReaders:
    def __init__(self, raw_root: Path) -> None:
        self._root = raw_root

    def read(self, dataset: str) -> dict[str, list[RawExample]]:
        reader = getattr(self, f"_read_{dataset}", None)
        if reader is None:
            raise ValueError(f"no reader for dataset {dataset!r}")
        return reader()

    # --- CLINC150: config "plus", in-scope only --------------------------------------------
    def _read_clinc150(self) -> dict[str, list[RawExample]]:
        base = self._root / "clinc150"
        readme = (base / "README.md").read_text(encoding="utf-8")
        names = {int(i): n.strip("'") for i, n in re.findall(r"'(\d+)': (\S+)", readme)}
        out = {}
        for source, split in (("train", "train"), ("validation", "dev"), ("test", "test")):
            df = _parquet(str(base / "plus" / f"{source}-*.parquet"))
            rows = []
            for idx, (text, intent) in enumerate(zip(df["text"], df["intent"])):
                name = names[int(intent)]
                if name == "oos":
                    continue
                rows.append(RawExample(source, idx, text, name))
            out[split] = rows
        return out

    # --- HWU64: DeepPavlov train/test; dev carved later -------------------------------------
    def _read_hwu64(self) -> dict[str, list[RawExample]]:
        base = self._root / "hwu64"
        intents = _parquet(str(base / "intents" / "intents-*.parquet"))
        names = dict(zip(intents["id"].astype(int), intents["name"]))
        out = {}
        for split in ("train", "test"):
            df = _parquet(str(base / "data" / f"{split}-*.parquet"))
            out[split] = [
                RawExample(split, idx, text, names[int(label)])
                for idx, (text, label) in enumerate(zip(df["utterance"], df["label"]))
            ]
        return out

    # --- FPB: atrost 64/16/20 split of sentences_50agree ------------------------------------
    def _read_fpb(self) -> dict[str, list[RawExample]]:
        base = self._root / "financial_phrasebank_split" / "data"
        out = {}
        for source, split in (("train", "train"), ("validation", "dev"), ("test", "test")):
            df = _parquet(str(base / f"{source}-*.parquet"))
            out[split] = [
                RawExample(source, idx, text, str(int(label)))
                for idx, (text, label) in enumerate(zip(df["sentence"], df["label"]))
            ]
        return out

    # --- HateXplain: original files, majority of 3 annotators -------------------------------
    def _read_hatexplain(self) -> dict[str, list[RawExample]]:
        base = self._root / "hatexplain"
        data = json.loads((base / "dataset.json").read_text(encoding="utf-8"))
        divisions = json.loads((base / "post_id_divisions.json").read_text(encoding="utf-8"))

        majority: dict[str, str | None] = {}
        for pid, post in data.items():
            labels = [a["label"] for a in post["annotators"]]
            top, count = Counter(labels).most_common(1)[0]
            majority[pid] = top if count >= 2 else None
        undecided = {pid for pid, m in majority.items() if m is None}
        assert len(undecided) == _HATEXPLAIN_UNDECIDED, len(undecided)
        assert {k: len(v) for k, v in divisions.items()} == _HATEXPLAIN_SPLIT_SIZES
        assert not any(pid in undecided for ids in divisions.values() for pid in ids)

        out = {}
        for source, split in (("train", "train"), ("val", "dev"), ("test", "test")):
            rows = []
            for idx, pid in enumerate(divisions[source]):
                post = data[pid]
                platform = pid.rsplit("_", 1)[-1] if pid.rsplit("_", 1)[-1] in ("twitter", "gab") else "unknown"
                rows.append(
                    RawExample(source, idx, " ".join(post["post_tokens"]), majority[pid],
                               {"post_id": pid, "platform": platform})
                )
            out[split] = rows
        assert dict(Counter(r.label for r in out["test"])) == _HATEXPLAIN_TEST_COUNTS
        return out

    # --- TREC-6: Li & Roth original files, coarse labels; dev carved later ------------------
    def _read_trec6(self) -> dict[str, list[RawExample]]:
        base = self._root / "trec"
        out = {}
        for fname, split in (("train_5500.label", "train"), ("TREC_10.label", "test")):
            rows = []
            lines = [ln for ln in (base / fname).read_bytes().decode("latin-1").splitlines() if ln.strip()]
            for idx, line in enumerate(lines):
                tag, text = line.split(" ", 1)
                coarse, fine = tag.split(":", 1)
                rows.append(RawExample(split, idx, text.strip(), coarse, {"fine_label": f"{coarse}:{fine}"}))
            out[split] = rows
        return out
