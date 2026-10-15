"""Use case: raw -> processed (spec sec 4): dev carve-outs, cleaning rules, stable ids, manifest."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from s1bench.domain.examples import SPLITS, carve_dev, clean_splits, label_counts, to_examples
from s1bench.domain.ports import Clock, LabelRepository, ProcessedStore, RawSplitReader


@dataclass(frozen=True)
class PrepareResult:
    dataset: str
    counts: dict[str, int]
    report: dict[str, object]


class PrepareDatasets:
    def __init__(
        self,
        reader: RawSplitReader,
        labels: LabelRepository,
        store: ProcessedStore,
        clock: Clock,
        split_seed: int,
        dev_carve: Mapping[str, int],
    ) -> None:
        self._reader = reader
        self._labels = labels
        self._store = store
        self._clock = clock
        self._seed = split_seed
        self._dev_carve = dev_carve

    def execute(self, datasets: Iterable[str]) -> list[PrepareResult]:
        results = []
        for dataset in datasets:
            label_set = self._labels.load(dataset)
            raw = self._reader.read(dataset)

            if "dev" not in raw:
                size = self._dev_carve.get(dataset)
                if size is None:
                    raise ValueError(f"{dataset}: no dev split and no dev_carve size configured")
                raw["train"], raw["dev"] = carve_dev(raw["train"], size, self._seed)

            cleaned, cleaning = clean_splits(raw)
            files = {}
            for split in SPLITS:
                examples = to_examples(dataset, split, cleaned[split], label_set)
                files[split] = self._store.write_split(dataset, split, examples)

            report = {
                "dataset": dataset,
                "prepared_at": self._clock.now_iso(),
                "split_seed": self._seed,
                "dev_carved_from_train": dataset in self._dev_carve,
                "counts": {s: len(cleaned[s]) for s in SPLITS},
                "label_counts": {s: label_counts(cleaned[s]) for s in SPLITS},
                "cleaning": cleaning.as_dict(),
            }
            report_file = self._store.write_report(dataset, report)
            self._store.write_manifest({
                dataset: {
                    "prepared_at": report["prepared_at"],
                    "files": {f.path: {"sha256": f.sha256, "size_bytes": f.size_bytes}
                              for f in [*files.values(), report_file]},
                }
            })
            results.append(PrepareResult(dataset, report["counts"], report))
        return results
