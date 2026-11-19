"""Use case: build the frozen candidate files (spec sec 6) from processed data and label texts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict

from s1bench.domain.candidates import balanced_gold_positions, balanced_label_groups, build_candidate_set
from s1bench.domain.ports import CandidateStore, Clock, LabelRepository, ProcessedReader, TextEmbedder


def _label_prefix(label_id: str) -> str:
    """HWU64-style scenario prefix (e.g. 'iot' in 'iot_hue_lighton'); used only for reporting."""
    return label_id.split("_", 1)[0]


class BuildCandidateSets:
    def __init__(
        self,
        reader: ProcessedReader,
        labels: LabelRepository,
        store: CandidateStore,
        embedder_factory,  # () -> TextEmbedder; created lazily, only if a dataset needs it
        clock: Clock,
        seed: int,
        max_candidates: int,
        target_group_size: int,
        group_restarts: int,
        splits: Sequence[str],
    ) -> None:
        self._reader = reader
        self._labels = labels
        self._store = store
        self._embedder_factory = embedder_factory
        self._embedder: TextEmbedder | None = None
        self._clock = clock
        self._seed = seed
        self._k = max_candidates
        self._group_size = target_group_size
        self._restarts = group_restarts
        self._splits = list(splits)

    def _embed(self, texts: list[str]) -> tuple[list[list[float]], dict[str, str]]:
        if self._embedder is None:
            self._embedder = self._embedder_factory()
        return self._embedder.embed(texts), self._embedder.describe()

    def execute(self, datasets: Iterable[str]) -> list[dict[str, object]]:
        summaries = []
        for dataset in datasets:
            label_set = self._labels.load(dataset)
            label_ids = label_set.ids()
            files = []
            groups = None
            embedder_info: dict[str, str] = {}
            report: dict[str, object] = {
                "dataset": dataset,
                "built_at": self._clock.now_iso(),
                "n_labels": len(label_ids),
                "seed": self._seed,
                "mode": "all_labels_shuffled" if len(label_ids) <= self._k else "fixed_label_groups",
                "gold_position": "counterbalanced per split and per candidate set size (each position equally "
                                 "often, ±1)",
            }

            if len(label_ids) > self._k:
                import numpy as np

                texts = [f"{lab.name}: {lab.description}" for lab in label_set.labels]
                embeddings, embedder_info = self._embed(texts)
                groups = balanced_label_groups(label_ids, embeddings, self._group_size, self._seed, self._restarts)
                x = np.asarray(embeddings, dtype=float)
                x = x / np.linalg.norm(x, axis=1, keepdims=True)
                idx = {lid: i for i, lid in enumerate(label_ids)}
                group_of = {lid: g for g, grp in enumerate(groups) for lid in grp}
                within = [float(np.mean([x[idx[a]] @ x[idx[b]] for a in grp for b in grp if a != b])) for grp in groups]
                nn_same = np.mean([group_of[max((o for o in label_ids if o != lid), key=lambda o: x[idx[lid]] @ x[idx[o]])]
                                   == group_of[lid] for lid in label_ids])
                files.append(self._store.write_json(dataset, "label_groups", {
                    "embedder": embedder_info, "text_template": "{name}: {description}",
                    "target_group_size": self._group_size, "restarts": self._restarts, "groups": groups,
                }))
                report.update({
                    "embedder": embedder_info,
                    "n_groups": len(groups),
                    "group_sizes": [len(g) for g in groups],
                    "k": f"{min(len(g) for g in groups)}-{max(len(g) for g in groups)} (group size)",
                    "within_group_mean_cosine": [round(w, 4) for w in within],
                    "all_label_pairs_mean_cosine": round(float(((x @ x.T).sum() - len(x)) / (len(x) * (len(x) - 1))), 4),
                    "nearest_neighbour_in_same_group_share": round(float(nn_same), 4),
                    "same_prefix_share_within_groups": round(float(np.mean([
                        np.mean([_label_prefix(a) == _label_prefix(b) for b in grp if b != a])
                        for grp in groups for a in grp])), 4),
                })
            else:
                report["k"] = len(label_ids)

            per_split = {}
            for split in self._splits:
                rows = []
                positions: dict[int, Counter] = {}
                examples = self._reader.read_split(dataset, split)
                # counterbalance gold positions within each candidate-set size (= group, or all labels)
                set_of = {ex.id: (next(i for i, g in enumerate(groups) if ex.label in g) if groups else 0)
                          for ex in examples}
                sizes = {i: len(g) for i, g in enumerate(groups)} if groups else {0: len(label_ids)}
                slots: dict[str, int] = {}
                for gi, size in sizes.items():
                    slots.update(balanced_gold_positions([e.id for e in examples if set_of[e.id] == gi], size,
                                                         self._seed))
                for ex in examples:
                    cs = build_candidate_set(ex.id, ex.label, label_ids, groups, self._k, self._seed,
                                             gold_position=slots[ex.id])
                    positions.setdefault(len(cs.candidates), Counter())[cs.gold_position] += 1
                    row = asdict(cs)
                    row["candidates"] = list(cs.candidates)
                    rows.append(row)
                files.append(self._store.write(dataset, split, rows))
                dev_by_k = {}
                for size, cnt in sorted(positions.items()):
                    m = sum(cnt.values())
                    expected = m / size
                    dev_by_k[str(size)] = {"n": m, "gold_position_counts": {str(p): cnt[p] for p in range(size)},
                                           "max_abs_deviation_from_uniform": round(
                                               max(abs(cnt[p] - expected) for p in range(size)), 2)}
                per_split[split] = {"n": len(rows), "by_set_size": dev_by_k}
            report["splits"] = per_split
            files.append(self._store.write_json(dataset, "report", report))
            self._store.write_manifest({dataset: {
                "built_at": report["built_at"],
                "files": {f.path: {"sha256": f.sha256, "size_bytes": f.size_bytes} for f in files},
            }})
            summaries.append(report)
        return summaries
