"""Command-line entry point. Wires infrastructure into use cases; no logic of its own."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from s1bench.application.build_candidates import BuildCandidateSets
from s1bench.application.prepare_data import PrepareDatasets
from s1bench.application.raw_data import DownloadRawDatasets, VerifyRawDatasets
from s1bench.domain.entities import SourceKind
from s1bench.infrastructure.clock import UtcClock
from s1bench.infrastructure.config import load_dataset_sources
from s1bench.infrastructure.datasets.fetchers import HfSnapshotFetcher, UrlFetcher
from s1bench.infrastructure.hashing import Sha256Hasher
from s1bench.infrastructure.storage.json_manifest import JsonManifestStore
from s1bench.infrastructure.storage.processed_store import (
    JsonlCandidateStore,
    JsonlProcessedReader,
    JsonlProcessedStore,
    YamlLabelRepository,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="s1bench")
    parser.add_argument("--config", type=Path, default=Path("configs/datasets.yaml"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    sub = parser.add_subparsers(dest="command", required=True)

    download = sub.add_parser("download", help="Download raw datasets into data/raw/")
    download.add_argument("--dataset", action="append", help="Only this dataset id (repeatable)")
    download.add_argument("--force", action="store_true", help="Re-download even if intact")

    sub.add_parser("verify", help="Check data/raw against MANIFEST.json hashes")
    sub.add_parser("verify-frozen", help="Check frozen data/labels, data/processed, data/candidates against their manifests")

    prepare = sub.add_parser("prepare", help="Build data/processed/ from data/raw/ (spec sec 4)")
    prepare.add_argument("--dataset", action="append", help="Only this dataset id (repeatable)")
    prepare.add_argument("--processing-config", type=Path, default=Path("configs/processing.yaml"))

    pred = sub.add_parser("predict", help="Run a model on a dataset split (writes results/<run_id>/)")
    pred.add_argument("--model", required=True, help="jev, or a local model id from configs/models.yaml")
    pred.add_argument("--models-config", type=Path, default=Path("configs/models.yaml"))
    pred.add_argument("--device", default="cpu", help="cpu or cuda (local models only)")
    pred.add_argument("--dataset", required=True)
    pred.add_argument("--split", default="test", choices=["train", "dev", "test", "contentfree"],
                      help="contentfree = G4 content-free probes (needs --processed-dir and --candidates-dir "
                           "data/contentfree/...)")
    pred.add_argument("--limit", type=int, default=None, help="Seeded subset size (omit for the full split)")
    pred.add_argument("--subset-seed", type=int, default=20261006)
    pred.add_argument("--run-id", default=None, help="Default: dev_<UTC timestamp> (dev runs are never paper results)")
    pred.add_argument("--results-dir", type=Path, default=Path("results"))
    pred.add_argument("--workers", type=int, default=1, help="Parallel requests (API models; spec: 3 for Jev)")
    pred.add_argument("--candidates-dir", type=Path, default=None,
                      help="Revision protocol v1.3 (G3): read candidate sets from this folder (e.g. "
                           "data/candidates_router) and score only the examples listed there; default = frozen "
                           "data/candidates, every example")
    pred.add_argument("--processed-dir", type=Path, default=None,
                      help="Revision protocol v1.3 (G4): read examples from this folder (e.g. "
                           "data/contentfree/processed); default = frozen data/processed")
    pred.add_argument("--prompt-config", type=Path, default=None,
                      help="Revision protocol v1.3 (G5): configs/experiments/prompt_sensitivity.yaml")
    pred.add_argument("--prompt-variant", default=None,
                      help="G5: a variant of --prompt-config (local models; predictions under '<model>+<variant>')")
    pred.add_argument("--bf16-offload-gib", type=float, default=None,
                      help="Revision protocol v1.3 (G6): load the model unquantized in bf16 with CPU offload, keeping "
                           "this many GiB of weights on the GPU (predictions under '<model>+bf16')")
    pred.add_argument("--repeat-check", action="store_true",
                      help="Spec 12 determinism check: score --limit seeded examples twice, compare exactly "
                           "(writes determinism/<model>_<dataset>_<split>.json, no predictions)")
    pred.add_argument("--time-prefix-cache", action="store_true",
                      help="Spec 9 / audit C4: local models, latency with prefix caching on and off on --limit seeded "
                           "examples (writes latency/<model>_<dataset>_<split>.json, no predictions)")

    lr = sub.add_parser("train", help="Train tfidf-lr or deberta-v3-large on a label budget; predicts dev/test")
    lr.add_argument("--model", required=True, choices=["tfidf-lr", "deberta-v3-large"])
    lr.add_argument("--device", default="cpu")
    lr.add_argument("--dev-smoke", action="store_true", help="Tiny DeBERTa settings for code checks on a small GPU (never paper)")
    lr.add_argument("--smoke-base-model", action="store_true",
                    help="With --dev-smoke: use deberta-v3-base so training fits a small GPU (code check only)")
    lr.add_argument("--dataset", action="append", required=True)
    lr.add_argument("--budget", action="append", default=None, help="int or 'full' (repeatable; default full)")
    lr.add_argument("--seed", action="append", type=int, default=None,
                    help="repeatable; default first seed of configs/seeds.yaml. Run ALL seeds of a (dataset, budget) in "
                         "ONE call: the reserve-seed pool (max 3 restarts) is shared per (dataset, budget) (spec 10.3)")
    lr.add_argument("--skip-test", action="store_true", help="Pilot / dev checks: fit + dev only, never predict test")
    lr.add_argument("--lr", action="append", type=float, default=None,
                    help="DeBERTa LR grid, repeatable; default = frozen grid {1e-5, 2e-5} (spec 10.3 v1.1)")
    lr.add_argument("--run-id", required=True)
    lr.add_argument("--results-dir", type=Path, default=Path("results"))
    lr.add_argument("--realistic-config", type=Path, default=None,
                    help="Revision protocol v1.3 (e.g. configs/experiments/realistic_budget.yaml): selection inside the "
                         "budget, realistic step rule and early stopping; default = frozen Exp 3 protocol")
    lr.add_argument("--save-full-probs", action="store_true",
                    help="Revision protocol v1.3: also write test probabilities over all classes (full_probs/)")

    ql = sub.add_parser("train-qlora", help="Exp 4 (b): QLoRA task fine-tune of Qwen3.5-4B on a label budget; "
                                           "predicts dev/test with the candidate-letter readout (spec 10.3, v1.2)")
    ql.add_argument("--dataset", action="append", required=True)
    ql.add_argument("--budget", action="append", type=int, default=None,
                    help="repeatable; default = budgets of configs/experiments/qlora.yaml")
    ql.add_argument("--seed", action="append", type=int, default=None,
                    help="repeatable; default = seeds of configs/experiments/qlora.yaml. Run all seeds of a "
                         "(dataset, budget) in ONE call: the reserve-seed pool is shared per (dataset, budget)")
    ql.add_argument("--device", default="cuda")
    ql.add_argument("--qlora-config", type=Path, default=Path("configs/experiments/qlora.yaml"))
    ql.add_argument("--models-config", type=Path, default=Path("configs/models.yaml"))
    ql.add_argument("--dev-smoke", action="store_true",
                    help="Code check only (never paper): few steps, 4-bit prediction base, run id must start with dev_")
    ql.add_argument("--skip-test", action="store_true", help="Pilot / dev checks: fit + dev only, never predict test")
    ql.add_argument("--eval-every", type=int, default=None,
                    help="Pilot only (run id dev_*): override the dev evaluation spacing in optimizer steps")
    ql.add_argument("--dev-limit", type=int, default=None,
                    help="Pilot only (run id dev_*): evaluate on a seeded subset of this many dev examples")
    ql.add_argument("--dev-force-restart", action="store_true",
                    help="Pilot only (run id dev_*): count every first attempt as failed, to exercise the restart path")
    ql.add_argument("--run-id", required=True)
    ql.add_argument("--results-dir", type=Path, default=Path("results"))

    bc = sub.add_parser("budget-curve", help="Exp 3 / Exp 4: paired difference of a trained system at each label "
                                            "budget against a reference system on the test set (spec 10.3, 10.4)")
    bc.add_argument("--dataset", required=True)
    bc.add_argument("--system", required=True, help="prediction-folder prefix, e.g. deberta-v3-large, qwen3.5-4b-qlora")
    bc.add_argument("--system-run", action="append", required=True, help="run id(s) holding the system's predictions")
    bc.add_argument("--ref", required=True, help="reference model, e.g. jev, qwen3.5-4b, tev1-4b")
    bc.add_argument("--ref-run", required=True)
    bc.add_argument("--metric", action="append", choices=["macro_f1", "ece"], default=None)
    bc.add_argument("--b", type=int, default=10_000)
    bc.add_argument("--expect-budget", action="append", required=True,
                    help="every budget the curve must have (int or 'full'), repeatable; anything else is an error")
    bc.add_argument("--expect-seeds", type=int, required=True,
                    help="number of training seeds (the first N of configs/seeds.yaml) every budget must have")
    bc.add_argument("--expect-full-seeds", type=int, default=None,
                    help="seeds at 'full' if different (e.g. 1 for TF-IDF); default = --expect-seeds")
    bc.add_argument("--out-run", required=True)
    bc.add_argument("--results-dir", type=Path, default=Path("results"))

    pa = sub.add_parser("primary-pa", help="PA1-PA3 (spec 11.1) for one dataset: model vs baseline, nested bootstrap")
    pa.add_argument("--dataset", required=True)
    pa.add_argument("--model", default="jev")
    pa.add_argument("--model-run", required=True)
    pa.add_argument("--baseline", default="deberta-v3-large")
    pa.add_argument("--baseline-run", required=True)
    pa.add_argument("--out-run", required=True)
    pa.add_argument("--b", type=int, default=None, help="default exp2.yaml pa.b (10,000); other values = dev check")
    pa.add_argument("--n-priors", type=int, default=None, help="default exp2.yaml pa.n_priors (50)")
    pa.add_argument("--results-dir", type=Path, default=Path("results"))
    pa.add_argument("--exp-config", type=Path, default=Path("configs/experiments/exp2.yaml"))
    pa.add_argument("--pseudo-split", action="store_true", help="Dev check on the DEV split only (two halves)")

    pp = sub.add_parser("primary-pair", help="P1/P2 (spec 11.1) on one dataset: TDM vs its base, paired bootstrap")
    pp.add_argument("--test", required=True, choices=["P1", "P2"])
    pp.add_argument("--dataset", required=True)
    pp.add_argument("--model-run", required=True, help="run id holding the TDM predictions")
    pp.add_argument("--base-run", required=True, help="run id holding the base-model predictions")
    pp.add_argument("--out-run", required=True)
    pp.add_argument("--b", type=int, default=None, help="Override bootstrap B (dev checks only; default from config)")
    pp.add_argument("--results-dir", type=Path, default=Path("results"))
    pp.add_argument("--primary-config", type=Path, default=Path("configs/experiments/primary.yaml"))
    pp.add_argument("--pseudo-split", action="store_true", help="Dev check on the DEV split (never test)")

    ph = sub.add_parser("primary-holm", help="Holm over all primary results saved in <run>/primary/ (35-test family)")
    ph.add_argument("--run", required=True)
    ph.add_argument("--primary-config", type=Path, default=Path("configs/experiments/primary.yaml"))
    ph.add_argument("--results-dir", type=Path, default=Path("results"))

    vr = sub.add_parser("verify-results", help="Check results/<run_id>/ brought back from the run machine (spec 14)")
    vr.add_argument("--run", action="append", required=True, help="run id (repeatable; Jev version checked across runs)")
    vr.add_argument("--results-dir", type=Path, default=Path("results"))
    vr.add_argument("--freeze", type=Path, default=None,
                    help="docs/FREEZE_v1.1.json: every environment's config_snapshot must equal the frozen one (spec 16)")

    fz = sub.add_parser("freeze", help="Spec 16: write the protocol freeze file (hashes), or --check this machine")
    fz.add_argument("--out", type=Path, default=Path("docs/FREEZE_v1.1.json"))
    fz.add_argument("--version", default="v1.1")
    fz.add_argument("--check", action="store_true", help="Compare this project folder with an existing freeze file")

    ov = sub.add_parser("overlap", help="Test-set overlap with vendor-documented reference corpora (spec 4.5)")
    ov.add_argument("--references-config", type=Path, default=Path("configs/overlap_references.yaml"))
    ov.add_argument("--reference-dir", type=Path, default=Path("data/reference"))
    ov.add_argument("--processing-config", type=Path, default=Path("configs/processing.yaml"))
    ov.add_argument("--threshold", type=float, default=0.9, help="near-duplicate cosine threshold")
    ov.add_argument("--out-dir", type=Path, default=Path("data/overlap"))

    e1 = sub.add_parser("exp1", help="Exp 1 metric table from saved predictions (spec 9, 10.1, 13: blocked = wrong + abstain)")
    e1.add_argument("--run", action="append", required=True)
    e1.add_argument("--split", default="test", choices=["dev", "test"])
    e1.add_argument("--out", type=Path, required=True)
    e1.add_argument("--results-dir", type=Path, default=Path("results"))

    ef = sub.add_parser("efficiency", help="Latency / VRAM / tokens / throughput table from saved runs (spec 9)")
    ef.add_argument("--run", action="append", required=True)
    ef.add_argument("--out", type=Path, required=True, help="output JSON path")
    ef.add_argument("--results-dir", type=Path, default=Path("results"))
    ef.add_argument("--efficiency-config", type=Path, default=Path("configs/experiments/efficiency.yaml"))

    e2 = sub.add_parser("exp2", help="Exp 2 prevalence shift on saved predictions (no inference)")
    e2.add_argument("--run-id", required=True)
    e2.add_argument("--model", required=True)
    e2.add_argument("--dataset", required=True)
    e2.add_argument("--results-dir", type=Path, default=Path("results"))
    e2.add_argument("--exp-config", type=Path, default=Path("configs/experiments/exp2.yaml"))
    e2.add_argument("--draws", type=int, default=None, help="Override draws per alpha (dev checks only)")
    e2.add_argument("--eps", type=float, default=None, help="Log floor sensitivity (default 1e-6, spec 10.2)")
    e2.add_argument("--pseudo-split", action="store_true", help="Dev check on the DEV split only (two halves)")

    rc = sub.add_parser("router-candidates", help="Revision protocol v1.3 (G3): candidate sets of a learned intent-group "
                                                  "router for the misrouted test examples (CLINC150, HWU64)")
    rc.add_argument("--dataset", action="append", default=None, help="default: clinc150 and hwu64")
    rc.add_argument("--out", type=Path, default=Path("data/candidates_router"))
    cf = sub.add_parser("contentfree-probes", help="Revision protocol v1.3 (G4): content-free probe files "
                                                   "(Zhao et al. 2021) for content-free calibration")
    cf.add_argument("--dataset", action="append", default=None, help="default: all five datasets")
    cf.add_argument("--out", type=Path, default=Path("data/contentfree"))
    cands = sub.add_parser("candidates", help="Build frozen candidate files in data/candidates/ (spec sec 6)")
    cands.add_argument("--dataset", action="append", help="Only this dataset id (repeatable)")
    cands.add_argument("--candidates-config", type=Path, default=Path("configs/candidates.yaml"))
    return parser


NL = chr(10)


def _pseudo_halves(ids: list[str]) -> tuple[set[str], set[str]]:
    """Dev checks never touch the test split (audit 2026-10-07): the DEV split is cut into two seeded halves
    that play 'dev' (fitting) and 'test' (evaluation)."""
    import hashlib

    ranked = sorted(ids, key=lambda i: hashlib.sha256(f"pseudo:{i}".encode("utf-8")).digest())
    half = len(ranked) // 2
    return set(ranked[:half]), set(ranked[half:])


def _fmt_p(p: float, b: int | None) -> str:
    """spec 11.2: a two-sided bootstrap p of 0 is reported as '< 2/B' (smallest non-zero value of the formula)."""
    if b and p < 2 / b:
        return f"< {2 / b:g}"
    return f"{p:.4f}"


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    hasher = Sha256Hasher()
    manifest = JsonManifestStore(args.data_dir / "MANIFEST.json")
    raw_root = args.data_dir / "raw"

    if args.command == "download":
        sources = load_dataset_sources(args.config)
        if args.dataset:
            known = {s.dataset_id for s in sources}
            unknown = sorted(set(args.dataset) - known)
            if unknown:
                print(f"Unknown dataset id(s): {', '.join(unknown)}. Known: {', '.join(sorted(known))}")
                return 2
            sources = [s for s in sources if s.dataset_id in set(args.dataset)]
        use_case = DownloadRawDatasets(
            fetchers={SourceKind.HF_SNAPSHOT: HfSnapshotFetcher(hasher), SourceKind.URL: UrlFetcher(hasher)},
            manifest=manifest,
            hasher=hasher,
            clock=UtcClock(),
            raw_root=raw_root,
        )
        for result in use_case.execute(sources, force=args.force):
            r = result.record
            print(f"{r.dataset_id:22s} {result.status.value:11s} {len(r.files):3d} files {r.total_bytes():>12,d} bytes")
        return 0

    if args.command == "verify":
        problems = VerifyRawDatasets(manifest, hasher, raw_root).execute()
        for p in problems:
            print(f"{p.dataset_id}: {p.path}: {p.problem}")
        print("OK: all raw files match MANIFEST.json" if not problems else f"{len(problems)} problem(s)")
        return 0 if not problems else 1

    if args.command == "verify-frozen":
        import json

        problems, n = [], 0
        labels_m = json.loads((args.data_dir / "labels" / "MANIFEST.json").read_text(encoding="utf-8"))
        for name, sha in labels_m["files"].items():
            n += 1
            p = args.data_dir / "labels" / name
            if not p.exists() or hasher.sha256(p) != sha:
                problems.append(f"labels/{name}: {'missing' if not p.exists() else 'sha256 differs'}")
        for part in ("processed", "candidates"):
            m = json.loads((args.data_dir / part / "MANIFEST.json").read_text(encoding="utf-8"))
            for dataset, rec in m["datasets"].items():
                for rel, meta in rec["files"].items():
                    n += 1
                    p = args.data_dir / part / rel
                    if not p.exists() or hasher.sha256(p) != meta["sha256"]:
                        problems.append(f"{part}/{rel}: {'missing' if not p.exists() else 'sha256 differs'}")
        for p in problems:
            print("  " + p)
        print(f"OK: {n} frozen files match their manifests" if not problems else f"{len(problems)} problem(s) in {n} files")
        return 0 if not problems else 1

    if args.command == "prepare":
        import yaml

        from s1bench.infrastructure.datasets.readers import RawDatasetReaders

        cfg = yaml.safe_load(args.processing_config.read_text(encoding="utf-8"))
        datasets = args.dataset or cfg["datasets"]
        unknown = sorted(set(datasets) - set(cfg["datasets"]))
        if unknown:
            print(f"Unknown dataset id(s): {', '.join(unknown)}. Known: {', '.join(cfg['datasets'])}")
            return 2
        use_case = PrepareDatasets(
            reader=RawDatasetReaders(raw_root),
            labels=YamlLabelRepository(args.data_dir / "labels"),
            store=JsonlProcessedStore(args.data_dir / "processed", hasher),
            clock=UtcClock(),
            split_seed=int(cfg["split_seed"]),
            dev_carve={k: int(v) for k, v in (cfg.get("dev_carve") or {}).items()},
        )
        for r in use_case.execute(datasets):
            c = r.report["cleaning"]
            print(f"{r.dataset:11s} train {r.counts['train']:6,d}  dev {r.counts['dev']:5,d}  test {r.counts['test']:5,d}"
                  f"  | removed: test-overlap {sum(c['removed_overlap_with_test'].values())}, "
                  f"conflicting {sum(c['removed_conflicting'].values())}, dev-overlap {c['removed_train_overlap_with_dev']}, "
                  f"dup {c['removed_same_label_duplicates_train']}, test-conflicts {c['conflicting_in_test']}")
        return 0

    if args.command == "predict":
        import hashlib
        from datetime import datetime, timezone

        from s1bench.application.run_predictions import RunPredictions
        from s1bench.infrastructure.secrets import load_env

        if (args.split == "contentfree") != (args.processed_dir is not None) or (
                args.split == "contentfree" and args.candidates_dir is None):
            print("--split contentfree goes with --processed-dir data/contentfree/processed and --candidates-dir "
                  "data/contentfree/candidates (and only then)")
            return 2
        if args.model == "jev" and (args.prompt_config is not None or args.prompt_variant is not None):
            print("--prompt-config/--prompt-variant are for local models only")
            return 2
        from s1bench.infrastructure.storage.results_store import (
            JsonlCandidateReader,
            JsonlPredictionStore,
            capture_environment,
            config_snapshot,
        )

        if args.model == "jev":
            from s1bench.infrastructure.models.jev_cloudflare import JevCloudflareModel

            env = load_env()
            missing = [k for k in ("CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN") if not env.get(k)]
            if missing:
                print(f"Missing in .env: {', '.join(missing)}")
                return 2
            model = JevCloudflareModel(env["CLOUDFLARE_ACCOUNT_ID"], env["CLOUDFLARE_API_TOKEN"])
        else:
            import yaml

            from s1bench.infrastructure.models.hf_letter_scorer import HfLetterScoringModel

            mcfg = yaml.safe_load(args.models_config.read_text(encoding="utf-8"))
            if args.model not in mcfg["models"]:
                print(f"Unknown model {args.model!r}. Known: jev, {', '.join(mcfg['models'])}")
                return 2
            m = mcfg["models"][args.model]
            name, fmt, fmt_name = args.model, mcfg["prompt_formats"][m["prompt_format"]], m["prompt_format"]
            if (args.prompt_config is None) != (args.prompt_variant is None):
                print("--prompt-config and --prompt-variant go together")
                return 2
            if args.prompt_config is not None:
                pcfg = yaml.safe_load(args.prompt_config.read_text(encoding="utf-8"))
                if args.prompt_variant not in pcfg["variants"]:
                    print(f"Unknown variant {args.prompt_variant!r}. Known: {', '.join(pcfg['variants'])}")
                    return 2
                var = pcfg["variants"][args.prompt_variant]
                name, fmt_name = f"{args.model}+{args.prompt_variant}", var["format"]
                fmt = {"system": var["system"], "expected_prefix": var["expected_prefix"]}
            if args.bf16_offload_gib is not None:
                if m.get("dtype", "bfloat16") != "bfloat16":
                    print("--bf16-offload-gib needs a bfloat16 model")
                    return 2
                name = f"{name}+bf16"
            model = HfLetterScoringModel(
                name=name, repo=m["repo"], revision=m["revision"],
                prompt_format=fmt, format_name=fmt_name,
                dtype=m.get("dtype", "bfloat16"), device=args.device,
                adapter_repo=m.get("adapter_repo"), adapter_revision=m.get("adapter_revision"),
                quantization=None if args.bf16_offload_gib is not None else m.get("quantization"),
                offload_gpu_gib=args.bf16_offload_gib,
            )

        run_id = args.run_id or "dev_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        if args.repeat_check:
            from s1bench.application.run_predictions import repeat_check

            if not args.limit:
                print("--repeat-check needs --limit (spec 12: 50 examples)")
                return 2
            rep = repeat_check(model, JsonlProcessedReader(args.data_dir / "processed"),
                               JsonlCandidateReader(args.data_dir / "candidates"),
                               YamlLabelRepository(args.data_dir / "labels"), args.dataset, args.split,
                               args.limit, args.subset_seed)
            rep["environment"] = capture_environment({"model": model.describe(),
                                                      "config_snapshot": config_snapshot(Path("."), hasher)})
            JsonlPredictionStore(args.results_dir / run_id, hasher).write_json(
                f"determinism/{args.model}_{args.dataset}_{args.split}.json", rep)
            print(f"{args.model} {args.dataset}/{args.split}: n={rep['n']} deterministic={rep['deterministic']} "
                  f"(not identical {rep['n_not_identical']}, max |dp| {rep['max_abs_prob_difference']:.2e})")
            return 0 if rep["deterministic"] else 1
        if args.time_prefix_cache:
            from s1bench.application.run_predictions import prefix_cache_timing

            if not args.limit:
                print("--time-prefix-cache needs --limit (spec 9: 210 = 10 warm-up + 200)")
                return 2
            rep = prefix_cache_timing(model, JsonlProcessedReader(args.data_dir / "processed"),
                                      JsonlCandidateReader(args.data_dir / "candidates"),
                                      YamlLabelRepository(args.data_dir / "labels"), args.dataset, args.split,
                                      args.limit, args.subset_seed)
            rep["environment"] = capture_environment({"model": model.describe(),
                                                      "config_snapshot": config_snapshot(Path("."), hasher)})
            JsonlPredictionStore(args.results_dir / run_id, hasher).write_json(
                f"latency/{args.model}_{args.dataset}_{args.split}.json", rep)
            print(f"{args.model} {args.dataset}/{args.split}: prefix {rep['prefix_tokens']}/{rep['mean_input_tokens']} "
                  f"tokens; p50 off {rep['cache_off']['p50_ms']} ms, on {rep['cache_on']['p50_ms']} ms; "
                  f"max |dp| on vs off {rep['max_abs_prob_difference_on_vs_off']:.1e}")
            return 0
        run_dir = args.results_dir / run_id
        store = JsonlPredictionStore(run_dir, hasher)
        use_case = RunPredictions(
            model=model,
            examples=JsonlProcessedReader(args.processed_dir or args.data_dir / "processed"),
            candidates=JsonlCandidateReader(args.candidates_dir or args.data_dir / "candidates"),
            labels=YamlLabelRepository(args.data_dir / "labels"),
            store=store,
            progress=lambda i, n: print(f"  {i}/{n}", end=chr(13), flush=True),
            workers=args.workers,
            only_listed=args.candidates_dir is not None,
        )
        started = UtcClock().now_iso()
        summary = use_case.execute(args.dataset, args.split, args.limit, args.subset_seed)
        store.write_environment(capture_environment({
            "run_id": run_id, "started_at": started, "finished_at": UtcClock().now_iso(),
            "model": model.describe(), "dataset": args.dataset, "split": args.split,
            "limit": args.limit, "subset_seed": args.subset_seed, "workers": args.workers,
            **({"processed_dir": args.processed_dir.as_posix(),
                "processed_file_sha256": hashlib.sha256((args.processed_dir / args.dataset / f"{args.split}.jsonl")
                                                        .read_bytes()).hexdigest()}
               if args.processed_dir is not None else {}),
            **({"candidates_dir": args.candidates_dir.as_posix(),
                "candidates_file_sha256": hashlib.sha256((args.candidates_dir / args.dataset / f"{args.split}.jsonl")
                                                         .read_bytes()).hexdigest()}
               if args.candidates_dir is not None else {}),
            "model_versions_seen": summary.model_versions,
            "config_snapshot": config_snapshot(Path("."), hasher),
        }), name=f"{model.name}/{args.dataset}_{args.split}")
        acc = f"{summary.accuracy_ok:.3f}" if summary.accuracy_ok is not None else "n/a"
        lat = f"{summary.mean_latency_ms:.0f} ms" if summary.mean_latency_ms is not None else "n/a"
        print(f"{summary.model} {summary.dataset}/{summary.split}: n={summary.n} ok={summary.n_ok} "
              f"statuses={summary.statuses} acc(ok)={acc} mean latency={lat} "
              f"input_tokens={summary.input_tokens} versions={summary.model_versions} -> {run_dir}")
        return 0

    if args.command == "train":
        import json
        from datetime import datetime, timezone

        from s1bench.application.train_classifier import TrainAndPredict
        from s1bench.infrastructure.models.deberta_classifier import DebertaClassifier
        from s1bench.infrastructure.models.tfidf_lr import TfidfLogReg
        from s1bench.infrastructure.storage.results_store import (
            JsonlCandidateReader,
            JsonlPredictionStore,
            capture_environment,
            config_snapshot,
        )

        import yaml

        seeds_cfg = yaml.safe_load(Path("configs/seeds.yaml").read_text(encoding="utf-8"))
        run_dir = args.results_dir / args.run_id
        store = JsonlPredictionStore(run_dir, hasher)
        rcfg = (yaml.safe_load(args.realistic_config.read_text(encoding="utf-8"))
                if args.realistic_config is not None else None)
        rkw = {"max_steps": rcfg["max_steps"], "patience": rcfg["patience"]} if rcfg else {}
        if args.model == "tfidf-lr":
            factory, grid, describe = (lambda c, seed: TfidfLogReg(c, seed)), list(TfidfLogReg.C_GRID), TfidfLogReg.describe()
        else:
            smoke = dict(min_steps=10, eval_every=5, max_length=64, epochs_small=1, epochs_large=1) if args.dev_smoke else {}
            if args.smoke_base_model:
                if not args.dev_smoke:
                    print("--smoke-base-model is only allowed with --dev-smoke")
                    return 2
                smoke.update(repo="microsoft/deberta-v3-base", revision="8ccc9b6f36199bec6961081d44eb72fb3f7353f3")
            factory = lambda lr_, seed: DebertaClassifier(seed, lr=lr_, device=args.device, **smoke, **rkw)  # noqa: E731
            if args.dev_smoke and not args.run_id.startswith("dev_"):
                print("--dev-smoke runs must use a run id starting with dev_ (never paper results)")
                return 2
            if not args.dev_smoke and not args.device.startswith("cuda"):
                print("DeBERTa real runs need --device cuda (fp16 mixed precision; CPU only for --dev-smoke)")
                return 2
            grid = args.lr or [1e-5, 2e-5]   # spec 10.3 v1.1: frozen LR grid
            describe = {"name": "deberta-v3-large", "lr_grid": grid, "dev_smoke": args.dev_smoke,
                        "hparams": DebertaClassifier(0, lr=grid[0], device=args.device, **smoke, **rkw).hparams()}
        if rcfg:
            import hashlib

            describe = {**describe, "realistic_config": rcfg, "realistic_config_path": args.realistic_config.as_posix(),
                        "realistic_config_sha256": hashlib.sha256(
                            args.realistic_config.read_bytes().replace(b"\r\n", b"\n")).hexdigest()}
        use_case = TrainAndPredict(
            factory=factory, c_grid=grid,
            examples=JsonlProcessedReader(args.data_dir / "processed"),
            candidates=JsonlCandidateReader(args.data_dir / "candidates"),
            labels=YamlLabelRepository(args.data_dir / "labels"), store=store,
            reserve_seeds=seeds_cfg["training"]["reserve_seeds"],
            dev_share=rcfg["dev_share"] if rcfg else None, save_full_probs=args.save_full_probs,
        )
        budgets = [None if b == "full" else int(b) for b in (args.budget or ["full"])]
        seeds = args.seed or [seeds_cfg["training"]["seeds"][0]]
        summaries = []
        for dataset in args.dataset:
            for budget in budgets:
                # TF-IDF + LR is deterministic: one fit at full data. DeBERTa: every seed at every budget incl.
                # full (crossover CIs resample seeds); the FIRST seed's full-data model is the Exp 2 baseline.
                for seed in (seeds if budget is not None or args.model != "tfidf-lr" else seeds[:1]):
                    if budget is None:
                        name = args.model if seed == seeds[0] else f"{args.model}@fulls{seed}"
                    else:
                        name = f"{args.model}@b{budget}s{seed}"
                    r = use_case.execute(name, dataset, budget, seed, predict_test=not args.skip_test)
                    summaries.append(r)
                    hp = "C" if args.model == "tfidf-lr" else "LR"
                    print(f"{dataset:11s} {name:22s} n_train={r['n_train']:6d} classes={r['classes_in_train']:3d} "
                          f"{hp}={r['selected_c']:<4} dev F1={r['dev_macro_f1']:.3f} (test score only in train_summary.json)")
        summary_path = run_dir / "train_summary.json"
        old = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else []
        store.write_json("train_summary.json", old + summaries)
        store.write_environment(capture_environment({
            "run_id": args.run_id, "model": describe, "datasets": args.dataset, "budgets": args.budget or ["full"],
            "seeds": seeds, "config_snapshot": config_snapshot(Path("."), hasher),
        }), name=f"{args.model}/train_{'_'.join(args.dataset)}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}")
        return 0

    if args.command == "train-qlora":
        import hashlib
        import json
        from datetime import datetime, timezone

        import yaml

        from s1bench.application.train_letter_model import TrainLetterModel, restarts_used
        from s1bench.domain.predictions import seeded_subset
        from s1bench.infrastructure.models.qlora_letter import QloraLetterClassifier
        from s1bench.infrastructure.storage.results_store import (
            JsonlCandidateReader,
            JsonlPredictionStore,
            capture_environment,
            config_snapshot,
        )

        pilot = (args.dev_smoke or args.eval_every is not None or args.dev_limit is not None
                 or args.dev_force_restart)
        if pilot and not args.run_id.startswith("dev_"):
            print("--dev-smoke, --eval-every, --dev-limit and --dev-force-restart are pilot options: the run id must "
                  "start with dev_")
            return 2
        if pilot and not args.skip_test:
            print("pilot options need --skip-test (pilots never touch the test split)")
            return 2
        if not args.device.startswith("cuda"):
            print("train-qlora needs --device cuda (4-bit quantization)")
            return 2
        qcfg = yaml.safe_load(args.qlora_config.read_text(encoding="utf-8"))
        mcfg = yaml.safe_load(args.models_config.read_text(encoding="utf-8"))
        seeds_cfg = yaml.safe_load(Path("configs/seeds.yaml").read_text(encoding="utf-8"))
        base_key = qcfg["base_model"]
        m = mcfg["models"][base_key]
        prompt_format = mcfg["prompt_formats"][m["prompt_format"]]
        smoke = dict(min_steps=4, eval_every=2, epochs_small=1, epochs_large=1) if args.dev_smoke else {}
        if args.eval_every is not None:
            smoke["eval_every"] = args.eval_every
        if args.dev_smoke:
            qcfg = {**qcfg, "predict_base": "4bit"}
        run_dir = args.results_dir / args.run_id
        store = JsonlPredictionStore(run_dir, hasher)
        model_name = f"{base_key}-qlora"
        budgets = args.budget or qcfg["budgets"]
        seeds = args.seed or qcfg["seeds"]
        summary_path = run_dir / "train_summary.json"
        old = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else []
        # Before any training: a (model, dataset) that is already done is never retrained or overwritten; adapters
        # left by a call that crashed before its summary was written are moved aside (kept, not reused).
        done = {(str(s.get("model")), str(s.get("dataset"))) for s in old}
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        refuse, stale = [], []
        for dataset in args.dataset:
            for budget in budgets:
                for seed in seeds:
                    name = f"{model_name}@b{budget}s{seed}"
                    files = [run_dir / "predictions" / name / f"{dataset}_{split}.jsonl" for split in ("dev", "test")]
                    if (name, dataset) in done or any(f.exists() for f in files):
                        refuse.append(f"{dataset} {name}")
                    stale += sorted((run_dir / "adapters" / dataset).glob(f"{name}__seed*"))
        if refuse:
            print("already in " + str(summary_path) + " or has predictions (drop these seeds or use a new run id):"
                  + NL + NL.join("  " + r for r in refuse))
            return 2
        for path in stale:
            target = run_dir / "adapters_crashed" / stamp / path.parent.name / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            path.rename(target)
            print(f"moved adapter of an unfinished earlier call to {target}")
        sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()  # noqa: E731
        store.write_environment(capture_environment({    # at the start, so a call that crashes still has its record
            "run_id": args.run_id, "model": {"name": model_name, "base": base_key, "qlora_config": qcfg},
            "datasets": args.dataset, "budgets": budgets, "seeds": seeds, "pilot": pilot,
            "pilot_options": {"dev_smoke": args.dev_smoke, "eval_every": args.eval_every, "dev_limit": args.dev_limit,
                              "dev_force_restart": args.dev_force_restart},
            "qlora_config_sha256": sha(args.qlora_config),
            "code:src/s1bench/application/train_letter_model.py": sha("src/s1bench/application/train_letter_model.py"),
            "code:src/s1bench/infrastructure/models/qlora_letter.py": sha("src/s1bench/infrastructure/models/qlora_letter.py"),
            "config_snapshot": config_snapshot(Path("."), hasher),
        }), name=f"{model_name}/train_{'_'.join(args.dataset)}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}")

        class _DevLimited(JsonlProcessedReader):
            def read_split(self, dataset, split):
                rows = super().read_split(dataset, split)
                if split != "dev" or args.dev_limit is None:
                    return rows
                keep = set(seeded_subset([e.id for e in rows], min(args.dev_limit, len(rows)),
                                         seeds_cfg["data"]["subset_seed"]))
                return [e for e in rows if e.id in keep]

        use_case = TrainLetterModel(
            factory=lambda s, n: QloraLetterClassifier(s, n, m, prompt_format, qcfg, device=args.device,
                                                       smoke=smoke or None, adapter_root=run_dir / "adapters"),
            examples=_DevLimited(args.data_dir / "processed"),
            candidates=JsonlCandidateReader(args.data_dir / "candidates"),
            labels=YamlLabelRepository(args.data_dir / "labels"), store=store,
            reserve_seeds=seeds_cfg["training"]["reserve_seeds"], reserve_used=restarts_used(old, model_name),
            force_fail_first=args.dev_force_restart, dev_share=qcfg.get("dev_share"))
        summaries = []
        for dataset in args.dataset:
            for budget in budgets:
                for seed in seeds:
                    name = f"{model_name}@b{budget}s{seed}"
                    r = use_case.execute(name, dataset, budget, seed, predict_test=not args.skip_test)
                    summaries.append(r)
                    store.write_json("train_summary.json", old + summaries)   # after every seed
                    log = r["train_log"]
                    print(f"{dataset:11s} {name:28s} n_train={r['n_train']:5d} steps={log.get('total_steps')} "
                          f"best_step={log.get('best_step')} dev F1 4-bit={log.get('best_dev_macro_f1_4bit')} "
                          f"deployed={log.get('dev_macro_f1_deployed')} restarts={len(r['restarts'])} "
                          f"(test score only in train_summary.json)")
        return 0

    if args.command == "budget-curve":
        import json

        import yaml

        from s1bench.application.budget_curves import align, crossover, curve_difference, parse_model_name
        from s1bench.infrastructure.storage.results_store import read_predictions

        runs = [*args.system_run, args.ref_run, args.out_run]
        if any(r.startswith("dev_") for r in runs):
            print("budget-curve uses final runs only (no dev_* run ids)")
            return 2
        if args.out_run in (*args.system_run, args.ref_run):
            print("--out-run must be a separate analysis run id, not one of the input runs")
            return 2
        out_dir = args.results_dir / args.out_run / "curves"
        path = out_dir / f"{args.system}_vs_{args.ref}@{args.ref_run}_{args.dataset}.json"
        if path.exists():
            print(f"{path} already exists (never overwritten)")
            return 2
        seeds_cfg = yaml.safe_load(Path("configs/seeds.yaml").read_text(encoding="utf-8"))
        first_seed = seeds_cfg["training"]["seeds"][0]
        labels = YamlLabelRepository(args.data_dir / "labels").load(args.dataset)
        points: dict[object, dict[int, list]] = {}
        sources: dict[str, str] = {}
        for run in args.system_run:
            pred_dir = args.results_dir / run / "predictions"
            for folder in sorted(pred_dir.iterdir()) if pred_dir.exists() else []:
                key = parse_model_name(folder.name, args.system, first_seed)
                if key is None or not (folder / f"{args.dataset}_test.jsonl").exists():
                    continue
                budget, seed = key
                if seed in points.get(budget, {}):
                    print(f"duplicate predictions for budget {budget} seed {seed} ({sources[f'{budget}/{seed}']}, {run})")
                    return 2
                points.setdefault(budget, {})[seed] = read_predictions(args.results_dir / run, folder.name,
                                                                       args.dataset, "test")
                sources[f"{budget}/{seed}"] = f"{run}/{folder.name}"
        if not points:
            print(f"no test predictions for {args.system} on {args.dataset} in {args.system_run}")
            return 2
        # the curve must be complete: exactly the expected budgets, each with the first N training seeds
        want = {("full" if b == "full" else int(b)): seeds_cfg["training"]["seeds"][
            :(args.expect_full_seeds or args.expect_seeds) if b == "full" else args.expect_seeds]
            for b in args.expect_budget}
        found = {p: sorted(points[p]) for p in points}
        if found != want:
            print(f"incomplete or unexpected curve for {args.system} on {args.dataset}:" + NL
                  + f"  expected {want}" + NL + f"  found    {found}")
            return 2
        n_train = len(JsonlProcessedReader(args.data_dir / "processed").read_split(args.dataset, "train"))
        sizes = {p: (n_train if p == "full" else int(p)) for p in points}
        order = sorted(points, key=lambda p: sizes[p])
        aligned = align(points, read_predictions(args.results_dir / args.ref_run, args.ref, args.dataset, "test"),
                        labels)
        boot_seed = seeds_cfg["analysis"]["bootstrap_seed"]
        out = {"dataset": args.dataset, "system": args.system, "system_runs": args.system_run, "reference": args.ref,
               "reference_run": args.ref_run, "b": args.b, "bootstrap_seed": boot_seed, "n_test": len(aligned.ids),
               "sizes": {str(p): sizes[p] for p in order}, "sources": sources, "curves": {}}
        for metric in args.metric or ["macro_f1", "ece"]:
            rows = curve_difference(aligned, order, metric, b=args.b, seed=boot_seed)
            out["curves"][metric] = {"rows": rows}
            if metric == "macro_f1":
                out["curves"][metric]["crossover"] = crossover(rows, sizes)
            for r in rows:
                print(f"{args.dataset:11s} {args.system:20s} b={str(r['budget']):>5s} seeds={r['n_seeds']} {metric:8s} "
                      f"delta={r['estimate']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]")
        out_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
        if "macro_f1" in out["curves"]:
            print("crossover:", out["curves"]["macro_f1"]["crossover"])
        print("wrote", path)
        return 0

    if args.command == "primary-pa":
        import json

        import numpy as np
        import yaml

        from s1bench.application.primary_tests import aligned_matrices, input_check, pa_tests, with_holm
        from s1bench.infrastructure.storage.results_store import read_predictions

        cfg = yaml.safe_load(args.exp_config.read_text(encoding="utf-8"))
        labels = YamlLabelRepository(args.data_dir / "labels").load(args.dataset)
        reader = JsonlProcessedReader(args.data_dir / "processed")
        loaded, checks = {}, {}
        if args.pseudo_split:
            dev_ids = [e.id for e in reader.read_split(args.dataset, "dev")]
            halves = dict(zip(("dev", "test"), _pseudo_halves(dev_ids)))
            for who, run, model in (("model", args.model_run, args.model), ("baseline", args.baseline_run, args.baseline)):
                preds = read_predictions(args.results_dir / run, model, args.dataset, "dev")
                for split, keep in halves.items():
                    loaded[(who, split)] = [p for p in preds if p.example_id in keep]
                    checks[f"{who}_{split}"] = input_check(loaded[(who, split)], sorted(keep))
        else:
            for split in ("dev", "test"):
                ids = [e.id for e in reader.read_split(args.dataset, split)]
                for who, run, model in (("model", args.model_run, args.model), ("baseline", args.baseline_run, args.baseline)):
                    preds = read_predictions(args.results_dir / run, model, args.dataset, split)
                    loaded[(who, split)] = preds
                    checks[f"{who}_{split}"] = input_check(preds, ids)
        problems = [f"{k}: {p}" for k, c in checks.items() for p in c["problems"]]
        pa_cfg = cfg["pa"]
        args.b = args.b or int(pa_cfg["b"])
        args.n_priors = args.n_priors or int(pa_cfg["n_priors"])
        dev_check = (args.pseudo_split or args.b != int(pa_cfg["b"]) or args.n_priors != int(pa_cfg["n_priors"])
                     or args.model != "jev"
                     or args.baseline != "deberta-v3-large"
                     or args.model_run.startswith("dev_") or args.baseline_run.startswith("dev_"))
        if problems and not dev_check:
            print("Refusing primary test (spec sec 13):" + NL + NL.join("  " + p for p in problems))
            return 1
        a_dev, b_dev, drop_dev = aligned_matrices(loaded[("model", "dev")], loaded[("baseline", "dev")], labels)
        a_test, b_test, drop_test = aligned_matrices(loaded[("model", "test")], loaded[("baseline", "test")], labels)
        report = json.loads((args.data_dir / "processed" / args.dataset / "report.json").read_text(encoding="utf-8"))
        counts = report["label_counts"]["train"]
        train_prior = np.array([counts.get(lid, 0) for lid in labels.ids()], dtype=float)
        train_prior /= train_prior.sum()
        kind = cfg["source_prior"].get(args.model, cfg["source_prior"]["default"])
        results = pa_tests(args.dataset, a_dev, a_test, b_dev, b_test, train_prior, kind,
                           coverage=cfg["coverage"], b=args.b, seed=cfg["seed"], n_priors=args.n_priors)
        rows = with_holm(results)
        out = args.results_dir / args.out_run / "primary"
        out.mkdir(parents=True, exist_ok=True)
        payload = {"dataset": args.dataset, "model": args.model, "baseline": args.baseline,
                   "model_run": args.model_run, "baseline_run": args.baseline_run, "b": args.b,
                   "n_priors": args.n_priors, "dev_check": dev_check, "pseudo_split": args.pseudo_split,
                   "input_checks": checks,
                   "dropped_dev": drop_dev, "dropped_test": drop_test, "n_test": int(len(a_test.gold)),
                   "source_prior_kind": kind, "results": rows}
        (out / f"pa_{args.model}_vs_{args.baseline}_{args.dataset}.json").write_text(
            json.dumps(payload, indent=2) + NL, encoding="utf-8")
        print(f"{args.model} vs {args.baseline} on {args.dataset} (n_test={payload['n_test']}, B={args.b}, "
              f"dropped dev/test={drop_dev}/{drop_test}{', DEV CHECK' if dev_check else ''})")
        for c, v in checks.items():
            print(f"  input {c}: {v['status']}" + (f"  PROBLEMS: {v['problems']}" if v["problems"] else ""))
        for r in rows:
            print(f"  {r['test']}: estimate {r['estimate']:+.4f}  95% CI [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]  "
                  f"p={_fmt_p(r['p_value'], args.b)}")
        return 0

    if args.command == "primary-pair":
        import json

        import yaml

        from s1bench.application.primary_tests import aligned_matrices, input_check, pair_tests
        from s1bench.infrastructure.storage.results_store import read_predictions

        cfg = yaml.safe_load(args.primary_config.read_text(encoding="utf-8"))
        if args.dataset not in cfg["datasets"]:
            print(f"{args.dataset!r} is not a preregistered dataset ({cfg['datasets']})")
            return 2
        pair = cfg["pairs"][args.test]
        b = args.b or int(cfg["bootstrap_b"])
        labels = YamlLabelRepository(args.data_dir / "labels").load(args.dataset)
        eval_split = "dev" if args.pseudo_split else "test"
        ids = [e.id for e in JsonlProcessedReader(args.data_dir / "processed").read_split(args.dataset, eval_split)]
        pa_ = read_predictions(args.results_dir / args.model_run, pair["model"], args.dataset, eval_split)
        pb_ = read_predictions(args.results_dir / args.base_run, pair["base"], args.dataset, eval_split)
        checks = {"model": input_check(pa_, ids), "base": input_check(pb_, ids)}
        problems = [f"{k}: {p}" for k, c in checks.items() for p in c["problems"]]
        dev_check = (args.pseudo_split or b != int(cfg["bootstrap_b"]) or args.model_run.startswith("dev_")
                     or args.base_run.startswith("dev_"))
        if problems and not dev_check:
            print("Refusing primary test (spec sec 13):" + NL + NL.join("  " + p for p in problems))
            return 1
        a_test, b_test, dropped = aligned_matrices(pa_, pb_, labels)
        if len(a_test.gold) == 0:
            print("No example is ok for both models")
            return 1
        results = pair_tests(args.test, args.dataset, a_test, b_test, b=b, seed=int(cfg["seed"]))
        out = args.results_dir / args.out_run / "primary"
        out.mkdir(parents=True, exist_ok=True)
        payload = {"test": args.test, "dataset": args.dataset, "model": pair["model"], "base": pair["base"],
                   "model_run": args.model_run, "base_run": args.base_run, "dropped_test": dropped,
                   "input_checks": checks, "n_test": int(len(a_test.gold)), "b": b, "dev_check": dev_check,
                   "evaluated_split": eval_split,
                   "results": [r.__dict__ for r in results]}
        (out / f"{args.test.lower()}_{args.dataset}.json").write_text(json.dumps(payload, indent=2) + NL, encoding="utf-8")
        print(f"{args.test} {pair['model']} vs {pair['base']} on {args.dataset} (n_test={payload['n_test']}, B={b}, "
              f"dropped={dropped}{', DEV CHECK' if dev_check else ''})")
        for c, v in checks.items():
            print(f"  input {c}: {v['status']}" + (f"  PROBLEMS: {v['problems']}" if v["problems"] else ""))
        for r in results:
            print(f"  {r.test}: delta {r.estimate:+.4f}  95% CI [{r.ci_low:+.4f}, {r.ci_high:+.4f}]  p={_fmt_p(r.p_value, b)}")
        return 0

    if args.command == "primary-holm":
        import json

        import yaml

        from s1bench.application.primary_tests import expected_family_keys, holm_family

        cfg = yaml.safe_load(args.primary_config.read_text(encoding="utf-8"))
        expected = expected_family_keys(cfg["datasets"], list(cfg["pairs"]))
        folder = args.results_dir / args.run / "primary"
        rows = []
        for f in sorted(folder.glob("*.json")):
            if f.name == "holm_family.json":
                continue
            data = json.loads(f.read_text(encoding="utf-8"))
            if f.name.startswith("pa_") and (data.get("model") != "jev" or data.get("baseline") != "deberta-v3-large"):
                data["dev_check"] = True   # PA family is defined as jev vs deberta-v3-large (spec 11.1)
            if "dev_check" not in data:
                data["dev_check"] = True   # files written before b/dev_check were recorded
            rows += [{**r, "source_file": f.name, "dev_check": bool(data["dev_check"]),
                      "b": data.get("b")} for r in data["results"]]
        try:
            family, status = holm_family(rows, expected)
        except ValueError as e:
            print(f"ERROR {e}")
            return 1
        (folder / "holm_family.json").write_text(json.dumps({**status, "results": family}, indent=2) + NL,
                                                 encoding="utf-8")
        for r in sorted(family, key=lambda r: (r["test"], r["dataset"])):
            print(f"  {r['test']:7s} {r['dataset']:11s} est {r['estimate']:+.4f}  p={_fmt_p(r['p_value'], r.get('b'))}  "
                  f"p_holm={_fmt_p(r['p_holm'], r.get('b'))}")
        if status["complete"]:
            print(f"Holm family: COMPLETE ({status['family_size']} tests)")
        else:
            print(f"Holm family: INCOMPLETE ({status['n_present']}/{status['family_size']}; m = {status['family_size']}, "
                  f"p_holm not final); missing {len(status['missing'])}; dev-check files: {status['dev_check_files']}")
        return 0 if status["complete"] else 3

    if args.command == "freeze":
        import json
        from datetime import datetime, timezone

        from s1bench.infrastructure.storage.results_store import freeze_differences, freeze_record

        if args.check:
            if not args.out.exists():
                print(f"no freeze file {args.out}")
                return 2
            diffs = freeze_differences(json.loads(args.out.read_text(encoding="utf-8")), Path("."))
            for d in diffs:
                print("  DIFF", d)
            print(f"{'OK: this folder matches' if not diffs else f'{len(diffs)} difference(s) from'} {args.out}")
            return 0 if not diffs else 1
        if args.out.exists():
            print(f"{args.out} already exists: a freeze is never overwritten (new protocol = new version file)")
            return 2
        rec = {"version": args.version, "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "note": "Protocol fixed before the final test-set runs (spec sec 16; the pre-freeze FPB test exposure "
                       "during pipeline checks is disclosed in docs/deviations.md, 2026-10-07). Final runs must match "
                       "config_snapshot exactly (verify-results --freeze).", **freeze_record(Path("."))}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rec, indent=2, sort_keys=False) + NL, encoding="utf-8")
        print(f"wrote {args.out}: {len(rec['config_snapshot'])} snapshot entries, {len(rec['files'])} files")
        return 0

    if args.command == "verify-results":
        import json

        from s1bench.application.verify_results import JEV_EXPECTED_VERSION, VerifyResults, check_jev_versions_across_runs
        from s1bench.infrastructure.secrets import load_env
        from s1bench.infrastructure.storage.results_store import JsonlCandidateReader, config_snapshot

        secrets = [v for k, v in load_env().items() if any(w in k.upper() for w in ("TOKEN", "KEY", "SECRET", "ACCOUNT"))]
        use_case = VerifyResults(
            hasher=hasher,
            read_examples=JsonlProcessedReader(args.data_dir / "processed").read_split,
            read_candidates=JsonlCandidateReader(args.data_dir / "candidates").read_split,
            local_snapshot=config_snapshot(Path("."), hasher),
            secret_values=secrets,
            frozen_snapshot=(json.loads(args.freeze.read_text(encoding="utf-8"))["config_snapshot"]
                             if args.freeze else None),
            examples_from=lambda root, d, s: JsonlProcessedReader(root).read_split(d, s),
            candidates_from=lambda root, d, s: JsonlCandidateReader(root).read_split(d, s),
        )
        reports = [use_case.execute(args.results_dir / run) for run in args.run]
        jev = check_jev_versions_across_runs(reports)
        if len(jev) > 1:
            for r in reports:
                r.errors.append(f"Jev versions differ across the verified runs {list(args.run)}: {sorted(jev)} "
                                "(spec sec 12: rerun all Jev in one window)")
        for run, r in zip(args.run, reports):
            (args.results_dir / run / "verify_report.json").write_text(
                json.dumps({"run_id": r.run_id, "ok": r.ok, "verified_together_with": list(args.run),
                            "errors": r.errors, "warnings": r.warnings, "files": r.files}, indent=2) + NL,
                encoding="utf-8")
            print(f"{run}: {'OK' if r.ok else 'FAILED'}  ({len(r.files)} prediction files, "
                  f"{len(r.errors)} errors, {len(r.warnings)} warnings)")
            for e in r.errors:
                print(f"  ERROR {e}")
            for w in r.warnings:
                print(f"  warn  {w}")
        if jev:
            print(f"Jev version(s) across runs: {sorted(jev)} (expected {JEV_EXPECTED_VERSION})")
        return 0 if all(r.ok for r in reports) else 1

    if args.command == "overlap":
        import json

        import yaml

        from s1bench.application.check_overlap import CheckTestOverlap, Reference
        from s1bench.infrastructure.overlap_refs import CharNgramSimilarity, iter_reference_texts

        ref_manifest = JsonManifestStore(args.reference_dir / "MANIFEST.json")
        problems = VerifyRawDatasets(ref_manifest, hasher, args.reference_dir / "raw").execute()
        if problems:
            print(f"{len(problems)} reference file(s) do not match data/reference/MANIFEST.json; run download first")
            return 1
        cfg = yaml.safe_load(args.references_config.read_text(encoding="utf-8"))
        listed = json.loads((args.reference_dir / "MANIFEST.json").read_text(encoding="utf-8"))
        listed_ids = set(listed.get("datasets", listed))
        missing = [item["id"] for item in cfg["datasets"] if item["id"] not in listed_ids]
        if missing:
            print(f"References not downloaded (not in data/reference/MANIFEST.json): {missing}")
            return 1
        refs = []
        for item in cfg["datasets"]:
            o = item["overlap"]
            root = args.reference_dir / "raw" / item["id"]
            refs.append(Reference(item["id"], o["exposure"], bool(o["near_dup"]),
                                  lambda root=root, o=o: iter_reference_texts(root, o["format"], o["glob"], o["fields"])))
        datasets = yaml.safe_load(args.processing_config.read_text(encoding="utf-8"))["datasets"]
        reader = JsonlProcessedReader(args.data_dir / "processed")
        use_case = CheckTestOverlap(lambda d: reader.read_split(d, "test"), refs,
                                    CharNgramSimilarity(args.threshold), progress=lambda m: print("  " + m, flush=True))
        report = use_case.execute(datasets)
        report["reference_manifest_sha256"] = hasher.sha256(args.reference_dir / "MANIFEST.json")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + NL, encoding="utf-8")
        for d in datasets:
            (args.out_dir / f"flagged_{d}.json").write_text(
                json.dumps(sorted(report["flagged"][d]), indent=1) + NL, encoding="utf-8")
        print(f"{'dataset':11s} {'reference':14s} {'n_test':>6s} {'exact':>6s} {'short':>6s} {'near':>5s}")
        for r in report["pairs"]:
            print(f"{r['dataset']:11s} {r['reference']:14s} {r['n_test']:6d} {r['n_exact']:6d} {r['n_exact_short']:6d} "
                  f"{r['n_near'] if r['near_dup_pass'] else '-':>5}")
        print("flagged per dataset (exact non-short + near):", report["n_flagged"])
        return 0

    if args.command == "exp1":
        import json

        from s1bench.application.exp1_metrics import exp1_row
        from s1bench.infrastructure.storage.results_store import read_predictions

        rows = []
        for run in args.run:
            for f in sorted((args.results_dir / run / "predictions").glob(f"*/*_{args.split}.jsonl")):
                dataset = f.stem.rsplit("_", 1)[0]
                preds = read_predictions(args.results_dir / run, f.parent.name, dataset, args.split)
                rows.append({"run": run, "model": f.parent.name, "dataset": dataset, "split": args.split,
                             **exp1_row(preds)})
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2) + NL, encoding="utf-8")
        fmt = lambda r, k: "-" if r.get(k) is None else f"{r[k]:.3f}"  # noqa: E731
        print(f"{'model':28s} {'dataset':11s} {'acc':>6s} {'mF1':>6s} {'MCC':>6s} {'ECE':>6s} {'AURC':>6s} status")
        for r in rows:
            print(f"{r['model']:28s} {r['dataset']:11s} {fmt(r, 'accuracy'):>6s} {fmt(r, 'macro_f1'):>6s} "
                  f"{fmt(r, 'mcc'):>6s} {fmt(r, 'ece'):>6s} {fmt(r, 'aurc'):>6s} {r['status']}")
        print(f"wrote {args.out}")
        return 0

    if args.command == "efficiency":
        import json

        import yaml

        from s1bench.application import efficiency_report

        cfg = yaml.safe_load(args.efficiency_config.read_text(encoding="utf-8"))
        efficiency_report.WARMUP = int(cfg["warmup_rows"])
        costs = {k: float(v) for k, v in (cfg.get("api_cost_per_1000") or {}).items() if v is not None}
        rows = [r for run in args.run for r in efficiency_report.efficiency_rows(args.results_dir / run, costs)]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2) + NL, encoding="utf-8")
        fmt = lambda v, f: "-" if v is None else format(v, f)  # noqa: E731
        print(f"{'model':24s} {'dataset':11s} {'split':5s} {'p50 ms':>8s} {'p95 ms':>8s} {'cache-on':>8s} {'tok':>6s} "
              f"{'VRAM MB':>8s} paper")
        for r in rows:
            print(f"{r['model']:24s} {r['dataset']:11s} {r['split']:5s} {fmt(r['p50_ms'], '.1f'):>8s} "
                  f"{fmt(r['p95_ms'], '.1f'):>8s} {fmt(r['prefix_cache_on_p50_ms'], '.1f'):>8s} "
                  f"{fmt(r['mean_input_tokens'], '.0f'):>6s} "
                  f"{fmt(r['peak_vram_mb'], '.0f'):>8s} {'yes' if r['paper_eligible'] else 'NO: ' + ', '.join(r['not_eligible_because'])}")
        return 0

    if args.command == "exp2":
        import json

        import numpy as np
        import yaml

        from s1bench.application.prevalence_shift import PrevalenceShift, to_matrices
        from s1bench.domain.predictions import Prediction

        cfg = yaml.safe_load(args.exp_config.read_text(encoding="utf-8"))
        labels = YamlLabelRepository(args.data_dir / "labels").load(args.dataset)
        run_dir = args.results_dir / args.run_id

        def load(split: str) -> list[Prediction]:
            path = run_dir / "predictions" / args.model / f"{args.dataset}_{split}.jsonl"
            with path.open(encoding="utf-8") as fh:
                return [Prediction(**{**json.loads(line), "candidates": tuple(json.loads(line)["candidates"]),
                                      "probabilities": tuple(json.loads(line)["probabilities"])}) for line in fh]

        report = json.loads((args.data_dir / "processed" / args.dataset / "report.json").read_text(encoding="utf-8"))
        counts = report["label_counts"]["train"]
        train_prior = np.array([counts.get(lid, 0) for lid in labels.ids()], dtype=float)
        train_prior /= train_prior.sum()
        kind = cfg["source_prior"].get(args.model, cfg["source_prior"]["default"])
        exp = PrevalenceShift(cfg["alphas"], args.draws or cfg["draws_per_alpha"], cfg["seed"], cfg["coverage"],
                              cfg["em"]["max_iter"], cfg["em"]["tol"])
        index = {lid: i for i, lid in enumerate(labels.ids())}
        if args.pseudo_split:
            dev_all = JsonlProcessedReader(args.data_dir / "processed").read_split(args.dataset, "dev")
            fit_ids, eval_ids = _pseudo_halves([e.id for e in dev_all])
            dev_preds = load("dev")
            dev_m = to_matrices([p for p in dev_preds if p.example_id in fit_ids], labels)
            test_m = to_matrices([p for p in dev_preds if p.example_id in eval_ids], labels)
            full = [e for e in dev_all if e.id in eval_ids]
        else:
            dev_m, test_m = to_matrices(load("dev"), labels), to_matrices(load("test"), labels)
            full = JsonlProcessedReader(args.data_dir / "processed").read_split(args.dataset, "test")
        full_test = ([e.id for e in full], np.array([index[e.label] for e in full]))
        if args.eps is not None:   # sensitivity of temperature/BCTS to the log floor (spec 10.2)
            from s1bench.domain import label_shift as _ls

            _ls.EPS = args.eps
        rows, meta = exp.run(dev_m, test_m, train_prior, kind, full_test=full_test)
        meta["pseudo_split"] = args.pseudo_split
        out = run_dir / "exp2" / args.model
        out.mkdir(parents=True, exist_ok=True)
        stem = args.dataset + ("" if args.eps is None else f"_eps{args.eps:g}") + ("_pseudo" if args.pseudo_split else "")
        with (out / f"{stem}.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        (out / f"{stem}_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        import collections
        agg = collections.defaultdict(list)
        for r in rows:
            if r["sampling"] in ("original", "with_replacement"):
                agg[(r["alpha"], r["arm"])].append(r)
        print(f"{args.model} {args.dataset}: T={meta['temperature']:.3f} source_prior={kind} n_test={meta['n_test']}")
        print("alpha  arm          macroF1   ECE     excess_risk  unique")
        for (alpha, arm), rs in agg.items():
            ex = [r["excess_risk"] for r in rs if r["excess_risk"] is not None]
            print(f"{alpha:>5}  {arm:11s}  {np.mean([r['macro_f1'] for r in rs]):.3f}   {np.mean([r['ece'] for r in rs]):.3f}"
                  f"   {np.mean(ex) if ex else float('nan'):+.3f}       {np.mean([r['unique_frac'] for r in rs]):.2f}")
        return 0

    if args.command == "router-candidates":
        import json

        import yaml

        from s1bench.application.router_candidates import ROUTER_DATASETS, build_router_candidates

        seed = int(yaml.safe_load(Path("configs/candidates.yaml").read_text(encoding="utf-8"))["candidate_seed"])
        reports = []
        for ds in args.dataset or list(ROUTER_DATASETS):
            r = build_router_candidates(ds, args.data_dir, args.out, seed)
            reports.append(r)
            print(f"{ds:9s} router C={r['router']['C']} test group accuracy {r['test_group_accuracy']:.4f} "
                  f"misrouted {r['n_misrouted']}/{r['n_test']} sha256 {r['sha256'][:12]}")
        man = args.out / "MANIFEST.json"
        old = json.loads(man.read_text(encoding="utf-8")) if man.exists() else {}
        old.update({r["dataset"]: r for r in reports})
        man.write_text(json.dumps(old, indent=1, sort_keys=True) + NL, encoding="utf-8", newline=NL)
        return 0

    if args.command == "contentfree-probes":
        import json

        import yaml

        from s1bench.application.content_free_probes import build_content_free_probes

        seed = int(yaml.safe_load(Path("configs/candidates.yaml").read_text(encoding="utf-8"))["candidate_seed"])
        reports = []
        for ds in args.dataset or ["clinc150", "hwu64", "fpb", "hatexplain", "trec6"]:
            r = build_content_free_probes(ds, args.data_dir, args.out, seed)
            reports.append(r)
            print(f"{ds:10s} orders {r['n_orders']} (rotation {r['n_rotation_orders']}, matched "
                  f"{r['n_matched_orders']} of {r['n_distinct_test_orders']} distinct test orders) "
                  f"probes {r['n_probes']}")
        man = args.out / "MANIFEST.json"
        old = json.loads(man.read_text(encoding="utf-8")) if man.exists() else {}
        old.update({r["dataset"]: r for r in reports})
        man.write_text(json.dumps(old, indent=1, sort_keys=True) + NL, encoding="utf-8", newline=NL)
        return 0

    if args.command == "candidates":
        import yaml

        from s1bench.infrastructure.models.embedder import SentenceTransformerEmbedder

        cfg = yaml.safe_load(args.candidates_config.read_text(encoding="utf-8"))
        datasets = args.dataset or cfg["datasets"]
        emb = cfg["embedding_model"]
        use_case = BuildCandidateSets(
            reader=JsonlProcessedReader(args.data_dir / "processed"),
            labels=YamlLabelRepository(args.data_dir / "labels"),
            store=JsonlCandidateStore(args.data_dir / "candidates", hasher),
            embedder_factory=lambda: SentenceTransformerEmbedder(emb["id"], emb["revision"]),
            clock=UtcClock(),
            seed=int(cfg["candidate_seed"]),
            max_candidates=int(cfg["max_candidates"]),
            target_group_size=int(cfg["target_group_size"]), group_restarts=int(cfg["group_restarts"]),
            splits=cfg["splits"],
        )
        for r in use_case.execute(datasets):
            test = r["splits"]["test"]
            worst = max(v["max_abs_deviation_from_uniform"] for v in test["by_set_size"].values())
            extra = (f", groups {r['group_sizes']}, nearest neighbour in same group {r['nearest_neighbour_in_same_group_share']}"
                     if "group_sizes" in r else "")
            print(f"{r['dataset']:11s} {r['mode']:18s} K={r['k']}  test n={test['n']:5d}  "
                  f"gold-position max abs deviation {worst}{extra}")
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
