# Typed Decision Models under Prevalence Shift

Code, frozen protocol and per-example outputs for the paper

> **Typed Decision Models under Prevalence Shift: Decision Training, Calibration, and Implicit Priors**
> Abdul Rahman Iqbal, Muhammad Fawad Nasim, Amir Saleem
> arXiv preprint, 2026 (link will be added after publication)

We compare three typed decision models (Jev, Tev1-4B, Nimble-9B), their base models (Qwen3.5-4B, Qwen3.5-9B) and two
trained classifiers (DeBERTa-v3-large, TF-IDF + logistic regression) on identical candidate-set decisions from five
public datasets (CLINC150, HWU64, Financial PhraseBank, HateXplain, TREC-6), under simulated prevalence (label) shift.
The 35 confirmatory tests were fixed before any test was computed; all runs were verified against SHA-256 freeze files.

## Repository layout

```
src/s1bench/        evaluation code (domain -> application -> infrastructure, cli)
configs/            datasets, models, candidate sets, experiments (frozen)
data/labels/        frozen label names and descriptions (one file per dataset)
data/candidates/    frozen candidate sets: example id, candidate order, gold position (no text)
data/overlap/       test-set ids that overlap with vendor-documented corpora
data/*MANIFEST.json SHA-256 hashes of the raw downloads and processed splits
docs/               benchmark_spec_v1.md (protocol), deviations.md (deviation log), FREEZE_v1.1*.json (freeze hashes)
results/*.zip       final per-example outputs and analyses (see below)
scripts/paper/      scripts that build every table and figure of the paper from results/
paper/              LaTeX source of the paper
tests/              unit and integration tests
```

The protocol document `docs/benchmark_spec_v1.md` is partly written in Roman Urdu, the authors' working language. Its
hash is part of the freeze file, so it is released unchanged.

## Setup

Python 3.11.

```
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"        # Linux/macOS: .venv/bin/python
.venv/Scripts/python -m pip install -r requirements-run.txt     # model inference and training
.venv/Scripts/python -m pip install -r requirements-paper.txt   # tables and figures only
```

Check that the code is the frozen code:

```
s1bench freeze --check --out docs/FREEZE_v1.1.1-release.json
```

The protocol was frozen on 2026-10-08 (`docs/FREEZE_v1.1.json`) and amended once before any test was computed
(`docs/FREEZE_v1.1.1.json`). For this public release, internal workflow notes in the protocol document were reworded;
`docs/FREEZE_v1.1.1-release.json` is identical to `FREEZE_v1.1.1.json` except the hash of
`docs/benchmark_spec_v1.md`. Against the original `FREEZE_v1.1.1.json`, `freeze --check` therefore reports exactly
one difference, this file; all code, configurations, labels and candidate sets match the original freeze, and every
released run verifies against it (`s1bench verify-results`). See the last row of `docs/deviations.md`.

## Data

The dataset texts are not redistributed here, because some licences do not allow it. Download them from their pinned
public sources and check them against the recorded hashes:

```
s1bench download        # fetch the pinned sources into data/raw/
s1bench verify          # SHA-256 check against data/MANIFEST.json
s1bench prepare         # build data/processed/ (cleaning, splits)
s1bench verify-frozen   # processed splits, label texts and candidate sets against their manifests
```

All five datasets are public. Please follow each dataset's licence: CLINC150, HWU64, Financial PhraseBank (50%
agreement subset, non-commercial), HateXplain and TREC-6. `pytest` passes completely only after `s1bench prepare`; one
test checks the candidate files against the processed texts.

## Released outputs

Unzip the files in `results/` into `results/` to obtain the run folders.

| Archive | Contents |
|---|---|
| `final_local.zip` | Qwen3.5-4B, Tev1-4B, Qwen3.5-9B, Nimble-9B (and the exploratory Nimble-9B `main` release): dev and test predictions, latency, environment snapshots, shift-experiment rows |
| `final_train.zip` | DeBERTa-v3-large (5 seeds) and TF-IDF + LR: predictions and training summaries |
| `final_jev.zip`, `final_jev_rep1-3.zip` | Jev: full run and three repeats on 500 test examples per dataset |
| `final_determinism.zip` | repeat-scoring check of the four local models |
| `final_analysis.zip` | no-shift metrics, the 35 confirmatory tests with Holm correction, and all descriptive/exploratory analyses |
| `final_analysis_priorsens_s1-5.zip` | prior-set sensitivity of PA1-PA3 (exploratory) |

Every prediction row holds the example id, the candidate order, the probability vector over the candidates and the
predicted label. No dataset text is included.

## Reproducing the paper

**Every command** behind every number in the paper, in the order in which it was run, is in
[`REPRODUCE.md`](REPRODUCE.md). The short version:

**From the released outputs (no GPU, no API key).** After unzipping `results/`:

```
python scripts/paper/make_tables.py            # paper/tables/*.tex
python scripts/paper/make_figures.py           # paper/figures/*.pdf
python scripts/paper/make_pipeline_figure.py   # paper/figures/fig_pipeline.pdf
python scripts/paper/exp2_descriptives.py      # descriptive shift tables
```

The other scripts in `scripts/paper/` produce the exploratory analyses listed in Appendix A of the paper.

**The analysis from the predictions:**

```
s1bench exp1 --run final_local --split test --out results/final_analysis/exp1_test.json
s1bench exp2 --run-id final_jev --model jev --dataset fpb            # one model x dataset
s1bench primary-pair --test P2 --dataset fpb --model-run final_local --base-run final_local --out-run final_analysis
s1bench primary-pa --dataset fpb --model-run final_jev --baseline-run final_train --out-run final_analysis
s1bench primary-holm --run final_analysis
```

**The predictions themselves.** The local models need one GPU with 12 GB (the 9B models run in 8-bit); Jev needs a
Cloudflare Workers AI account (copy `.env.example` to `.env`). For example:

```
s1bench predict --model nimble-9b --dataset fpb --split test --device cuda --run-id my_run
s1bench train --model deberta-v3-large --dataset fpb --seed 1001 --device cuda --run-id my_run
s1bench predict --model jev --dataset fpb --split test --workers 3 --run-id my_run
s1bench verify-results --run my_run --freeze docs/FREEZE_v1.1.1.json
```

Jev is a hosted model whose version cannot be pinned and whose outputs are not deterministic, so its numbers may
differ slightly from ours.

## Citation

```
@misc{iqbal2026typed,
  author = {Abdul Rahman Iqbal and Muhammad Fawad Nasim and Amir Saleem},
  title  = {Typed Decision Models under Prevalence Shift: Decision Training, Calibration, and Implicit Priors},
  year   = {2026},
  note   = {arXiv preprint}
}
```

## Licence

Code: MIT (see `LICENSE`). The datasets keep their own licences; the model weights and the Jev API keep their
providers' terms.
