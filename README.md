# Typed Decision Models under Prevalence Shift and Label Budgets

Code, frozen protocol and per-example outputs for the paper

> **Typed Decision Models under Prevalence Shift and Label Budgets: Decision Training, Calibration, and Implicit Priors**
> Anonymous authors (under review)

We compare three typed decision models (Jev, Tev1-4B, Nimble-9B), their base models (Qwen3.5-4B, Qwen3.5-9B) and two
trained classifiers (DeBERTa-v3-large, TF-IDF + logistic regression) on identical candidate-set decisions from five
public datasets (CLINC150, HWU64, Financial PhraseBank, HateXplain, TREC-6), under simulated prevalence (label) shift.
The confirmatory test family (35 tests) and the analysis protocol were frozen before the final confirmatory runs and
analyses; earlier test-split pipeline checks, including exploratory FPB shift analyses, are disclosed in Appendix A of
the paper. All runs were verified against SHA-256 freeze files.

Version 2 of the paper adds a label-budget study: DeBERTa-v3-large and TF-IDF + LR trained on 100 to 5,000 labels
(5 seeds), and Qwen3.5-4B fine-tuned with QLoRA on 100, 500 and 2,500 labels (3 seeds), each compared with Jev. Its
protocol was frozen separately (`docs/FREEZE_v1.2.json`) before any of its test predictions; it adds estimates, not
tests.

The revision of version 2 adds five exploratory analyses under a further freeze (`docs/FREEZE_v1.3.json`, frozen
after all earlier results were known and before any of its test predictions): label budgets with model selection
inside the budget, routing to the intent group by a learned router, content-free calibration, two generic prompt
formats on FPB, and the Nimble-9B pair in bf16. None replaces a frozen result.

## How to reproduce: three levels

| Level | What you reproduce | What you need | Time |
|---|---|---|---|
| 1 | Integrity of every released file, and every table and figure of the paper from the released outputs (`python scripts/paper/reproduce_paper.py --unzip`, `s1bench verify-results`) | Any computer with Python 3.11, the downloaded datasets (see Data) | about 1.5 h on a CPU |
| 2 | Every test and analysis from the released predictions: the 35 confirmatory tests, the shift experiments, the budget curves and all exploratory analyses (`REPRODUCE.md` sections 6-9, 11-13) | CPU | hours |
| 3 | The predictions and trainings themselves (`REPRODUCE.md` sections 2-4, 10, 13) | One 12 GB GPU; about 40 GB of model weights from Hugging Face; a Cloudflare Workers AI account for Jev | about 2.5 weeks of GPU time; about 26,000 Jev calls |

Levels 1 and 2 reproduce the paper's numbers exactly. Level 3 does not, in two places: Jev is a hosted model whose
version cannot be pinned and whose outputs are not deterministic, and GPU training is not guaranteed to be
bit-identical across hardware and library versions. The local decoder models score identically on repeated runs
(`final_determinism`).

For level 3, set `HF_HOME` to a folder with enough space for the weights (Qwen3.5-4B, Tev1-4B and DeBERTa-v3-large are
downloaded once; Qwen3.5-9B is shared by Qwen3.5-9B and Nimble-9B, whose adapter is small); the pinned revisions are in
`configs/models.yaml`. Access to the Tev1-4B weights depends on their licence, which was still being finalised when
the paper was written. The QLoRA adapters are not released and must be retrained.

## Repository layout

```
src/s1bench/        evaluation code (domain -> application -> infrastructure, cli)
configs/            datasets, models, candidate sets, experiments (frozen)
data/labels/        frozen label names and descriptions (one file per dataset)
data/candidates/    frozen candidate sets: example id, candidate order, gold position (no text)
data/overlap/       test-set ids that overlap with vendor-documented corpora
data/*MANIFEST.json SHA-256 hashes of the raw downloads and processed splits
docs/               benchmark_spec_v1.md (protocol), deviations.md (deviation log), FREEZE_v1.*.json (freeze hashes)
results/*.zip       final per-example outputs and analyses (see below)
scripts/paper/      scripts that build every table and figure of the paper from results/
paper/              LaTeX source of the paper
tests/              unit and integration tests
```

The protocol document `docs/benchmark_spec_v1.md` is partly written in Roman Urdu. Its
hash is part of the freeze file, so it is released unchanged.

## Setup

Python 3.11 (exactly; not 3.12 or later). Python 3.12 changed the built-in `sum()` of floats to compensated
summation. Jev's probabilities lie on a 0.01 grid, so the last-bit change after renormalisation breaks ties in
confidence and moves Jev's AURC and accuracy at 80% coverage in the sixth decimal. The paper's tables and figures are
unchanged, but `scripts/paper/fig_calibration.py` stops on its exact check against Exp 1, and the released analysis
files are reproduced byte for byte only on 3.11.

```
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"        # Linux/macOS: .venv/bin/python
.venv/Scripts/python -m pip install -r requirements-run.txt     # model inference and training
.venv/Scripts/python -m pip install -r requirements-paper.txt   # tables and figures only
```

Check that the code is the frozen code:

```
s1bench freeze --check --out docs/FREEZE_v1.3.json
```

The protocol was frozen on 2026-10-08 (`docs/FREEZE_v1.1.json`) and amended once on the same day, before any test of
the confirmatory family and any shift analysis on the final runs (`docs/FREEZE_v1.1.1.json`; the timing is given in
Appendix A of the paper). For the first public release, internal workflow notes in the protocol document were
reworded; `docs/FREEZE_v1.1.1-release.json` is identical to `FREEZE_v1.1.1.json` except the hash of
`docs/benchmark_spec_v1.md`, and all Stage 1 runs verify against the original freeze files
(`s1bench verify-results`). The label-budget study was frozen on 2026-10-21 (`docs/FREEZE_v1.2.json`), after the
reworded protocol document, before any of its test predictions. The exploratory revision analyses were frozen on
2026-11-10 (`docs/FREEZE_v1.3.json`) before any of their test predictions. The released code matches
`FREEZE_v1.3.json`; every run verifies against the freeze file under which it was produced (`s1bench verify-results
--freeze`). See `docs/deviations.md`.

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
| `final_budget.zip`, `final_budget_predictions_<dataset>_<split>.zip` | label budgets: DeBERTa-v3-large and TF-IDF + LR at 100-5,000 labels, 5 seeds (training summaries, environment records, verification report; predictions split by dataset and split to keep each archive below 100 MB) |
| `final_qlora.zip` | Qwen3.5-4B fine-tuned with QLoRA at 100, 500 and 2,500 labels, 3 seeds: predictions, training summaries (with the sha256 of every adapter; the adapters themselves are not released) |
| `final_analysis_debseeds.zip` | shift experiment repeated for all five DeBERTa-v3-large seeds (exploratory seed sensitivity) |
| `final_budget_analysis.zip` | budget curves: differences to Jev (and of QLoRA to Qwen3.5-4B and Tev1-4B) per budget with bootstrap intervals, crossover budgets |
| `final_realistic.zip`, `final_realistic_predictions_<dataset>_<split>.zip` | v1.3: label budgets with model selection inside the budget, DeBERTa-v3-large and TF-IDF + LR, 5 seeds (predictions split by dataset and split) |
| `final_qlora_realistic.zip` | v1.3: QLoRA with model selection inside the budget, 3 seeds (adapters not released) |
| `final_realistic_analysis.zip` | v1.3: budget curves of the two runs above |
| `final_router.zip` | v1.3: predictions on the test examples that the learned router sends to a group without the gold label; DeBERTa and TF-IDF probabilities over all classes |
| `final_contentfree.zip` | v1.3: predictions on the content-free probes (all zero-shot systems) |
| `final_promptsens.zip`, `final_bf16.zip` | v1.3: the four local models on FPB in two generic prompt formats; the Nimble-9B pair in bf16 |
| `final_analysis_calibrob.zip` | outputs of the exploratory calibration analyses (content-free, routing, prompt format and precision, EM with the true prior, PA seed bootstrap) |

Every prediction row holds the example id, the candidate order, the probability vector over the candidates and the
predicted label. No dataset text is included.

## Reproducing the paper

**Every command** behind every number in the paper, in the order in which it was run, is in
[`REPRODUCE.md`](REPRODUCE.md). The short version:

**From the released outputs (no GPU, no API key).** One command rebuilds every table and figure that the paper
includes and compares them with `paper/` (needs `data/processed/`, see Data above):

```
python scripts/paper/reproduce_paper.py --unzip
```

The individual scripts, after unzipping `results/`:

```
python scripts/paper/make_tables.py            # paper/tables/*.tex
python scripts/paper/make_figures.py           # paper/figures/*.pdf
python scripts/paper/make_pipeline_figure.py   # paper/figures/fig_pipeline.pdf
python scripts/paper/exp2_descriptives.py      # descriptive shift tables
python scripts/paper/budget_tables.py          # label-budget tables
python scripts/paper/fig_budget.py             # label-budget figures
python scripts/paper/fig_calibration.py        # reliability, risk-coverage and implicit-prior figures
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
s1bench train --model deberta-v3-large --dataset fpb --budget 100 --seed 1001 --device cuda --run-id my_budget_run
s1bench train-qlora --dataset fpb --budget 100 --seed 1001 --device cuda --run-id my_qlora_run
s1bench verify-results --run my_budget_run --run my_qlora_run --freeze docs/FREEZE_v1.2.json
```

`train-qlora` writes its adapters to `results/<run>/adapters/`; move them out of the results folder before
`verify-results`, which refuses model weights inside a results folder.

Jev is a hosted model whose version cannot be pinned and whose outputs are not deterministic, so its numbers may
differ slightly from ours.

## Citation

Citation details will be added after the review period.

## Licence

Code: MIT (see `LICENSE`). The datasets keep their own licences; the model weights and the Jev API keep their
providers' terms.
