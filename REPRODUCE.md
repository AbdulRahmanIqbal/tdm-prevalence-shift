# Reproducing every run and analysis

These are the commands behind every number in the paper, in the order in which they were run. Run them from the
repository root with the environment from `README.md` active. The loops use bash syntax (on Windows, Git Bash works).

```bash
DATASETS="clinc150 hwu64 fpb hatexplain trec6"
```

You can skip steps 2-5 and start at step 6 with the released outputs: unzip `results/*.zip` into `results/`.

**Fastest check of the paper (CPU only).** After step 1, one command unpacks the released outputs, rebuilds every
table and figure that the paper includes, and compares them with the files in `paper/` (tables byte for byte, figures
ignoring the PDF creation date); it exits with an error if anything differs:

```bash
python scripts/paper/reproduce_paper.py --unzip              # about 1.5 h; --skip-slow leaves out the 1 h EM simulation
s1bench verify-results --run final_contentfree --run final_router --run final_promptsens --run final_bf16 \
  --run final_realistic --run final_qlora_realistic --freeze docs/FREEZE_v1.3.json
```

Use Python 3.11 (see Setup in `README.md`): with Python 3.12 or later, Jev's AURC and accuracy at 80% coverage differ
in the sixth decimal and `reproduce_paper.py` stops.

## 1. Code and data

```bash
s1bench freeze --check --out docs/FREEZE_v1.3.json           # the code is the frozen code (current version)
s1bench download                                       # pinned raw datasets -> data/raw/
s1bench verify                                         # SHA-256 against data/MANIFEST.json
s1bench prepare                                        # data/processed/
s1bench verify-frozen                                  # processed splits, label texts, candidate sets
python -m pytest -q tests                              # 143 tests
s1bench overlap                                        # test-set overlap with vendor-documented corpora
```

## 2. Local language models (one 12 GB GPU, about 8 h; decoder runs used freeze file v1.1)

```bash
# determinism check: 50 examples scored twice
for m in qwen3.5-4b tev1-4b qwen3.5-9b nimble-9b; do
  s1bench predict --model $m --dataset clinc150 --split dev --limit 50 --device cuda --repeat-check --run-id final_determinism
done

# predictions (nimble-9b-main is the exploratory later Nimble release)
for m in qwen3.5-4b tev1-4b qwen3.5-9b nimble-9b nimble-9b-main; do
  for d in $DATASETS; do for s in dev test; do
    s1bench predict --model $m --dataset $d --split $s --device cuda --run-id final_local
  done; done
done

# latency with and without prefix caching
for m in qwen3.5-4b tev1-4b qwen3.5-9b nimble-9b; do for d in $DATASETS; do
  s1bench predict --model $m --dataset $d --split dev --limit 210 --device cuda --time-prefix-cache --run-id final_local
done; done
```

## 3. Trained classifiers (freeze file v1.1.1; DeBERTa-v3-large about 1.5 days on the GPU, TF-IDF minutes)

```bash
for d in fpb trec6 hwu64 clinc150 hatexplain; do
  s1bench train --model deberta-v3-large --dataset $d --seed 1001 --seed 1002 --seed 1003 --seed 1004 --seed 1005 \
    --device cuda --run-id final_train
done
s1bench train --model tfidf-lr --dataset clinc150 --dataset hwu64 --dataset fpb --dataset hatexplain --dataset trec6 \
  --run-id final_train
```

## 4. Jev (hosted API; needs `.env`, see `.env.example`; all calls within 72 hours; about 23,700 calls, 2-3 h)

```bash
for d in $DATASETS; do for s in dev test; do
  s1bench predict --model jev --dataset $d --split $s --workers 3 --run-id final_jev
done; done

# three repeats on 500 test examples per dataset
for r in 1 2 3; do for d in $DATASETS; do
  s1bench predict --model jev --dataset $d --split test --limit 500 --workers 3 --run-id final_jev_rep$r
done; done
```

## 5. Verification and efficiency

```bash
s1bench verify-results --run final_determinism --run final_local --freeze docs/FREEZE_v1.1.json
s1bench verify-results --run final_train --freeze docs/FREEZE_v1.1.1.json
s1bench verify-results --run final_jev --run final_jev_rep1 --run final_jev_rep2 --run final_jev_rep3 \
  --freeze docs/FREEZE_v1.1.1.json
s1bench efficiency --run final_local --run final_train --out results/final_local/efficiency.json
```

## 6. No-shift metrics

```bash
s1bench exp1 --run final_local --run final_train --split test --out results/final_analysis/exp1_test.json
s1bench exp1 --run final_jev --split test --out results/final_analysis/exp1_test_jev.json
```

## 7. The 35 confirmatory tests

```bash
for d in $DATASETS; do
  s1bench primary-pair --test P1 --dataset $d --model-run final_local --base-run final_local --out-run final_analysis
  s1bench primary-pair --test P2 --dataset $d --model-run final_local --base-run final_local --out-run final_analysis
  s1bench primary-pa --dataset $d --model jev --baseline deberta-v3-large \
    --model-run final_jev --baseline-run final_train --out-run final_analysis
done
s1bench primary-holm --run final_analysis    # Holm over all 35 tests -> results/final_analysis/primary/holm_family.json
```

## 8. Descriptive shift experiment

```bash
for d in $DATASETS; do
  for m in qwen3.5-4b tev1-4b qwen3.5-9b nimble-9b nimble-9b-main; do s1bench exp2 --run-id final_local --model $m --dataset $d; done
  for m in deberta-v3-large tfidf-lr; do s1bench exp2 --run-id final_train --model $m --dataset $d; done
  s1bench exp2 --run-id final_jev --model jev --dataset $d
  s1bench exp2 --run-id final_jev --model jev --dataset $d --eps 0.001    # log-floor sensitivity
  s1bench exp2 --run-id final_jev --model jev --dataset $d --eps 1e-12
done
```

## 9. Exploratory analyses added after the results (Appendix A of the paper)

```bash
# PA1-PA3 with five other prior sets (B = 1,000)
for k in 1 2 3 4 5; do for d in $DATASETS; do
  s1bench primary-pa --dataset $d --model-run final_jev --baseline-run final_train \
    --out-run final_analysis_priorsens_s$k --exp-config configs/experiments/sensitivity/exp2_priorseed_$k.yaml
done; done

python scripts/paper/priorsens_table.py          # prior-set sensitivity table
python scripts/paper/pa_population.py            # PA1-PA3 averaged over 2,000 priors
python scripts/paper/pa_procedure_check.py       # PA estimates without the bootstrap base
python scripts/paper/hwu64_overlap_sensitivity.py  # HWU64 without the MASSIVE overlap
python scripts/paper/implicit_prior_table.py     # distance of the implicit priors from the true class distribution
python scripts/paper/review3_checks.py           # EM without shift; FPB confusion matrices
python scripts/paper/rank_correlations.py        # rank correlations between metrics
python scripts/paper/tfidf_grid_check.py         # TF-IDF grid extended to 1e5 and 1e6 (needs data/processed)
python scripts/paper/em_candidate_simulation.py --miscalibration  # EM simulations: candidate sets, distorted posteriors (~1 h, CPU)
python scripts/paper/deberta_seed_shift.py       # shift experiment for all five DeBERTa seeds (needs data/processed)
```

## 10. Label budgets (paper version 2; freeze file v1.2, one 12 GB GPU; DeBERTa about 1 day, QLoRA about 1 week)

Every (dataset, budget) is one call with all seeds, because the reserve-seed queue is shared within the call. The
full-data points are the Stage 1 runs in `final_train`.

```bash
s1bench freeze --check --out docs/FREEZE_v1.2.json

# DeBERTa-v3-large, 5 seeds
for d in fpb trec6; do for b in 100 250 500 1000 2500; do
  s1bench train --model deberta-v3-large --dataset $d --budget $b \
    --seed 1001 --seed 1002 --seed 1003 --seed 1004 --seed 1005 --device cuda --run-id final_budget
done; done
for d in hwu64 clinc150 hatexplain; do for b in 100 250 500 1000 2500 5000; do
  s1bench train --model deberta-v3-large --dataset $d --budget $b \
    --seed 1001 --seed 1002 --seed 1003 --seed 1004 --seed 1005 --device cuda --run-id final_budget
done; done

# TF-IDF + LR, 5 seeds (CPU)
s1bench train --model tfidf-lr --dataset fpb --dataset trec6 --budget 100 --budget 250 --budget 500 --budget 1000 \
  --budget 2500 --seed 1001 --seed 1002 --seed 1003 --seed 1004 --seed 1005 --run-id final_budget
s1bench train --model tfidf-lr --dataset hwu64 --dataset clinc150 --dataset hatexplain --budget 100 --budget 250 \
  --budget 500 --budget 1000 --budget 2500 --budget 5000 --seed 1001 --seed 1002 --seed 1003 --seed 1004 --seed 1005 \
  --run-id final_budget

# Qwen3.5-4B with QLoRA: budgets 100, 500, 2500 and seeds 1001-1003 from configs/experiments/qlora.yaml
for d in fpb trec6 hatexplain hwu64 clinc150; do
  s1bench train-qlora --dataset $d --device cuda --run-id final_qlora
  # move results/final_qlora/adapters/$d/ out of results/ (their sha256 are in train_summary.json)
done

s1bench verify-results --run final_budget --run final_qlora --freeze docs/FREEZE_v1.2.json
```

Budget curves (differences per budget with bootstrap intervals, crossover budget), one call per dataset and system:

```bash
for d in $DATASETS; do
  case $d in fpb|trec6) B="--expect-budget 100 --expect-budget 250 --expect-budget 500 --expect-budget 1000 --expect-budget 2500";;
    *) B="--expect-budget 100 --expect-budget 250 --expect-budget 500 --expect-budget 1000 --expect-budget 2500 --expect-budget 5000";; esac
  s1bench budget-curve --dataset $d --system deberta-v3-large --system-run final_budget --system-run final_train \
    --ref jev --ref-run final_jev $B --expect-budget full --expect-seeds 5 --out-run final_budget_analysis
  s1bench budget-curve --dataset $d --system tfidf-lr --system-run final_budget --system-run final_train \
    --ref jev --ref-run final_jev $B --expect-budget full --expect-seeds 5 --expect-full-seeds 1 --out-run final_budget_analysis
  s1bench budget-curve --dataset $d --system qwen3.5-4b-qlora --system-run final_qlora --ref jev --ref-run final_jev \
    --expect-budget 100 --expect-budget 500 --expect-budget 2500 --expect-seeds 3 --out-run final_budget_analysis
  for r in qwen3.5-4b tev1-4b; do
    s1bench budget-curve --dataset $d --system qwen3.5-4b-qlora --system-run final_qlora --ref $r --ref-run final_local \
      --expect-budget 100 --expect-budget 500 --expect-budget 2500 --expect-seeds 3 --out-run final_budget_analysis
  done
done
```

## 11. Paper tables and figures

```bash
python scripts/paper/make_tables.py
python scripts/paper/exp2_descriptives.py
python scripts/paper/make_figures.py
python scripts/paper/make_pipeline_figure.py
python scripts/paper/budget_tables.py
python scripts/paper/fig_budget.py
python scripts/paper/fig_calibration.py
```

All scripts read only `results/final_*` and `data/` and write to `results/final_analysis/` and `paper/`.

## 12. Further exploratory analyses on the stored predictions (CPU)

```bash
python scripts/paper/calibration_robustness.py   # calibration metrics used by the content-free analysis
python scripts/paper/pa_seed_bootstrap.py        # PA1/PA2 with the DeBERTa seeds resampled
python scripts/paper/router_end_to_end.py        # router accuracy (group choice only)
python scripts/paper/em_true_prior.py            # EM started from the true class prior instead of the implicit prior
```

## 13. Revision protocol v1.3 (exploratory; freeze file v1.3, one 12 GB GPU, Jev via the API; about 6 days of GPU time, mostly the QLoRA rerun, and 2,106 Jev calls)

Five exploratory analyses added after the confirmatory results; none replaces a frozen result (see
`docs/deviations.md` and Appendix A of the paper). The router candidate sets and the content-free probes are released
in `data/candidates_router/` and `data/contentfree/`; they can be rebuilt with:

```bash
s1bench freeze --check --out docs/FREEZE_v1.3.json
s1bench router-candidates                             # data/candidates_router/ (CLINC150, HWU64)
s1bench contentfree-probes                            # data/contentfree/ (all five datasets)
```

Content-free probes and routing (local models; DeBERTa and TF-IDF retrained to store probabilities over all classes):

```bash
CF="--split contentfree --processed-dir data/contentfree/processed --candidates-dir data/contentfree/candidates"
for m in qwen3.5-4b tev1-4b qwen3.5-9b nimble-9b; do for d in $DATASETS; do
  s1bench predict --model $m --dataset $d $CF --device cuda --run-id final_contentfree
done; done
for m in qwen3.5-4b tev1-4b qwen3.5-9b nimble-9b; do for d in clinc150 hwu64; do
  s1bench predict --model $m --dataset $d --split test --candidates-dir data/candidates_router --device cuda \
    --run-id final_router
done; done
s1bench train --model deberta-v3-large --dataset clinc150 --dataset hwu64 --save-full-probs --device cuda --run-id final_router
s1bench train --model tfidf-lr --dataset clinc150 --dataset hwu64 --save-full-probs --run-id final_router
```

Jev on the same probes and misrouted examples (needs `.env`):

```bash
for d in $DATASETS; do s1bench predict --model jev --dataset $d $CF --workers 3 --run-id final_contentfree; done
for d in clinc150 hwu64; do
  s1bench predict --model jev --dataset $d --split test --candidates-dir data/candidates_router --workers 3 \
    --run-id final_router
done
```

Generic prompt formats and bf16 (FPB test only):

```bash
for m in qwen3.5-4b tev1-4b qwen3.5-9b nimble-9b; do for v in plain plain_names; do
  s1bench predict --model $m --dataset fpb --split test --device cuda --run-id final_promptsens \
    --prompt-config configs/experiments/prompt_sensitivity.yaml --prompt-variant $v
done; done
for m in qwen3.5-9b nimble-9b; do
  s1bench predict --model $m --dataset fpb --split test --device cuda --bf16-offload-gib 9 --run-id final_bf16
done
```

Label budgets with model selection inside the budget (20% of each budget for selection and early stopping):

```bash
S5="--seed 1001 --seed 1002 --seed 1003 --seed 1004 --seed 1005"
R="--realistic-config configs/experiments/realistic_budget.yaml --run-id final_realistic"
for d in fpb trec6; do for b in 100 250 500 1000 2500; do
  s1bench train --model deberta-v3-large --dataset $d --budget $b $S5 --device cuda $R
done; done
for d in hwu64 clinc150 hatexplain; do for b in 100 250 500 1000 2500 5000; do
  s1bench train --model deberta-v3-large --dataset $d --budget $b $S5 --device cuda $R
done; done
s1bench train --model tfidf-lr --dataset fpb --dataset trec6 --budget 100 --budget 250 --budget 500 --budget 1000 \
  --budget 2500 $S5 $R
s1bench train --model tfidf-lr --dataset hwu64 --dataset clinc150 --dataset hatexplain --budget 100 --budget 250 \
  --budget 500 --budget 1000 --budget 2500 --budget 5000 $S5 $R
for d in fpb trec6 hwu64 clinc150 hatexplain; do for b in 100 500 2500; do
  s1bench train-qlora --qlora-config configs/experiments/qlora_realistic.yaml --run-id final_qlora_realistic \
    --dataset $d --budget $b --seed 1001 --seed 1002 --seed 1003
done
  # move results/final_qlora_realistic/adapters/$d/ out of results/ (their sha256 are in train_summary.json)
done

s1bench verify-results --run final_contentfree --run final_router --run final_promptsens --run final_bf16 \
  --run final_realistic --run final_qlora_realistic --freeze docs/FREEZE_v1.3.json
s1bench verify-results --run final_contentfree --run final_router --run final_jev   # Jev version across runs
```

Budget curves under in-budget selection (no full-data budget), then the v1.3 analyses, tables and figures:

```bash
for d in $DATASETS; do
  case $d in fpb|trec6) B="--expect-budget 100 --expect-budget 250 --expect-budget 500 --expect-budget 1000 --expect-budget 2500";;
    *) B="--expect-budget 100 --expect-budget 250 --expect-budget 500 --expect-budget 1000 --expect-budget 2500 --expect-budget 5000";; esac
  for s in deberta-v3-large tfidf-lr; do
    s1bench budget-curve --dataset $d --system $s --system-run final_realistic --ref jev --ref-run final_jev $B \
      --expect-seeds 5 --out-run final_realistic_analysis
  done
  Q="--expect-budget 100 --expect-budget 500 --expect-budget 2500 --expect-seeds 3 --out-run final_realistic_analysis"
  s1bench budget-curve --dataset $d --system qwen3.5-4b-qlora --system-run final_qlora_realistic --ref jev --ref-run final_jev $Q
  for r in qwen3.5-4b tev1-4b; do
    s1bench budget-curve --dataset $d --system qwen3.5-4b-qlora --system-run final_qlora_realistic --ref $r \
      --ref-run final_local $Q
  done
done

python scripts/paper/content_free_calibration.py     # content-free calibration vs EM
python scripts/paper/router_calibration.py           # accuracy and calibration with the router's group
python scripts/paper/prompt_precision_checks.py      # P1/P2 in generic formats and in bf16
python scripts/paper/fig_revision.py                 # figures of the v1.3 analyses
```

The QLoRA adapters are not released (one per dataset, budget and seed); each is identified by the SHA-256 in
`train_summary.json`, and they can be retrained with the commands above.
