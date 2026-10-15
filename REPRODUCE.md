# Reproducing every run and analysis

These are the commands behind every number in the paper, in the order in which they were run. Run them from the
repository root with the environment from `README.md` active. The loops use bash syntax (on Windows, Git Bash works).

```bash
DATASETS="clinc150 hwu64 fpb hatexplain trec6"
```

You can skip steps 2-5 and start at step 6 with the released outputs: unzip `results/*.zip` into `results/`.

## 1. Code and data

```bash
s1bench freeze --check --out docs/FREEZE_v1.1.1-release.json   # the code is the frozen code
s1bench download                                       # pinned raw datasets -> data/raw/
s1bench verify                                         # SHA-256 against data/MANIFEST.json
s1bench prepare                                        # data/processed/
s1bench verify-frozen                                  # processed splits, label texts, candidate sets
python -m pytest -q tests                              # 92 tests
s1bench overlap                                        # test-set overlap with vendor-documented corpora
```

## 2. Local language models (one 12 GB GPU; decoder runs used freeze file v1.1)

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

## 3. Trained classifiers (freeze file v1.1.1)

```bash
for d in fpb trec6 hwu64 clinc150 hatexplain; do
  s1bench train --model deberta-v3-large --dataset $d --seed 1001 --seed 1002 --seed 1003 --seed 1004 --seed 1005 \
    --device cuda --run-id final_train
done
s1bench train --model tfidf-lr --dataset clinc150 --dataset hwu64 --dataset fpb --dataset hatexplain --dataset trec6 \
  --run-id final_train
```

## 4. Jev (hosted API; needs `.env`, see `.env.example`; all calls within 72 hours)

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
```

## 10. Paper tables and figures

```bash
python scripts/paper/make_tables.py
python scripts/paper/exp2_descriptives.py
python scripts/paper/make_figures.py
python scripts/paper/make_pipeline_figure.py
```

All scripts read only `results/final_*` and `data/` and write to `results/final_analysis/` and `paper/`.
